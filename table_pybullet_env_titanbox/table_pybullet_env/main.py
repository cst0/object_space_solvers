#!/usr/bin/env python

"""
Load the scene.urdf into pybullet and run the simulation.
"""

import pybullet as p
import random
import time
import pybullet_data
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from typing import List

import gymnasium as gym
import numpy as np

MODE = p.DIRECT


class ActionSpaceEnv(gym.Env):
    def __init__(self):
        self.ids, self.indices, self.physics_client = self.setup_pybullet(MODE)

        self.goal_xyz = np.array(
            [0, 0.2, 0.3]
        )  # somewhere over the center of the table
        self.drawer_open = False

        self.action_space_list = self.construct_action_space()
        self.action_space = gym.spaces.Discrete(len(self.action_space_list))
        self.observation = self.get_observation()
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=self.observation.shape, dtype=self.observation.dtype
        )

    def setup_pybullet(self, mode):
        # Connect to the physics server
        physics_client = p.connect(mode)

        # Add search path for loadURDFs, load the URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        _ = p.loadURDF("plane.urdf")

        current_path = os.path.abspath(os.path.dirname(__file__))
        urdf_path = os.path.join(current_path, "urdfs")
        p.setAdditionalSearchPath(urdf_path)
        scene_id = p.loadURDF("scene.urdf", useFixedBase=True)
        cube_id = p.loadURDF("cube.urdf", useFixedBase=True, basePosition=[0, 0, 0.69])

        # load manipulatable joints
        scene_joint_ids = []
        for i in range(p.getNumJoints(scene_id)):
            joint_info = p.getJointInfo(scene_id, i)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                scene_joint_ids.append(i)

        cube_joint_ids = []
        for i in range(p.getNumJoints(cube_id)):
            joint_info = p.getJointInfo(cube_id, i)
            joint_type = joint_info[2]
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                cube_joint_ids.append(i)

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)
        p.setRealTimeSimulation(0)

        return [scene_id, cube_id], [scene_joint_ids, cube_joint_ids], physics_client

    def test_movements(self):
        # test movements by picking a random velocity for each joint and playing it out.
        # we should see a bunch of valid movements.
        pairs = []
        for n in range(len(self.ids)):
            for i in self.indices[n]:
                pairs.append((self.ids[n], i))
        while True:
            for pair in pairs:
                choices = [-10, 0, 10]
                random_choice = random.choice(choices)
                p.setJointMotorControl2(
                    pair[0],
                    pair[1],
                    p.VELOCITY_CONTROL,
                    targetVelocity=random_choice,
                    force=100,
                )
            for i in range(100):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

    def get_observation(self):
        # first, find the position/velocity of each possible joint
        current_joint_positions = []
        current_joint_velocities = []
        min_joint_positions = []
        min_joint_velocities = []
        max_joint_positions = []
        max_joint_velocities = []
        self.pairings = []
        for n in range(len(self.ids)):
            for i in self.indices[n]:
                current_joint_positions.append(p.getJointState(self.ids[n], i)[0])
                min_joint_positions.append(p.getJointInfo(self.ids[n], i)[8])
                max_joint_positions.append(p.getJointInfo(self.ids[n], i)[9])
                self.pairings.append((self.ids[n], i))

                current_joint_velocities.append(p.getJointState(self.ids[n], i)[1])
                max_joint_velocities.append(p.getJointInfo(self.ids[n], i)[11])
                min_joint_velocities.append(-max_joint_velocities[-1])

        # scale to be between -1 and 1
        try:
            scaled_current_joint_positions = [
                (float(x) - float(min_joint_positions[i]))
                / (float(max_joint_positions[i]) - float(min_joint_positions[i]))
                for i, x in enumerate(current_joint_positions)
            ]
            scaled_current_joint_velocities = [
                (float(x) - float(min_joint_velocities[i]))
                / (float(max_joint_velocities[i]) - float(min_joint_velocities[i]))
                for i, x in enumerate(current_joint_velocities)
            ]
        except (TypeError, ValueError) as e:
            print(
                e,
                current_joint_positions,
                min_joint_positions,
                max_joint_positions,
                current_joint_velocities,
                min_joint_velocities,
                max_joint_velocities,
            )
            raise e

        # convert to numpy array
        scaled_current_joint_positions = np.array(scaled_current_joint_positions)
        scaled_current_joint_velocities = np.array(scaled_current_joint_velocities)
        observations = np.concatenate(
            (scaled_current_joint_positions, scaled_current_joint_velocities)
        )

        return observations

    def step(self, action):
        as_action = self.action_space_list[int(action)]  # type:ignore
        self.execute_action(as_action)

        self.observation = self.get_observation()
        self.reward = self.get_reward()
        self.terminated = self.get_terminated()
        self.truncated = self.get_truncated()
        self.info = self.get_info()

        return (
            self.observation,
            self.reward,
            bool(self.terminated),
            self.truncated,
            self.info,
        )

    def reset(self, seed=None, options=None):
        random.seed(seed)
        del options  # unused

        # reset the joints
        for n in range(len(self.ids)):
            for i in self.indices[n]:
                p.resetJointState(self.ids[n], i, targetValue=0, targetVelocity=0)

        # reset the observation
        self.observation = self.get_observation()

        return (self.observation, {})

    def construct_action_space(self):
        # construct the action space for the scene and cube
        # the action space is a list of tuples of the form (id, index, velocity)
        # where id is the body id, index is the joint index, and velocity is
        # the velocity of the joint
        action_space = []
        for n in range(len(self.ids)):
            for i in self.indices[n]:
                for v in [0.1, -0.1, 0]:
                    if self.ids[n] == 1:  # cube
                        validation_function = (
                            self.get_drawer_open
                        )  # cube is only valid to move if drawer is open
                    action_space.append((self.ids[n], i, v, None))
        return action_space

    def get_drawer_open(self, id_, index, velocity, client, ids):
        return self.drawer_open

    def execute_action(self, action, step_count=100):
        """
        Take an action in the environment
        Action is a tuple of the form (id, index, velocity, validation_function), where:
        - id is the body id
        - index is the joint index
        - velocity is the velocity of the joint
        - validation_function is a function that takes in the current state of the
            environment and returns a boolean if the requested action is valid, or
            validation_function is None to indicate always valid
        """
        id_, index, velocity, validation_function = action

        valid = True
        if validation_function is not None:
            valid = validation_function(
                id_, index, velocity, self.physics_client, self.ids
            )

        if valid:
            p.setJointMotorControl2(
                id_, index, p.VELOCITY_CONTROL, targetVelocity=velocity, force=100
            )
            for _ in range(step_count):
                p.stepSimulation()
            # then stop the joint
            p.setJointMotorControl2(
                id_, index, p.VELOCITY_CONTROL, targetVelocity=0, force=100
            )
            p.stepSimulation()
            return True
        else:
            return False

    def test_actions(self):
        action_space = self.construct_action_space()
        while True:
            for a in action_space:
                print(a)
                self.execute_action(a)
            print("done")

    def spin(self):
        while True:
            for _ in range(50):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

            print(
                self.get_reward(),
                self.get_body_state("cube"),
                self.get_body_state("top_drawer"),
                self.get_body_state("bottom_drawer"),
            )

    def get_body_state(self, body: str) -> List[float]:
        if body == "cube":
            id_ = self.ids[1]
            # get cube prismatic joint positions
            cube_x = p.getJointState(id_, 0)[0]
            cube_y = p.getJointState(id_, 1)[0]
            cube_z = p.getJointState(id_, 2)[0]
            return [cube_x, cube_y, cube_z]
        elif body == "top_drawer":
            id_ = self.ids[0]
            # get top drawer prismatic joint position
            drawer_x = p.getJointState(id_, 3)[0]
            return [drawer_x]
        elif body == "bottom_drawer":
            id_ = self.ids[0]
            # get bottom drawer prismatic joint position
            drawer_x = p.getJointState(id_, 2)[0]
            return [drawer_x]
        else:
            raise ValueError(f"Unsupported body: {body}")

    def get_reward(self):
        cube_xyz = self.get_body_state("cube")
        drawer_state = True if self.get_body_state("top_drawer")[0] > 0.3 else False
        reward = -np.linalg.norm(cube_xyz - self.goal_xyz)

        if drawer_state:
            self.drawer_open = True
            reward += 0.1
        else:
            self.drawer_open = False

        return reward

    def get_truncated(self):
        return False

    def get_terminated(self):
        cube_xyz = self.get_body_state("cube")
        terminated = (
            np.linalg.norm(cube_xyz - self.goal_xyz) < 0.05
        )  # cube is within 5cm of goal
        assert terminated in [True, False]
        return terminated

    def get_info(self):
        return {}

    def render(self, mode="human"):
        # instead of p.DIRECT, use p.GUI for graphical version
        # p.GUI is much slower than p.DIRECT
        p.disconnect()
        p.connect(p.DIRECT)


class TrainOnPlaceCube:
    def __init__(
        self, save_file: str, total_num_timesteps: int = 100000, save_every: int = 100
    ):
        # register with gymnasium
        gym.register(
            id="ActionSpaceEnv-v0",
            entry_point="main:ActionSpaceEnv",
        )

        # check environment
        #self.env = gym.make("gymnasium:ActionSpaceEnv-v0")
        #n_cpu = 16
        #self.env = SubprocVecEnv([lambda: gym.make("gymnasium:ActionSpaceEnv-v0")]*32)

        self.env = gym.make("gymnasium:ActionSpaceEnv-v0")
        #self.env = make_vec_env(lambda: make_env(), n_envs=n_cpu)
        self.model = PPO("MlpPolicy", self.env, verbose=1)

        for n in range(save_every):
            print("Training iteration", n)
            self.model.learn(total_timesteps=int(total_num_timesteps / save_every))
            self.model.save(f"{save_file}_{n}.zip")


class RunOnPlaceCube:
    def __init__(self):
        # register with gymnasium

        def gen_env():
            gym.register(
                id="ActionSpaceEnv-v0",
                entry_point="main:ActionSpaceEnv",
            )

            return gym.make("gymnasium:ActionSpaceEnv-v0")

        # check environment
        # check_env(self.env)
        # self.env = DummyVecEnv([lambda: self.env])
        # self.env = make_vec_env(lambda: gen_env(), n_envs=32)
        self.env = gen_env()
        self.model = PPO.load("ppo_action_space")
        obs = self.env.reset()
        while True:
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)


class RunEnv:
    def __init__(self):
        a = ActionSpaceEnv()
        a.reset()
        a.spin()


if __name__ == "__main__":
   train = TrainOnPlaceCube(
       f"policies/ppo_action_space_iter_{sys.argv[1]}",
       total_num_timesteps=50000,
       save_every=200
   )
