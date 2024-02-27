import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np


class HoopGripperPushEnv(gym.Env):
    def __init__(self):
        super(HoopGripperPushEnv, self).__init__()

        # Connect to PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # we can move a hoop x, y
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        # we know the xy position of the the hoop and the block, and the target
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(11,), dtype=np.float32
        )

        # Initialize a block with a random position
        self.block_id = p.loadURDF(
            "block.urdf", np.concatenate([np.random.uniform(-1, 1, 2), [0.1]])
        )
        self.hoop_id = p.loadURDF("hoop.urdf", [0, 0, 0.1])

        self.target_position = np.random.uniform(-1, 1, 2)

        self.hoop_grasped = False
        self.point_position = [0, 0]  # Initial position of the point
        self.total_timesteps = 0

    def reset(self, seed=None, options=None):
        # set the seed
        del options
        if seed is not None:
            np.random.seed(seed)
        self.target_position = np.random.uniform(-1, 1, 2)
        # Reset block position
        p.resetBasePositionAndOrientation(
            self.block_id,
            np.concatenate([np.random.uniform(-1, 1, 2), [0.1]]),
            [0, 0, 0, 1],
        )
        # Reset hoop position
        p.resetBasePositionAndOrientation(self.hoop_id, [0, 0, 0.1], [0, 0, 0, 1])

        self.hoop_grasped = False
        self.point_position = [0, 0]
        self.total_timesteps = 0
        return self._get_observation(), {}

    def _get_observation(self):
        # Return point position, block positions, and grasped block info
        return np.concatenate(
            [
                self.target_position,
                self.point_position,
                p.getBasePositionAndOrientation(self.block_id)[0],
                p.getBasePositionAndOrientation(self.hoop_id)[0],
                [1.0 if self.hoop_grasped else -1.0],
            ]
        )

    def step(self, action):
        self.total_timesteps += 1
        # Perform action. action is continuous box form -1 to 1,
        # and it's how much we move the point in the x, y direction
        self.point_position[0] += action[0]
        self.point_position[1] += action[1]
        if action[2] > 0:
            # If action[2] is positive, we toggle grasp of the hoop
            if not self.hoop_grasped:
                # If the hoop is not grasped, we check if the point is close enough
                # to the hoop to grasp it
                if (
                    np.linalg.norm(
                        np.array(self.point_position)
                        - np.array(p.getBasePositionAndOrientation(self.hoop_id)[0])[:2]
                    )
                    < 0.1
                ):
                    self.hoop_grasped = True
            else:
                # If the hoop is already grasped, we release it
                self.hoop_grasped = False

        # enforce that we can't move the point outside the workspace
        self.point_position = np.clip(self.point_position, -1, 1)

        # Move the grasped hoop along with the point
        if self.hoop_grasped:
            # don't use resetBasePositionAndOrientation to move the hoop,
            # because we also want to push the block if it's in contact
            p.setBasePositionAndOrientation(
                self.hoop_id,
                np.concatenate([self.point_position, [0.1]]),
                [0, 0, 0, 1],
            )

        # Step simulation
        p.stepSimulation()

        # Calculate reward and check if the task is done
        reward = self._calculate_reward()
        done = self._is_done()

        return self._get_observation(), reward, self.total_timesteps >= 1000, done, {}

    def _calculate_reward(self):
        # reward is block proximity to target point
        block_pos = p.getBasePositionAndOrientation(self.block_id)[0]
        return -np.linalg.norm(block_pos - np.concatenate([self.target_position, [0.1]]))

    def _is_done(self):
        # We are done if the block is close enough to the target position
        block_pos = p.getBasePositionAndOrientation(self.block_id)[0]
        return (
            np.linalg.norm(block_pos - np.concatenate([self.target_position, [0.1]]))
            < 0.1
        )

    def render(self, mode="human"):
        pass  # You can add visualization if needed

    def close(self):
        p.disconnect()
