import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np


class HoopObjectEnv(gym.Env):
    def __init__(self):
        super(HoopObjectEnv, self).__init__()

        # Connect to PyBullet
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # we can move a point x, y in the workspace
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # we know the xy position of the point, the hoop, and the block
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )

        # Initialize a block with a random position
        self.block_position = [0,0,.1]
        self.block_id = p.loadURDF(
            "block.urdf", self.block_position, [0, 0, 0, 1]
        )
        self.hoop_id = p.loadURDF("hoop.urdf", [1, 1, 0.1])
        self.target_position = [-1,-1]

        self.total_timesteps = 0

    def reset(self, seed=None, options=None):
        # set the seed
        del options
        if seed is not None:
            np.random.seed(seed)
        # Reset block position
        p.resetBasePositionAndOrientation(
            self.block_id,
            self.block_position,
            [0, 0, 0, 1],
        )
        # Reset hoop position
        p.resetBasePositionAndOrientation(self.hoop_id, [0, 0, 0.1], [0, 0, 0, 1])
        p.resetBaseVelocity(
            self.hoop_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
        )
        self.total_timesteps = 0
        p.stepSimulation()
        return self._get_observation(), {}

    def _get_observation(self):
        # Return point position, block positions, and grasped block info
        return np.concatenate(
            [
                self.target_position[:2],
                p.getBasePositionAndOrientation(self.block_id)[0][:2],
                p.getBasePositionAndOrientation(self.hoop_id)[0][:2],
            ]
        )

    def step(self, action):
        self.total_timesteps += 1
        # Perform action. action is continuous box form -1 to 1,
        # and it's how much we move the point in the x, y direction
        p.resetBaseVelocity(
            self.hoop_id,
            linearVelocity=[action[0], action[1], 0],
            angularVelocity=[0, 0, 0],
        )

        # enforce that we can't move the hoop outside the workspace
        hoop_pos = p.getBasePositionAndOrientation(self.hoop_id)[0]
        hoop_pos = np.clip(hoop_pos, -1, 1)
        if np.any(hoop_pos != p.getBasePositionAndOrientation(self.hoop_id)[0]):
            p.resetBasePositionAndOrientation(self.hoop_id, hoop_pos, [0, 0, 0, 1])

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
