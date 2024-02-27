import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np


class BlockStackingEnv(gym.Env):
    def __init__(self):
        super(BlockStackingEnv, self).__init__()

        # get pybullet data path
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")

        # Initialize three blocks with random xy positions, but set
        # z to 0.05
        self.block_start_positions = [
            [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.05]
            for _ in range(3)
        ]
        self.block_ids = [
            p.loadURDF("block.urdf", self.block_start_positions[i], [0, 0, 0, 1])
            for i in range(3)
        ]

        # Define action space: discrete control of each block's movement
        self.action_space = spaces.Discrete(
            6 * len(self.block_ids)
        )  # up down left right forward backward

        # Observation space: block positions
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(9,), dtype=np.float32
        )

        self.total_timsteps = 0

    def reset(self, seed=None, options=None):
        # set the seed
        del options
        if seed is not None:
            np.random.seed(seed)
        # Reset block positions
        for i in range(3):
            p.resetBasePositionAndOrientation(
                self.block_ids[i], self.block_start_positions[i], [0, 0, 0, 1]
            )

        self.total_timsteps = 0
        return self._get_observation(), {}

    def _get_observation(self):
        # Return block positions info
        block_positions = [
            p.getBasePositionAndOrientation(block_id)[0] for block_id in self.block_ids
        ]
        return np.concatenate([*block_positions])

    def step(self, action):
        self.total_timsteps += 1
        # Convert discrete action to continuous actions for each block.
        # Each block has 6 possible actions: move up, down, left, right, forward, backward
        discrete_actions = []
        for _ in range(len(self.block_ids)):
            discrete_actions.append(action % 6)
            action //= 6

        # Apply actions to move the blocks
        for i, block_id in enumerate(self.block_ids):
            dx, dy, dz = 0.0, 0.0, 0.0

            if discrete_actions[i] == 0:
                dz = 0.1
            elif discrete_actions[i] == 1:
                dz = -0.1
            elif discrete_actions[i] == 2:
                dx = -0.1
            elif discrete_actions[i] == 3:
                dx = 0.1
            elif discrete_actions[i] == 4:
                dy = -0.1
            elif discrete_actions[i] == 5:
                dy = 0.1

            new_pos = p.getBasePositionAndOrientation(block_id)[0] + np.array(
                [dx, dy, dz]
            )
            # enforce the block to stay within the workspace
            new_pos = np.clip(new_pos, -1, 1)
            p.resetBasePositionAndOrientation(block_id, new_pos, [0, 0, 0, 1])

        # Step simulation
        p.stepSimulation()

        # Calculate reward and check if the task is done
        reward = self._calculate_reward()
        done = self._is_done()

        return self._get_observation(), reward, self.total_timsteps >= 1000, done, {}

    def _calculate_reward(self):
        total_reward = 0.0

        for i in range(len(self.block_ids) - 1):
            # Calculate the vertical distance between each pair of blocks
            z_diff = (
                p.getBasePositionAndOrientation(self.block_ids[i])[0][2]
                - p.getBasePositionAndOrientation(self.block_ids[i + 1])[0][2]
            )
            total_reward += max(
                0.0, 0.1 - abs(z_diff)
            )  # Encourage blocks to be vertically aligned

        return total_reward

    def _is_done(self):
        # Check if all blocks are stacked
        for i in range(len(self.block_ids) - 1):
            # Check if each pair of adjacent blocks is vertically aligned
            z_diff = abs(
                p.getBasePositionAndOrientation(self.block_ids[i])[0][2]
                - p.getBasePositionAndOrientation(self.block_ids[i + 1])[0][2]
            )
            if z_diff > 0.1:  # Adjust the threshold as needed
                return False  # Not done if any pair is not vertically aligned

        return True  # All blocks are stacked, task is done

    def render(self, mode="human"):
        pass  # You can add visualization if needed

    def close(self):
        p.disconnect()
