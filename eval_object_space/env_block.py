import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np


class BlockStackingEnv(gym.Env):
    def __init__(self):
        super(BlockStackingEnv, self).__init__()

        # Connect to PyBullet
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        self.action_space = spaces.Discrete(
            5
        )  # 0: do nothing, 1-3: move left, right, forward, backward, 4: grab/release
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(13,), dtype=np.float32
        )

        self.block_start_positions = [
            [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.05]
            for _ in range(3)
        ]
        self.block_ids = [
            p.loadURDF("block.urdf", self.block_start_positions[i], [0, 0, 0, 1])
            for i in range(3)
        ]

        self.grasped_block = None
        self.point_position = [0, 0, 0.1]  # Initial position of the point
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

        self.grasped_block = None
        self.point_position = [0, 0, 0.1]
        self.total_timsteps = 0
        return self._get_observation(), {}

    def _get_observation(self):
        # Return point position, block positions, and grasped block info
        block_positions = [
            p.getBasePositionAndOrientation(block_id)[0] for block_id in self.block_ids
        ]
        return np.concatenate(
            [
                self.point_position,
                *block_positions,
                [1.0 if self.grasped_block else 0.0],
            ]
        )

    def step(self, action):
        # Perform action
        if action == 1:
            self.point_position[0] += 0.05
        elif action == 2:
            self.point_position[0] -= 0.05
        elif action == 3:
            self.point_position[1] += 0.05
        elif action == 4:
            self.point_position[1] -= 0.05
        elif action == 5:
            if self.grasped_block is None:
                # Check if the point is close enough to a block to grab it
                for block_id in self.block_ids:
                    block_pos, _ = p.getBasePositionAndOrientation(block_id)
                    if (
                        np.linalg.norm(
                            np.array(self.point_position) - np.array(block_pos)
                        )
                        < 0.1
                    ):
                        self.grasped_block = block_id
                        break
            else:
                # Release the grasped block
                self.grasped_block = None

        # enforce that we can't move the point outside the workspace
        self.point_position[0] = np.clip(self.point_position[0], -1, 1)
        self.point_position[1] = np.clip(self.point_position[1], -1, 1)
        self.point_position[2] = np.clip(self.point_position[2], 0.1, 1)

        # Move the grasped block along with the point
        if self.grasped_block is not None:
            p.resetBasePositionAndOrientation(
                self.grasped_block, self.point_position, [0, 0, 0, 1]
            )

        # Step simulation
        p.stepSimulation()

        # Calculate reward and check if the task is done
        reward = self._calculate_reward()
        done = self._is_done()

        return self._get_observation(), reward, self.total_timsteps >= 1000, done, {}

    def _calculate_reward(self):
        # Calculate the reward based on the vertical alignment of the blocks
        if self.grasped_block is not None:
            return 0.0  # No reward while holding a block

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
        if self.grasped_block is not None:
            return False  # Continue the task while holding a block

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
