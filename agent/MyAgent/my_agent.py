import numpy as np


class MyAgent:
    def __init__(self):
        pass

    @staticmethod
    def obs_preprocess(obs: dict) -> np.ndarray:
        """
        Get action based on observation

        Args:
            obs: dict
                `{'rgb_image': ndarray(128, 128, 3), 'lidar': ndarray(1080,), 'pose': ndarray(6,), 'velocity': ndarray(6,), 'acceleration': ndarray(6,), time: ndarray(1,}`

        Returns: np.ndarray
            agent observation input

        """

        # TODO Make your own observation preprocessing

        return np.concatenate([obs['pose'], obs['velocity'], obs['acceleration'], obs['lidar']], axis=-1)

    def get_action(self, obs: dict) -> dict[str, float]:
        """
        Get action based on observation

        Args:
            obs: dict
                `{'rgb_image': ndarray(128, 128, 3), 'lidar': ndarray(1080,), 'pose': ndarray(6,), 'velocity': ndarray(6,), 'acceleration': ndarray(6,), time: ndarray(1,}`

        Returns: dict
            `{'motor': float,"steering": float}`

        """
        # TODO: Select action

        return {'motor': 1, 'steering': 1}

    def load_model(self) -> None:
        pass

    def save_model(self) -> None:
        pass
