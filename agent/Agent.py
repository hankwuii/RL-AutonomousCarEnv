class Agent():
    def __init__(self):
        pass

    def get_action(self, obs: dict) -> dict:
        """

        Args:
            obs: dict
                `{'rgb_image': ndarray(128, 128, 3), 'lidar': ndarray(1080,), 'pose': ndarray(6,), 'velocity': ndarray(6,), 'acceleration': ndarray(6,), time: ndarray(1,}`

        Returns: dict
            `{'motor': float,"steering": float}`

        """

        # TODO Your algorithm here

        return {"motor": 1.0, "steering": 0.0}
