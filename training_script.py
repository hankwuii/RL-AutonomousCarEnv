from time import sleep
import gymnasium
import racecar_gym.envs.gym_api
from matplotlib import pyplot as plt
# import cv2


def main():
    # ======================================================================
    # Create the environment
    # ======================================================================
    render_mode = 'rgb_array_follow'  # 'human', 'rgb_array_birds_eye' and 'rgb_array_follow'
    env = gymnasium.make(
        'SingleAgentAustria-v0',
        render_mode=render_mode,
        scenario='scenarios/validation.yml',  # change the scenario here (change map)
    )
    done = False

    # ======================================================================
    # Run the environment
    # ======================================================================
    obs = env.reset(options=dict(mode='grid'))
    t = 0
    while not done:
        # ==================================
        # Execute RL model to obtain action
        # ==================================
        action = env.action_space.sample()
        obs, rewards, done, truncated, states = env.step(action)

        if t % 30 == 0 and "rgb" in render_mode:
            # ==================================
            # Render the environment
            # ==================================

            image = env.render()
            # cv2.imshow("image", image)
            # cv2.waitKey()
            plt.clf()
            plt.title("Pose")
            plt.imshow(image)
            plt.pause(0.01)
            plt.ioff()

        t += 1
        if done or truncated:
            break

    env.close()


if __name__ == '__main__':

    main()
