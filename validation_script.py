import threading
from threading import Lock, Thread
from time import sleep
import sys
import os
import sys
import time

import cv2
import gymnasium
import numpy as np
from matplotlib import pyplot as plt

import racecar_gym.envs.gym_api  # Necessary!!!Cannot be deleted!!!

# add private agent path


if 'racecar_gym.envs.gym_api' not in sys.modules:
    raise RuntimeError('Please run: pip install -e . and import racecar_gym.envs.gym_api first!')


list_frames = []
flag = True
def write_frame_to_video(writer):
    """將影像寫入視頻的線程函數"""
    while flag or len(list_frames) != 0:
        if len(list_frames) == 0:
            sleep(0.1)
            continue
        frame = list_frames.pop(0)
        writer.write(frame)


def main():
    global flag
    from agent.Agent import get_valid_agent

    # ======================================================================
    # Create the environment
    # ======================================================================
    render_mode = 'rgb_array_birds_eye'  # 'human', 'rgb_array_birds_eye' and 'rgb_array_follow'
    env = gymnasium.make(
        'SingleAgentAustria-v0',
        render_mode=render_mode,
        # scenario='scenarios/circle_cw.yml',  # change the scenario here (change map)
        scenario=f'{os.path.dirname(__file__)}/scenarios/{_map}.yml',  # change the scenario here (change map)
        # scenario='scenarios/validation.yml',  # change the scenario here (change map), ONLY USE THIS FOR VALIDATION
        # scenario='scenarios/validation2.yml',   # Use this during the midterm competition, ONLY USE THIS FOR VALIDATION
    )
    done = False
    agent = get_valid_agent(agent_model)
    # video_path = fr"Z:\Student_Work\_Share\RL競賽影片\{_map}\{student_id}.mp4"
    video_path = fr"{save_root}/{_map}/{student_id}_{name}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    output_video_writer = cv2.VideoWriter(filename=video_path,
                                          fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                          fps=20,
                                          frameSize=(640, 480))
    flag = True
    thread = threading.Thread(target=write_frame_to_video, args=(output_video_writer,))
    thread.start()

    print(f"student_id: {student_id}, name: {name}")
    print("save at", video_path)
    # ======================================================================
    # Run the environment
    # ======================================================================
    obs, info = env.reset(options=dict(mode='grid'))
    t = 0
    while not done:
        # ==================================
        # Execute RL model to obtain action
        # ==================================
        action = agent.get_action(obs)
        obs, rewards, done, truncated, states = env.step(action)

        if t % 5 == 0 and "rgb" in render_mode:
            # ==================================
            # Render the environment
            # ==================================

            image = env.render().astype(np.uint8)

            text = f"Student ID: {student_id}"
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

            # 將影像寫入視頻
            list_frames.append(image)
            # output_video_writer.write(image)

            if t % 30 == 0:
                # 顯示影像
                cv2.imshow(f"{student_id}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        # if t > 100:
        #     break

        t += 1
        if done or truncated:
            break

        print(f"\r{t} / 12000", "save frame", len(list_frames), end="")

    flag = False
    thread.join()

    env.close()
    output_video_writer.release()
    cv2.destroyAllWindows()
    print(f"student_id: {student_id}, name: {name}")
    print("finish", video_path)


if __name__ == '__main__':
    import re

    # ======================================================================

    # # AGENT_ROOT_DIR = f"Z:\Student_Work\_Share\RL鴻耀\M11352012@ 鄭語萱_931492_assignsubmission_file\M11352012\M11352012"
    # # AGENT_ROOT_DIR = f"Z:\Student_Work\_Share\RL鴻耀\M11352013@ 王冠文_931494_assignsubmission_file\M11352013\M11352013"  # PPO  # model path error，還有一個錯的weight.pth讓人混淆
    # # AGENT_ROOT_DIR = f"Z:\Student_Work\_Share\RL鴻耀\M11352015@ 黃梓軒_931488_assignsubmission_file\m11352015\m11352015"  # PPO
    # # AGENT_ROOT_DIR = f"Z:\Student_Work\_Share\RL鴻耀\M11352016@ 謝昕諺_931515_assignsubmission_file\m11352016\m11352016",  # PPO  # 有幫他再跑一次
    # # AGENT_ROOT_DIR = f"Z:\Student_Work\_Share\RL鴻耀\M11352017@ 陳威丞_931516_assignsubmission_file\M11352017\M11352017"  # PPO
    #
    # DIR_DICT = {
    #     # f"Z:\Student_Work\_Share\RL鴻耀\M11352018@ 賴立恩_931486_assignsubmission_file\M11352018\M11352018": "PPO",  # PPO  # 檔案給錯，沒有完成作業+model weight沒給，但影片是跑得不錯的
    #     # f"Z:\Student_Work\_Share\RL鴻耀\M11352020@ 余東樺_931496_assignsubmission_file\m11352020": "PPO",  # PPO
    #     # f"Z:\Student_Work\_Share\RL鴻耀\M11352021@ 徐瑞廷_931498_assignsubmission_file\m11352021\m11352021": "PPO",  # PPO
    #     # f"Z:\Student_Work\_Share\RL鴻耀\M11352023@ 徐彥崴_931506_assignsubmission_file\m11352023\m11352023": "PPO",  # PPO， 沒有影片
    #     # f"Z:\Student_Work\_Share\RL鴻耀\M11352026@ 曹濟_931508_assignsubmission_file\M11352026\M11352026": "PPO",  # PPO
    #    # f"Z:\Student_Work\_Share\RL鴻耀\M11352029@ 黃浚廷_931490_assignsubmission_file\M11352029\M11352029": "PPO",  # PPO
    #     # f"Z:\Student_Work\_Share\RL鴻耀\M11352030@ 詹岫尹_931512_assignsubmission_file\M11352030\M11352030": "DQN",  # DQN
    #     # f"Z:\Student_Work\_Share\RL鴻耀\M11352031@ 羅子程_931503_assignsubmission_file\M11352031\M11352031": "PPO",  # PPO
    #     # f"Z:\Student_Work\_Share\RL鴻耀\M11352034@ 歐銘耘_931487_assignsubmission_file\m11352034\m11352034": "PPO",  # PPO
    #     f"Z:\Student_Work\_Share\RL鴻耀\M11352035@ 吳冠霖_931491_assignsubmission_file\m11352035": "PPO",  # PPO
    # }

    # ============================================================================

    DIR_DICT = {
        # f"Z:\Student_Work\_Share\RL鴻耀\M11352016@ 謝昕諺_931515_assignsubmission_file\m11352016\m11352016": "PPO",  # PPO
        # f"Z:\Student_Work\_Share\RL鴻耀\M11352026@ 曹濟_931508_assignsubmission_file\M11352026\M11352026": "PPO",  # PPO

        f"Z:\Student_Work\_Share\RL鴻耀\m11352802@ Alexandre_\m11352802": "PPO",
    }


    for AGENT_ROOT_DIR, agent_model in DIR_DICT.items():
        print("NOW: ", AGENT_ROOT_DIR)
        save_root = fr"E:\python\racecar_gym\videos"
        _map = "validation1"

        # --- dont change ---
        student_id = AGENT_ROOT_DIR.split("\\")[-1]
        name = re.search(r"@ (.*?)_", AGENT_ROOT_DIR).group(1)

        os.chdir(AGENT_ROOT_DIR)
        sys.path.append(AGENT_ROOT_DIR)

        try:
            main()
        except Exception as e:
            print(e)
            print("Error", AGENT_ROOT_DIR)

        # remove AGENT_ROOT_DIR from sys.path
        sys.path.remove(AGENT_ROOT_DIR)

        print("finish", AGENT_ROOT_DIR)
