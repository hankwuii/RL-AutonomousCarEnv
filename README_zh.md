
# 赛道车虚拟环境

* 比赛PPT链接: [https://docs.google.com/presentation/d/1J6RE2CaqmXGGYuwT6-_0erj9Yp3HIR30mk6lciRfBpY/edit?usp=sharing](https://docs.google.com/presentation/d/1J6RE2CaqmXGGYuwT6-_0erj9Yp3HIR30mk6lciRfBpY/edit?usp=sharing)
* 中文版本內容未經驗證，重要資訊請參考英文版本

## 比赛规则
* 请使用Pytorch完成比赛。
* 在期中比赛中，只会使用本文档提供的环境，请注意不要在提交的代码中包含其他第三方库。
* 期中比赛分两轮进行。第一轮将有10名选手晋级到第二轮。最终，只有一位选手能赢得冠军。
* 我们将使用 `scenarios/validation.yml` 中描述的传感器和设置。

## 如何提交你的代码和强化学习（RL）代理？
* 将所有模型的权重和代码放在 `agent` 文件夹中。
* 将 `agent` 文件夹压缩为 `zip 文件` 并上传到 moodle。
  > 提交位置将在未来提供。

## 如何验证你的代码和RL代理可以使用？
* 重新下载项目
    > 下文称之为 `新项目`
* 按照下面的说明重新创建一个新的虚拟环境
    > 此步骤确保您的代码可以在本项目提供的标准环境中执行
* 将您的 `agent` 文件夹放入新项目中
* 执行 `python validation_script.py`
* 如果可以正常执行，说明您的提交文件没有问题。

## 如何控制汽车并训练RL代理？
> !!!!!!!!! 请注意 !!!!!!!!  
> 场景: `validation.yml` 和 `validation2.yml` 不可用于训练  
> 如果使用它们进行训练，可能会导致未知错误  
> 这两个场景仅用于验证算法  
* 修改 `agent/Agent.py` 脚本中的 `Agent.get_action` 函数。
* 在 `Agent.get_action` 函数中实现你的RL算法以控制汽车。

## 如果无法在环境中运行该怎么办？
* 确保你的系统安装了合适的依赖项，特别是 Python 和必要的库。
* 查看系统日志，查找可能的错误提示，并尝试解决。

## 如何使用虚拟环境？
```python
import gymnasium as gym

# 使用默认场景:
env = gym.make(
    'RacecarScenario-v0',
    render_mode='human'
)

# 自定义场景:
env = gym.make(
    id='SingleAgentRaceEnv-v0',
    scenario='path/to/scenario',
    render_mode='rgb_array_follow',  # 可选
    render_options=dict(width=320, height=240, agent='A')  # 可选
)

done = False
reset_options = dict(mode='grid')
obs, info = env.reset(options=reset_options)

while not done:
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, states = env.step(action)
    done = terminated or truncated

env.close()
```

## 地图

当前可用的地图如下所列。这些网格地图原始来自 [F1Tenth](https://github.com/f1tenth) 仓库。

| 图片                                 | 名称     |
|--------------------------------------|----------|
| ![austria](docs/tracks/austria.png)  | 奥地利   |
| ![berlin](docs/tracks/berlin.png)    | 柏林     |
| ![montreal](docs/tracks/montreal.png)| 蒙特利尔 |
| ![torino](docs/tracks/torino.png)    | 都灵     |
| ![circle](docs/tracks/circle.png)    | 圆形赛道 |
| ![plechaty](docs/tracks/plechaty.png)| Plechaty |

---

## 故障排除
### 错误: Microsoft Visual C++ 14.0 或更高版本是必需的
1. 下载并安装 Visual Studio 2022 Installer ([这里](https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/))
2. 安装C++包， 如下所示
![img.png](docs/VisualStudioInstall.png)

3. 删除 racing 环境并重新安装
   ```bash
    # 卸载环境
    conda deactivate
    conda env remove -n racing
    # 重新安装环境
    conda env create -f environment.yml
    conda activate racing
    pip install -e .
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   ```

## 致谢
* 本项目修改自 [axelbr/racecar_gym](https://github.com/axelbr/racecar_gym.git)
