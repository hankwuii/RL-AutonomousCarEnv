from agent.DQN.dqn_agent import DQNAgent
from agent.PPO.ppo_agent import PPOAgent


def get_training_agent(agent_name: str = 'DQN'):
    if agent_name == 'DQN':
        agent = DQNAgent(
            state_dim=1080,
            lr=2e-4,
            gamma=0.99,
            epsilon=0.8,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=10000,
            batch_size=32,
            target_update=100,
            is_training=True
        )
    elif agent_name == 'PPO':
        agent = PPOAgent(
            state_dim=1080,
            action_dim=2,
            max_step=4000,
            gamma=0.99,
            lamb=0.95,
            lr=1e-4,
            is_training=True
        )
    else:
        raise NotImplementedError

    return agent


def get_valid_agent(agent_name: str = 'DQN'):
    if agent_name == 'DQN':
        agent = DQNAgent(
            state_dim=1080,
            lr=2e-4,
            gamma=0.99,
            epsilon=0.8,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=10000,
            batch_size=32,
            target_update=1000,
            is_training=False
        )
    elif agent_name == 'PPO':
        agent = PPOAgent(
            state_dim=1080,
            action_dim=2,
            max_step=4000,
            gamma=0.99,
            lamb=0.95,
            lr=1e-4,
            is_training=False
        )
    else:
        raise NotImplementedError

    return agent
