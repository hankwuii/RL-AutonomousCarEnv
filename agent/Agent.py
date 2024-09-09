from agent.DQN.dqn_agent import DQNAgent


def get_training_agent():
    agent = DQNAgent(
        state_dim=1080,
        lr=3e-4,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        memory_size=30000,
        batch_size=32,
        target_update=50,
        is_training=True
    )

    return agent


def get_valid_agent():
    agent = DQNAgent(
        state_dim=1080,
        lr=3e-4,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        memory_size=30000,
        batch_size=32,
        target_update=50,
        is_training=False
    )

    return agent
