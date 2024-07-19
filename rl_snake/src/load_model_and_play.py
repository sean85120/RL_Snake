from rl_snake.src.deep_q_network import DQNAgent, play_game_with_dqn_agent

if __name__ == "__main__":
    agent = DQNAgent(state_size=12, action_size=4)
    agent.load("./model_checkpoints/dqn_snake_model.pth")
    play_game_with_dqn_agent(agent)
