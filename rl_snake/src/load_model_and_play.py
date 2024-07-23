import argparse

from rl_snake.src.deep_q_network import DQNAgent, play_game_with_dqn_agent

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load a DQN model and play the game.")
    parser.add_argument(
        "--model_path", type=str, help="Path to the DQN model checkpoint"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Create DQNAgent and load the model
    agent = DQNAgent(state_size=12, action_size=4)
    agent.load(args.model_path)

    # Play game with the loaded agent
    play_game_with_dqn_agent(agent)
