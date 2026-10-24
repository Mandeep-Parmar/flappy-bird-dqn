"""Training and evaluation logic for the Flappy Bird DQN agent.

This file connects the Gymnasium environment, DQN networks, replay memory,
training hyperparameters, model saving, and command-line options.
"""

import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse

# Choose the best available compute device.
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else :
    device = "cpu"

# Store training logs and saved model weights in this directory.
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    
    def __init__(self, param_set):
        # Load the named hyperparameter group from the YAML configuration.
        with open("parameters.yaml", "r") as f:
            all_params_set = yaml.safe_load(f)
            params = all_params_set[param_set]

        self.param_set = param_set

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        # Experience replay configuration.
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        # Training stop/synchronization settings.
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None   # Created when training starts; evaluation does not optimize.

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):
        """Run episodes in training mode or evaluate a saved model.

        Args:
            is_training: If True, learn and save model weights.
            render: If True, display the game window.
        """

        # Rendering is enabled only when requested (usually for evaluation).
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)

        num_states = env.observation_space.shape[0] # input dim
        num_actions = env.action_space.n # output dim

        # Policy network chooses actions using its current Q-value estimates.
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            # Store (state, action, next_state, reward, terminated) transitions.
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            # Target network provides more stable targets during Q-learning.
            target_dqn = DQN(num_states, num_actions).to(device)
            # copy the wt and bias values from policy => target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # we want to copy policy network to target network at every ten step during training
            steps = 0

            # Adam updates the policy network parameters using the loss gradient.
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            best_reward = float("-inf")  # Any first episode reward will beat this value.

        else:
            # Evaluation mode loads saved weights and disables training-specific layers.
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        # Keep starting episodes until the process is stopped or another stop condition is added.
        for episode in itertools.count():
            state, _ = env.reset()   # Start a fresh episode
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0
            terminated = False

            # Stop the episode when the environment ends, or during training if reward threshold is reached.
            while not terminated and (not is_training or episode_reward < self.reward_threshold):
                # Epsilon-greedy action selection:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()  # explore
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else: 
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploit

                # Apply the chosen action and receive the next observation and reward.
                next_state, reward, terminated, _, _ = env.step(action.item())
                episode_reward += reward

                # create tensors
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Save the transition so a random batch can be learned from later.
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                    # Start learning once the replay buffer has enough samples for a full batch.
                    if len(memory) > self.mini_batch_size:
                        # get sample
                        mini_batch = memory.sample(self.mini_batch_size)

                        # train our dqn
                        # 1 - get experience from our experience replay
                        # 2 - compute the target Q-value (y-true) - using target network
                        # 3 - y-pred using policy network 
                        # 4 - loss
                        # 5 - back propagation
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        # sync the network - Periodically copy policy weights to the target network.
                        if steps >= self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            steps = 0

                # Advance to the next state for the next interaction.
                state = next_state
                
            if is_training:
                print(
                    f"Episode={episode + 1}, "
                    f"Reward={episode_reward:.2f}, "
                    f"Epsilon={epsilon:.3f}"
                )
            else:
                print(f"Evaluation Episode={episode + 1}, Reward={episode_reward:.2f}")
                break

            if is_training: 
                # Epsilon decay - Gradually reduce random exploration, but never below epsilon_min.
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # Save a checkpoint whenever this episode achieves a new best reward.
                if episode_reward > best_reward:
                    log_msg = f"best reward = {episode_reward} for episode={episode+1}"

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")
                    
                    # Save only the model parameters (state dictionary), not the whole model object.
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # env.close() -> manually stop for training

        # for evaluation
        env.close()
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """Perform one DQN optimization step using a sampled replay batch."""

        # 1 - get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        # Stack individual tensors into batch tensors for vectorized network evaluation.
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        # env.step() returns terminated as a standard Python boolean (True / False). 
        # torch.stack only accepts a sequence of existing PyTorch tensors, 
        # so it raised: TypeError: expected Tensor as element 0 in argument 0, but got bool.
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        # 2 - calculate target Q-values - if termination = True => zero
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # 3 - calculate y_pred i.e. Q-value from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # 4 - compute loss
        loss = self.loss_fn(current_q, target_q)

        # 5 - optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    # parse command line inputs 
    parser = argparse.ArgumentParser(description="Train or Test model.")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)