#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import pickle
from typing import Literal

import numpy as np

SEARCH = 0
WAIT = 1
RECHARGE = 2

HIGH_BATTERY = 0
LOW_BATTERY = 1
DEAD_BATTERY = 2

ALPHA = 0.5
BETA = 0.5

REWARD_SEARCH = 0.8
REWARD_WAIT = 0.1
REWARD_DEAD_BATERY = -3
REWARD_RECHARGE = 0.0

EPSILON = 0.3
DISCOUNT = 0.9

PRINT_EVERY_N = 50
MAX_STEPS = 100


class RobotState:
    """Robot state with battery level and random outcomes."""

    def __init__(self, battery_level):
        """Initialize state with battery level."""
        self.battery_level = battery_level
        self.hash_val = None
        self.deplete_rng = np.random.random()
        self.reward_rng = np.random.random()

    def hash(self):
        """Compute unique hash for the state."""
        if self.hash_val is None:
            self.hash_val = hash(self.battery_level)
        return self.hash_val

    def get_valid_actions(self):
        """Return valid actions for current battery level."""
        if self.battery_level == HIGH_BATTERY:
            return [SEARCH, WAIT]
        elif self.battery_level == LOW_BATTERY:
            return [SEARCH, WAIT, RECHARGE]
        elif self.battery_level == DEAD_BATTERY:
            return [RECHARGE]
        else:
            raise ValueError("Invalid battery level")

    def next_state(self, action: Literal[0, 1, 2]):
        """Generate next state based on action and probabilities."""
        if action == SEARCH:
            if self.battery_level == HIGH_BATTERY:
                if self.deplete_rng < ALPHA:
                    new_state = RobotState(battery_level=HIGH_BATTERY)
                else:
                    new_state = RobotState(battery_level=LOW_BATTERY)
            elif self.battery_level == LOW_BATTERY:
                if self.deplete_rng < BETA:
                    new_state = RobotState(battery_level=LOW_BATTERY)
                else:
                    new_state = RobotState(battery_level=DEAD_BATTERY)
            elif self.battery_level == DEAD_BATTERY:
                raise ValueError("Cannot search with dead battery")

        elif action == WAIT:
            if self.battery_level == DEAD_BATTERY:
                raise ValueError("Cannot wait with dead battery")
            new_state = RobotState(battery_level=self.battery_level)

        elif action == RECHARGE:
            new_state = RobotState(battery_level=HIGH_BATTERY)

        else:
            raise ValueError("Invalid action")

        return new_state

    def get_reward(self, action: Literal[0, 1, 2]):
        """Calculate reward for taking action in this state."""
        if action == SEARCH:
            if self.battery_level == LOW_BATTERY and self.deplete_rng >= BETA:
                return REWARD_DEAD_BATERY

            if self.reward_rng < REWARD_SEARCH:
                return 1
            return 0

        elif action == WAIT:
            if self.reward_rng < REWARD_WAIT:
                return 1
            return 0

        elif action == RECHARGE:
            if self.reward_rng < REWARD_RECHARGE:
                return 1
            return 0
        else:
            raise ValueError("Invalid action")


class RobotAgent:
    """RL Agent using temporal difference learning with learned environment model."""

    def __init__(self, step_size=0.1, epsilon=EPSILON, discount=DISCOUNT):
        """Initialize agent with learning parameters."""
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.discount = discount
        self.states = []
        self.greedy = []

        self.transition_counts = {}
        self.reward_history = {}
        self.total_attempts = {}

        self.estimations[hash(HIGH_BATTERY)] = 0.5
        self.estimations[hash(LOW_BATTERY)] = 0.5
        self.estimations[hash(DEAD_BATTERY)] = 0.5

    def reset(self):
        """Reset episode data."""
        self.states = []
        self.greedy = []

    def set_state(self, state):
        """Add state to episode history."""
        self.states.append(state)
        self.greedy.append(True)

    def update_model(self, state, action, next_state, reward):
        """Update the learned environment model based on experience."""
        state_hash = state.hash()
        next_state_hash = next_state.hash()
        key = (state_hash, action)

        if key not in self.transition_counts:
            self.transition_counts[key] = {}
        if next_state_hash not in self.transition_counts[key]:
            self.transition_counts[key][next_state_hash] = 0
        self.transition_counts[key][next_state_hash] += 1

        if key not in self.reward_history:
            self.reward_history[key] = []

        self.reward_history[key].append(reward)

        self.total_attempts[key] = self.total_attempts.get(key, 0) + 1

    def get_transition_prob(self, state_hash, action, next_state_hash):
        """Get learned transition probability."""
        key = (state_hash, action)
        if key not in self.transition_counts:
            return 0.0

        total = self.total_attempts[key]
        count = self.transition_counts[key].get(next_state_hash, 0)
        return count / total if total > 0 else 0.0

    def get_expected_reward(self, state_hash, action):
        """Get expected reward for state-action pair."""
        key = (state_hash, action)
        if key not in self.reward_history or len(self.reward_history[key]) == 0:
            return 0.0
        return np.mean(self.reward_history[key])

    def get_expected_value(self, state, action):
        """Calculate expected value using learned model."""
        state_hash = state.hash()
        key = (state_hash, action)

        if key not in self.total_attempts or self.total_attempts[key] == 0:
            return 0.5

        expected_value = 0.0
        expected_reward = self.get_expected_reward(state_hash, action)

        if key in self.transition_counts:
            for next_state_hash, count in self.transition_counts[key].items():
                prob = count / self.total_attempts[key]
                next_state_value = self.estimations.get(next_state_hash, 0.0)
                expected_value += prob * (
                    expected_reward + self.discount * next_state_value
                )

        return expected_value

    def backup(self, episode_reward_history):
        """Update state values using temporal difference learning."""
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):
            state_hash = states[i]
            next_state_hash = states[i + 1]
            reward = episode_reward_history[i]

            td_error = self.greedy[i] * (
                reward
                + self.discount * self.estimations[next_state_hash]
                - self.estimations[state_hash]
            )  # V(s) = R + γV(s')
            self.estimations[state_hash] += self.step_size * td_error

    def act(self):
        """Choose action using epsilon-greedy policy."""
        state = self.states[-1]
        valid_actions = state.get_valid_actions()

        if np.random.rand() < self.epsilon:
            random_action = np.random.choice(valid_actions)
            self.greedy[-1] = False
            return random_action

        best_action = valid_actions[0]  # updated if better action exists
        best_value = self.get_expected_value(state, best_action)

        for action in valid_actions[1:]:
            value = self.get_expected_value(state, action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def save_policy(self):
        with open("robot_policy.bin", "wb") as f:
            data = {
                "estimations": self.estimations,
                "transition_counts": self.transition_counts,
                "reward_history": self.reward_history,
                "total_attempts": self.total_attempts,
            }
            pickle.dump(data, f)

    def load_policy(self):
        try:
            with open("robot_policy.bin", "rb") as f:
                data = pickle.load(f)
                self.estimations = data["estimations"]
                self.transition_counts = data["transition_counts"]
                self.reward_history = data["reward_history"]
                self.total_attempts = data["total_attempts"]
        except FileNotFoundError:
            print("Policy file not found. Starting with fresh policy.")

    def print_policy(self):
        """Print current policy for each state."""
        battery_names = {HIGH_BATTERY: "HIGH", LOW_BATTERY: "LOW", DEAD_BATTERY: "DEAD"}

        print("Policy Table:")
        print("State | SEARCH | WAIT   | RECHARGE")
        print("------|--------|--------|----------")

        for battery_level in [HIGH_BATTERY, LOW_BATTERY, DEAD_BATTERY]:
            state = RobotState(battery_level=battery_level)
            valid_actions = state.get_valid_actions()
            if not valid_actions:
                continue

            best_action = max(
                valid_actions, key=lambda a: self.get_expected_value(state, a)
            )

            row = f"{battery_names[battery_level]:5} |"

            for action in [SEARCH, WAIT, RECHARGE]:
                if action in valid_actions:
                    value = self.get_expected_value(state, action)
                    marker = "*" if action == best_action else " "
                    row += f" {value:5.2f}{marker} |"
                else:
                    row += "   --   |"

            print(row)


class CanCollectionJudger:
    """Manages game flow and coordinates agent-environment interaction."""

    def __init__(self, agent):
        self.agent = agent
        self.current_state = None
        self.total_episode_reward = 0
        self.steps = 0
        self.max_steps = MAX_STEPS

    def reset(self):
        """Reset environment for new episode."""
        self.agent.reset()
        self.current_state = RobotState(battery_level=HIGH_BATTERY)
        self.agent.set_state(self.current_state)
        self.total_episode_reward = 0
        self.steps = 0

    def play_episode(self):
        """Play complete episode and return rewards."""
        self.reset()
        episode_reward_history = []

        while self.steps < self.max_steps:
            action = self.agent.act()

            reward = self.current_state.get_reward(action)
            next_state = self.current_state.next_state(action)
            episode_reward_history.append(reward)

            self.agent.update_model(self.current_state, action, next_state, reward)

            self.current_state = next_state
            self.agent.set_state(self.current_state)

            self.total_episode_reward += reward
            self.steps += 1

        return self.total_episode_reward, episode_reward_history


def train(epochs, print_every_n=PRINT_EVERY_N):
    """Train robot agent using temporal difference learning."""
    agent = RobotAgent(epsilon=EPSILON)
    judger = CanCollectionJudger(agent)

    episode_total_reward_history = []

    for i in range(1, epochs + 1):
        episode_total_reward, episode_individual_reward_history = judger.play_episode()
        episode_total_reward_history.append(episode_total_reward)

        agent.backup(episode_individual_reward_history)

        if i % print_every_n == 0:
            avg_reward = (
                sum(episode_total_reward_history[-print_every_n:]) / print_every_n
            )
            print(f"\nEpoch {i}: Average reward = {avg_reward:.2f}")
            print(
                f"State Values: HIGH={agent.estimations.get(hash(HIGH_BATTERY), 0):.3f}, LOW={agent.estimations.get(hash(LOW_BATTERY), 0):.3f}, DEAD={agent.estimations.get(hash(DEAD_BATTERY), 0):.3f}"
            )

            agent.print_policy()

    agent.save_policy()

    print(
        f"Training completed! Final average reward: {sum(episode_total_reward_history[-100:]) / 100:.2f}"
    )


if __name__ == "__main__":
    train(1000)
