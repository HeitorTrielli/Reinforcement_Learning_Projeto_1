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

ALPHA = 0.6
BETA = 0.4

REWARD_SEARCH = 0.7
REWARD_WAIT = 0.5
REWARD_DEAD_BATERY = -3
REWARD_RECHARGE = 0.0

EPSILON = 0.3
DISCOUNT = 0.9


class RobotState:
    def __init__(self, battery_level):
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
        if self.battery_level == HIGH_BATTERY:
            return [SEARCH, WAIT]
        elif self.battery_level == LOW_BATTERY:
            return [SEARCH, WAIT, RECHARGE]
        elif self.battery_level == DEAD_BATTERY:
            return [RECHARGE]
        else:
            raise ValueError("Invalid battery level")

    def next_state(self, action: Literal[0, 1, 2]):
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
        if action == SEARCH:
            if self.battery_level == LOW_BATTERY and self.deplete_rng >= BETA:
                return DEAD_BATTERY_REWARD

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
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def backup(self):
        """
        Update the value estimation of the states in reverse order. Using the temporal difference method,
        the value of the previous state is updated based on the value of the next state.
        :return:
        """
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):
            state = states[i]
            td_error = self.greedy[i] * (
                self.estimations[states[i + 1]] - self.estimations[state]
            )
            self.estimations[state] += self.step_size * td_error

    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []

        valid_actions = state.get_valid_actions()

        values = []
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))

        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self):
        with open(
            "policy_%s.bin" % ("first" if self.symbol == 1 else "second"), "wb"
        ) as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open(
            "policy_%s.bin" % ("first" if self.symbol == 1 else "second"), "rb"
        ) as f:
            self.estimations = pickle.load(f)


def train(epochs, print_every_n=500):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print(
                "Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f"
                % (i, player1_win / i, player2_win / i)
            )
        player1.backup()
        player2.backup()
        judger.reset()
    player1.save_policy()
    player2.save_policy()


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.

if __name__ == "__main__":
    train(int(1e5))
    compete(int(1e3))
    play()
