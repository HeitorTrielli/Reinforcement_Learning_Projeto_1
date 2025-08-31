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

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

SEARCH = 0
WAIT = 1
RECHARGE = 2

HIGH_BATTERY = 0
LOW_BATTERY = 1
DEAD_BATTERY = 2

ALPHA = 0.7
BETA = 1


class RobotState:
    def __init__(self, battery_level):
        self.battery_level = battery_level
        self.hash_val = None

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
        else:
            return [RECHARGE]

    def next_state(self, action: Literal[0, 1, 2]):
        if action == SEARCH:
            if self.battery_level == HIGH_BATTERY:
                if np.random.random() < ALPHA:
                    new_state = RobotState(battery_level=HIGH_BATTERY)
                else:
                    new_state = RobotState(battery_level=LOW_BATTERY)
            elif self.battery_level == LOW_BATTERY:
                if np.random.random() < BETA:
                    new_state = RobotState(battery_level=LOW_BATTERY)
                else:
                    new_state = RobotState(battery_level=DEAD_BATTERY)

        elif action == WAIT:
            new_state = RobotState(battery_level=self.battery_level)

        elif action == RECHARGE:
            new_state = RobotState(battery_level=HIGH_BATTERY)

        else:
            raise ValueError("Invalid action")

        return new_state


class RobotAgent:
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.state = RobotState()

    def search(self):
        pass

    def wait(self):
        pass

    def recharge(self):
        pass

    def act(self):
        pass


# AI player
class Player:
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5

    # update value estimation
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

    # choose an action based on the state
    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(i, j, self.symbol).hash())

        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))
        # to select one of the actions of equal value at random due to Python's sort is stable
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


# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ["q", "w", "e", "a", "s", "d", "z", "x", "c"]
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // BOARD_COLS
        j = data % BOARD_COLS
        return i, j, self.symbol


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
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")


if __name__ == "__main__":
    train(int(1e5))
    compete(int(1e3))
    play()
