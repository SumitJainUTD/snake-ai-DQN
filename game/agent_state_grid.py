import json
import os

import torch
import random
import numpy as np
from collections import deque

from game.model import Linear_QNet, QTrainer
from game.snake import Game

MAX_MEMORY = 150_000
LEARNING_STARTS = 1000
BATCH_SIZE = 64
LR = 0.0001
TARGET_UPDATE_INTERVAL = 4000
BLOCK_WIDTH = 40


class Agent:
    def __init__(self, state_size, width, height):
        self.n_games = 0
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(state_size, 512, 3)
        self.target_model = Linear_QNet(state_size, 512, 3)
        self.trainer = QTrainer(self.model, self.target_model, LR, self.gamma)
        self.data_file = 'data.json'
        self.record = 0
        self.avg = 0
        self.total_eats = 0
        self.model_folder_path = './model'
        self.t_step = 0
        self.state_size = state_size
        self.width = width
        self.height = height
        # model, trainer

        self.random_count = 0
        self.training_count = 0
        self.total_decisions = 0

        pass

    def get_state(self, game):

        # Initialize the grid with zeros
        grid = np.zeros((self.width, self.height))

        # Mark the snake body on the grid
        for i in range(0, len(game.snake.x)):
            x = int(game.snake.x[i]/game.BLOCK_WIDTH)-1
            y = int(game.snake.y[i]/game.BLOCK_WIDTH)-1
            grid[x][y] = 1

        # Mark the food on the grid
        food_x = int(game.apple.x/game.BLOCK_WIDTH)-1
        food_y = int(game.apple.y/game.BLOCK_WIDTH)-1
        grid[food_x][food_y] = 2

        return grid.flatten()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_memory(self):
        if len(self.memory) > LEARNING_STARTS:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
            if len(self.memory) < LEARNING_STARTS+500:
                print("learning starts")
            # states, actions, rewards, next_states, dones = zip(*mini_sample)
            # self.trainer.train_step(states, actions, rewards, next_states, dones)
            for state, action, reward, next_state, done in mini_sample:
                self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: trade off exploration / exploitation
        self.total_decisions += 1
        final_move = [0, 0, 0]
        if np.random.rand() <= self.epsilon:
            self.random_count += 1
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            self.training_count += 1
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        if self.total_decisions % 1000 == 0:
            # print("random ", self.random_count)
            # print("training ", self.training_count)
            print("epsilon ", self.epsilon)
            print(len(self.memory))
        return final_move

    def load(self, file_name='model.pth'):

        file_path = os.path.join(self.model_folder_path, file_name)
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            print("Model loaded.")
            self.retrieve_data()

    def save_data(self, n_games, record, score, epsilon, file_name='data.json'):

        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        complete_path = os.path.join(self.model_folder_path, file_name)
        self.total_eats += score
        self.avg = round((self.total_eats / n_games), 2)
        data = {'episodes': n_games, 'record': record, 'avg': self.avg, 'epsilon': epsilon}
        with open(complete_path, 'w') as file:
            json.dump(data, file, indent=4)

    def retrieve_data(self):
        data = None
        model_data_path = os.path.join(self.model_folder_path, self.data_file)
        if os.path.exists(model_data_path):
            with open(model_data_path, 'r') as file:
                data = json.load(file)

            if data is not None:
                self.n_games = data['episodes']
                self.record = data['record']
                self.avg = data['avg']
                self.epsilon = data['epsilon']
                self.total_eats = self.n_games * self.avg


def train():
    game = Game()
    height = int(game.SCREEN_SIZE / game.BLOCK_WIDTH)
    width = int(game.SCREEN_SIZE / game.BLOCK_WIDTH)
    state_size = width * height
    agent = Agent(state_size=state_size, width=width, height=height)
    agent.load()

    while True:

        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get the new state
        reward, done, score = game.run_step(final_move)
        state_new = agent.get_state(game)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # train after every 4 steps
        if agent.t_step % 4 == 0:
            agent.train_memory()

        # update target model
        agent.t_step = (agent.t_step + 1) % 5000
        if agent.t_step == 0:
            print("updating target model")
            agent.target_model.load_state_dict(agent.model.state_dict())

        if done:
            # train long term memory (Experience Replay)
            game.reset()
            agent.n_games += 1
            if len(agent.memory) > LEARNING_STARTS and agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            if score > agent.record:
                agent.record = score
                agent.model.save()

            message = "Episodes: " + str(agent.n_games) \
                      + "    Record: " + str(agent.record)
            # print(message)
            game.message = message
            agent.save_data(agent.n_games, agent.record, score, agent.epsilon)


if __name__ == '__main__':
    train()
