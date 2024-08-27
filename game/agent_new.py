import json
import os

import torch
import random
import numpy as np
from collections import deque

from game.model import Linear_QNet, QTrainer
from game.snake_grid import Game

MAX_MEMORY = 150_000
LEARNING_STARTS = 100
BATCH_SIZE = 32
LR = 0.001
TARGET_UPDATE_INTERVAL = 4000
BLOCK_WIDTH = 40

# Universal device selection: CUDA, MPS, or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.n_segments = 0
        self.state_size = 2 + 2 + self.n_segments * 2
        self.model = Linear_QNet(self.state_size, 128, 4).to(device)
        self.target_model = Linear_QNet(self.state_size, 128, 4).to(device)
        self.trainer = QTrainer(self.model, self.target_model, LR, self.gamma)
        self.data_file = 'data.json'
        self.record = 0
        self.avg = 0
        self.total_eats = 0
        self.model_folder_path = './model'
        self.t_step = 0

        # model, trainer

        self.random_count = 0
        self.training_count = 0
        self.total_decisions = 0

        pass

    def get_state(self, game):
        # [food_x, food_y, head_x, head_y, ..last 5 segments ..] = [2+2+5*2] = 14

        state = []

        # snake head
        head_x = game.snake.x[0]
        head_y = game.snake.y[0]

        # food relative position
        food_x = game.apple.x - head_x
        food_y = game.apple.y - head_y

        state.append(food_x)
        state.append(food_y)

        state.append(head_x)
        state.append(head_y)

        # last N segment
        for i in range(1, min(self.n_segments, len(game.snake.x))):
            state.append(game.snake.x[-i])
            state.append(game.snake.y[-i])

        # Pad the state if necessary
        while len(state) < self.state_size:
            state.append(0)

        # current direction
        if game.snake.direction == 'up':
            state.append(0)
        elif game.snake.direction == 'right':
            state.append(1)
        elif game.snake.direction == 'down':
            state.append(2)
        elif game.snake.direction == 'left':
            state.append(4)



        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_memory(self):
        if len(self.memory) > LEARNING_STARTS:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
            if len(self.memory) < LEARNING_STARTS+500:
                print("learning starts")

            for state, action, reward, next_state, done in mini_sample:
                self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: trade off exploration / exploitation
        if np.random.rand() <= self.epsilon:
                return random.randrange(4)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

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
    agent = Agent()
    agent.load()

    while True:

        # get old state
        state_old = agent.get_state(game)

        # get move
        action = agent.get_action(state_old)

        # perform move and get the new state
        reward, done, score = game.run_step(action)
        state_new = agent.get_state(game)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        # train after every 4 steps
        if agent.t_step % 4 == 0:
            agent.train_memory()

        # update target model
        agent.t_step = (agent.t_step + 1) % TARGET_UPDATE_INTERVAL
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
            print(message)
            game.message = message
            agent.save_data(agent.n_games, agent.record, score, agent.epsilon)


if __name__ == '__main__':
    train()
