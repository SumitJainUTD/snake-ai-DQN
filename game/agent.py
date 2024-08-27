import json
import os

import torch
import random
import numpy as np
from collections import deque

from game.model import Linear_QNet, QTrainer
# from game.snake import Game
from game.snake_no_ui import Game
from game.plot import plot

MAX_MEMORY = 150_000
BATCH_SIZE = 32
LR = 0.001
TARGET_UPDATE_INTERVAL = 4000

# Universal device selection: CUDA, MPS, or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("cpu")
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
        self.model = Linear_QNet(11, 64, 3)
        # self.model = Linear_QNet(11, 64, 3).to(device)
        self.target_model = Linear_QNet(11, 64, 3)
        # self.target_model = Linear_QNet(11, 64, 3).to(device)
        self.trainer = QTrainer(self.model, self.target_model, LR, self.gamma)
        self.data_file = 'data.json'
        self.record = 0
        self.model_folder_path = './model'
        self.t_step = 0
        self.LEARNING_STARTS_IN = 10000
        # model, trainer

        pass

    def get_state(self, game):
        head_x = game.snake.x[0]
        head_y = game.snake.y[0]

        point_left = [(head_x - game.BLOCK_WIDTH), head_y]
        point_right = [head_x + game.BLOCK_WIDTH, head_y]
        point_up = [head_x, head_y - game.BLOCK_WIDTH]
        point_down = [head_x, head_y + game.BLOCK_WIDTH]

        dir_l = game.snake.direction == "left"
        dir_r = game.snake.direction == "right"
        dir_u = game.snake.direction == "up"
        dir_d = game.snake.direction == "down"

        # [danger straight, danger right, danger left,
        #  direction left, direction right, direction up, direction down,
        #  food left, food right, food up, food down]

        state = [
            # danger straight
            (dir_r and game.is_collision(point=point_right)) or
            (dir_l and game.is_collision(point=point_left)) or
            (dir_u and game.is_collision(point=point_up)) or
            (dir_d and game.is_collision(point=point_left)),

            # danger right
            (dir_u and game.is_collision(point=point_right)) or
            (dir_r and game.is_collision(point=point_down)) or
            (dir_d and game.is_collision(point=point_left)) or
            (dir_l and game.is_collision(point=point_up)),

            # danger left
            (dir_u and game.is_collision(point=point_left)) or
            (dir_l and game.is_collision(point=point_down)) or
            (dir_d and game.is_collision(point=point_right)) or
            (dir_r and game.is_collision(point=point_up)),

            # move directions
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location - compare food location with snake head
            game.apple.x < game.snake.x[0],  # food left
            game.apple.x > game.snake.x[0],  # food right
            game.apple.y < game.snake.y[0],  # food up
            game.apple.y > game.snake.x[0]  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.LEARNING_STARTS_IN >= 0:
            self.LEARNING_STARTS_IN -= 1

    def train_memory(self):
        if self.LEARNING_STARTS_IN <= 0 and len(self.memory) >= BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
            if self.LEARNING_STARTS_IN == 0:
                print("learning started")
            for state, action, reward, next_state, done in mini_sample:
                self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: trade off exploration / exploitation
        final_move = [0, 0, 0]
        if np.random.rand() <= self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:

            state0 = torch.tensor(state, dtype=torch.float)
            # state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            # Ensure the prediction is on the correct device
            # prediction = prediction.to(device)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

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
        data = {'episodes': n_games, 'record': record, 'epsilon': epsilon,
                'learning_starts_in': self.LEARNING_STARTS_IN}
        with open(complete_path, 'w') as file:
            json.dump(data, file, indent=4)

    def retrieve_data(self):
        model_data_path = os.path.join(self.model_folder_path, self.data_file)
        if os.path.exists(model_data_path):
            with open(model_data_path, 'r') as file:
                data = json.load(file)

            if data is not None:
                self.n_games = data['episodes']
                self.record = data['record']
                self.epsilon = data['epsilon']
                self.LEARNING_STARTS_IN = data['learning_starts_in']


def train():
    agent = Agent()
    agent.load()
    game = Game()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    avg_last_10_episodes = 0
    score_in_games = deque(maxlen=50)

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
            if agent.LEARNING_STARTS_IN <= 0 and agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            if agent.LEARNING_STARTS_IN == 0:
                print("learning started, Episodes will reset")
                agent.n_games = 0

            if score > agent.record:
                agent.record = score
                agent.model.save()

            score_in_games.append(score)
            avg_last_50_episodes = round(sum(score_in_games) / len(score_in_games), 1)

            message = "Episodes: " + str(agent.n_games) \
                      + "    Record: " + str(agent.record) + "    Avg: " + str(avg_last_50_episodes)
            print(message)
            game.message = message
            agent.save_data(agent.n_games, agent.record, score, agent.epsilon)

            # plot_scores.append(score)
            # score_in_games.append(score)
            # avg_last_10_episodes = sum(score_in_games)/len(score_in_games)
            #
            # plot_mean_scores.append(avg_last_10_episodes)
            # plot(plot_scores, plot_mean_scores)

            # for name, param in agent.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient norm for {name}: {param.grad.norm()}")


if __name__ == '__main__':
    train()
