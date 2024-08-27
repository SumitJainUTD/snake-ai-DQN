import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Universal device selection: CUDA, MPS, or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # self.model = model.to(device)
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # state = torch.tensor(state, dtype=torch.float).to(device)
        # next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        # action = torch.tensor(action, dtype=torch.long).to(device)
        # reward = torch.tensor(reward, dtype=torch.float).to(device)
        # done = torch.tensor(done, dtype=torch.bool).to(device)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        target = pred.clone().detach()
        # target = pred.clone().detach().to(device)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    # def train_step(self, state, action, reward, next_state, done):
    #     state = torch.tensor(state, dtype=torch.float)
    #     next_state = torch.tensor(next_state, dtype=torch.float)
    #     action = torch.tensor(action, dtype=torch.long)
    #     reward = torch.tensor(reward, dtype=torch.float)
    #     # (n, x)
    #
    #     # (1, x)
    #     state = state.unsqueeze(0)
    #     next_state = next_state.unsqueeze(0)
    #     action = action.unsqueeze(0)
    #     reward = reward.unsqueeze(0)
    #
    #     # 1: predicted Q values with current state
    #     pred = self.model(state)
    #
    #     # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
    #     # pred.clone()
    #     # preds[argmax(action)] = Q_new
    #
    #     target = pred.clone()
    #     Q_new = reward
    #     if not done:
    #         Q_new = reward + self.gamma * torch.max(self.target_model(next_state))
    #
    #     target[torch.argmax(action).item()] = Q_new
    #
    #     self.optimizer.zero_grad()
    #     loss = self.criterion(target, pred)
    #     # print('loss', loss)
    #     loss.backward()
    #
    #     self.optimizer.step()
