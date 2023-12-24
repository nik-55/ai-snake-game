import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size, hidden_size, output_size ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        return out
    
    def save(self, file_name='game'):
        torch.save(self.state_dict(), './models/' + file_name + '.pth')


class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = state.reshape(1,-1)
            next_state = next_state.reshape(1,-1)
            action = action.reshape(1,-1)
            reward = reward.reshape(1,-1)
            done = (done, )

        predictions = self.model(state)

        target = predictions.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma*torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx])] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, predictions)
        loss.backward()

        self.optimizer.step()