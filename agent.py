import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self) -> None:
        self.number_of_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model ,LR, self.gamma)


    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.number_of_games
        action = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else: 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        action[move] = 1
        return action

def train():
    max_score = 0
    agent = Agent()
    game = SnakeGame()

    total_score = 0

    while True:
        old_state = agent.get_state(game)
        action = agent.get_action(old_state)
        reward, done, score = game.play_step(action)
        new_state = agent.get_state(game)
        agent.train_short_memory(old_state, action, reward, new_state, done)
        agent.remember(old_state, action, reward, new_state, done)

        if done:
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > max_score:
                max_score = score
                agent.model.save()

            total_score += score
            mean_score = total_score / agent.number_of_games
            print(
                "Game",
                agent.number_of_games,
                "Score",
                score,
                "Record",
                max_score,
                "Mean Score",
                mean_score,
            )


if __name__ == "__main__":
    train()