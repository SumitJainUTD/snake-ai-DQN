import random
from collections import deque
import numpy as np


class Base:
    def __init__(self):
        self.BLOCK_WIDTH = 20
        self.SCREEN_SIZE = 600
        self.apple_location_multiplier = self.SCREEN_SIZE / self.BLOCK_WIDTH


class Snake(Base):
    def __init__(self, length=1):
        super().__init__()
        self.length = length
        self.x = [self.BLOCK_WIDTH * 2] * self.length
        self.y = [self.BLOCK_WIDTH * 2] * self.length
        self.direction = "right"

    def increase(self):
        self.length += 1
        self.x.append(-1)
        self.y.append(-1)

    def move_left(self):
        if self.direction != "right":
            self.direction = "left"

    def move_right(self):
        if self.direction != "left":
            self.direction = "right"

    def move_up(self):
        if self.direction != "down":
            self.direction = "up"

    def move_down(self):
        if self.direction != "up":
            self.direction = "down"

    def move(self):

        for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        if self.direction == "right":
            self.x[0] += self.BLOCK_WIDTH
        if self.direction == "left":
            self.x[0] -= self.BLOCK_WIDTH
        if self.direction == "down":
            self.y[0] += self.BLOCK_WIDTH
        if self.direction == "up":
            self.y[0] -= self.BLOCK_WIDTH


class Apple(Base):
    def __init__(self):
        super().__init__()
        self.x = random.randint(0, 25) * self.BLOCK_WIDTH
        self.y = random.randint(0, 25) * self.BLOCK_WIDTH

    def move(self, snake):
        while True:  # make sure new food is not getting created over snake body
            x = random.randint(0, 25) * self.BLOCK_WIDTH
            y = random.randint(0, 25) * self.BLOCK_WIDTH
            clean = True
            for i in range(0, snake.length):
                if x == snake.x[i] and y == snake.y[i]:
                    clean = False
                    break
            if clean:
                self.x = x
                self.y = y
                return


class Game(Base):
    def __init__(self):
        super().__init__()
        self.snake = Snake(1)
        self.apple = Apple()
        self.score = 0
        self.reward = 0
        self.iterations_without_rewards = 0
        self.game_over = False
        self.message = ''
        self.position_history = deque(maxlen=50)
        self.loop_detect_counter = 0

    def eat(self, x1, y1, x2, y2):
        return x1 == x2 and y1 == y2

    def display_score(self):
        pass

    def display_message(self, message):
        pass

    def is_collision(self, point=None):
        is_head = False
        if point is None:
            point_x = self.snake.x[0]
            point_y = self.snake.y[0]
            is_head = True
        else:
            point_x = point[0]
            point_y = point[1]

        if is_head:
            for i in range(3, self.snake.length):
                if point_x == self.snake.x[i] and point_y == self.snake.y[i]:
                    return True
        else:
            for i in range(0, self.snake.length):
                if point_x == self.snake.x[i] and point_y == self.snake.y[i]:
                    return True

        if point_x > (self.SCREEN_SIZE - self.BLOCK_WIDTH) \
                or point_y > (self.SCREEN_SIZE - self.BLOCK_WIDTH) \
                or point_x < 0 \
                or point_y < 0:
            return True

        return False

    def play(self):
        self.snake.move()
        self.display_score()
        self.display_message(self.message)
        self.iterations_without_rewards += 1
        self.reward = 0

        if self.eat(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.snake.increase()
            self.apple.move(self.snake)
            self.score += 1
            self.iterations_without_rewards = 0  # reset
            self.reward = 10

        if self.is_collision():
            self.reward = -200
            self.game_over = True

        # if self.iterations_without_rewards > 400 * self.snake.length:
        #     self.reward = -1
        #     self.game_over = True
        #     print("Iterations exceeded: Game Over")

        # Check for loops by tracking positions
        # if (self.snake.x[0], self.snake.y[0]) in self.position_history:
        #     self.loop_detect_counter += 1
        #     if(self.loop_detect_counter>50):
        #         self.reward = -10
        #         self.game_over = True
        #         self.loop_detect_counter = 0
        #         print("loop detected")
        # self.position_history.append((self.snake.x[0], self.snake.y[0]))

    def reset(self):
        self.snake = Snake()
        self.apple = Apple()
        self.score = 0
        self.game_over = False
        self.iterations_without_rewards = 0
        self.position_history.clear()
        self.loop_detect_counter = 0

    def get_next_direction(self, action):
        # [straight, right, left]
        clockwise = ["right", "down", "left", "up"]
        idx = clockwise.index(self.snake.direction)

        # [1, 0, 0] - straight
        # [0, 1, 0] - right
        # [0, 0, 1] - left
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4
            new_dir = clockwise[new_idx]  # turn right  r -> d -> l -> u
        else:
            new_idx = (idx - 1) % 4
            new_dir = clockwise[new_idx]  # turn left r -> u -> l -> d

        return new_dir

    def run_step(self, action):

        dir = self.get_next_direction(action)

        if dir == "left":
            if self.snake.direction != "right":
                self.snake.move_left()
        elif dir == "right":
            if self.snake.direction != "left":
                self.snake.move_right()
        elif dir == "down":
            if self.snake.direction != "up":
                self.snake.move_down()
        else:
            if self.snake.direction != "down":
                self.snake.move_up()

        self.play()

        return self.reward, self.game_over, self.score
