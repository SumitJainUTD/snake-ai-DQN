import random
import time
import numpy as np

import pygame
from pygame.locals import *

BLOCK_WIDTH = 40
SCREEN_SIZE = 600


# reset
# reward
# play(action) -> direction
# game_iteration
# is_collision


class Snake():
    def __init__(self, parent_screen, length=1):
        self.length = length
        self.parent_screen = parent_screen
        self.block = pygame.image.load("resources/block.jpg")
        self.x = [BLOCK_WIDTH*4] * self.length
        self.y = [BLOCK_WIDTH*4] * self.length
        self.direction = "right"

    def draw(self):
        self.parent_screen.fill((58, 59, 36))
        for i in range(self.length):
            self.parent_screen.blit(self.block, (self.x[i], self.y[i]))
        # pygame.display.update()

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
            self.x[0] += BLOCK_WIDTH
        if self.direction == "left":
            self.x[0] -= BLOCK_WIDTH
        if self.direction == "down":
            self.y[0] += BLOCK_WIDTH
        if self.direction == "up":
            self.y[0] -= BLOCK_WIDTH

        # if self.x[0] >= SCREEN_SIZE:
        #     self.x[0] = 0
        #
        # if self.x[0] < 0:
        #     self.x[0] = SCREEN_SIZE
        #
        # if self.y[0] >= SCREEN_SIZE:
        #     self.y[0] = 0
        #
        # if self.y[0] < 0:
        #     self.y[0] = SCREEN_SIZE

        self.draw()
        # print(str(self.x) + " " + str(self.y))


class Apple:
    def __init__(self, parent_screen):
        self.parent_screen = parent_screen
        self.apple_img = pygame.image.load("resources/apple.jpg")
        self.x = random.randint(0, 10) * BLOCK_WIDTH
        self.y = random.randint(0, 10) * BLOCK_WIDTH

    def draw(self):
        self.parent_screen.blit(self.apple_img, (self.x, self.y))

    def move(self):
        self.x = random.randint(0, 10) * BLOCK_WIDTH
        self.y = random.randint(0, 10) * BLOCK_WIDTH


class Game():
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake Game - Sumit")
        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE, 1)
        self.surface = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.surface.fill((58, 59, 36))
        self.snake = Snake(self.surface, 1)
        self.snake.draw()
        self.apple = Apple(self.surface)
        self.apple.draw()
        self.score = 0
        self.reward = 0
        self.iterations_without_rewards = 0
        self.game_over = False

        pygame.mixer.init()
        self.play_background_music()

    def eat(self, x1, y1, x2, y2):
        return x1 == x2 and y1 == y2

    def play_background_music(self):
        pygame.mixer.music.load('resources/bg_music_1.mp3')
        pygame.mixer.music.play(-1, 0)

    def play_sound(self, sound_name):
        if sound_name == "crash":
            sound = pygame.mixer.Sound("resources/crash.mp3")
        elif sound_name == 'ding':
            sound = pygame.mixer.Sound("resources/ding.mp3")

        pygame.mixer.Sound.play(sound)

    def display_score(self):
        font = pygame.font.SysFont('arial', 30)
        score = font.render(f"{self.score}", True, (200, 200, 200))
        self.surface.blit(score, (750, 10))

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
                    self.play_sound('crash')
                    return True
        else:
            for i in range(0, self.snake.length):
                if point_x == self.snake.x[i] and point_y == self.snake.y[i]:
                    self.play_sound('crash')
                    return True

        if point_x > (SCREEN_SIZE - BLOCK_WIDTH) \
                or point_y > (SCREEN_SIZE - BLOCK_WIDTH) \
                or point_x < 0 \
                or point_y < 0:
            self.play_sound('crash')
            return True

        return False

    def play(self):
        self.snake.move()
        self.apple.draw()
        self.display_score()
        self.iterations_without_rewards +=1
        self.reward = 0

        # time.sleep(0.15)

        if self.eat(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.snake.increase()
            self.apple.move()
            self.score += 1
            self.play_sound("ding")
            # self.iterations_without_rewards = 0  # reset
            self.reward = 10

        if self.is_collision():
            self.reward -= 50
            self.game_over = True

        if self.iterations_without_rewards > 100 * self.snake.length:
            self.reward -= 10
            self.game_over = True
            print("Iterations exceeded: Game Over")

    def render_background(self):
        bg = pygame.image.load("resources/background.jpg")
        self.surface.blit(bg, (0, 0))

    def reset(self):
        self.snake = Snake(self.surface)
        self.apple = Apple(self.surface)
        self.score = 0
        self.game_over = False
        self.iterations_without_rewards = 0

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

        for event in pygame.event.get():
            if event.type == self.SCREEN_UPDATE:
                self.play()
                pygame.display.update()
                pygame.time.Clock().tick(100)
                break

        return self.reward, self.game_over, self.score
