import random
import sys
from collections import deque
import pygame

class Base():
    def __init__(self):
        self.BLOCK_WIDTH = 40
        self.SCREEN_SIZE = 600

class Snake(Base):
    def __init__(self, parent_screen, length=1):
        super().__init__()
        self.length = length
        self.parent_screen = parent_screen
        self.block = pygame.image.load("resources/block.jpg")
        self.x = [self.BLOCK_WIDTH * 4] * self.length
        self.y = [self.BLOCK_WIDTH * 4] * self.length
        self.direction = "right"

    def draw(self):
        self.parent_screen.fill((58, 59, 36))
        for i in range(self.length):
            self.parent_screen.blit(self.block, (self.x[i], self.y[i]))

    def increase(self):
        self.length += 1
        self.x.append(-1)
        self.y.append(-1)

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

        self.draw()


class Apple(Base):
    def __init__(self, parent_screen):
        super().__init__()
        self.parent_screen = parent_screen
        self.apple_img = pygame.image.load("resources/apple.jpg")
        self.x = random.randint(0, 10) * self.BLOCK_WIDTH
        self.y = random.randint(0, 10) * self.BLOCK_WIDTH

    def draw(self):
        self.parent_screen.blit(self.apple_img, (self.x, self.y))

    def move(self, snake):
        while True:  # make sure new food is not getting created over snake body
            x = random.randint(0, 10) * self.BLOCK_WIDTH
            y = random.randint(0, 10) * self.BLOCK_WIDTH
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
        pygame.init()
        pygame.display.set_caption("Snake Game - AI - Deep Q Learning")
        # self.SCREEN_UPDATE = pygame.USEREVENT
        # pygame.time.set_timer(self.SCREEN_UPDATE, 1)
        self.surface = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
        self.surface.fill((58, 59, 36))
        self.snake = Snake(self.surface, 1)
        self.snake.draw()
        self.apple = Apple(self.surface)
        self.apple.draw()
        self.score = 0
        self.reward = 0
        self.iterations_without_rewards = 0
        self.game_over = False
        self.message = ''
        self.position_history = deque(maxlen=50)
        self.loop_detect_counter = 0

    def eat(self, x1, y1, x2, y2):
        return x1 == x2 and y1 == y2

    def render_background(self):
        bg = pygame.image.load("resources/background.jpg")
        self.surface.blit(bg, (0, 0))

    def display_score(self):
        pass
        font = pygame.font.SysFont('arial', 20)
        msg = "Score: " + str(self.score)
        score = font.render(f"{msg}", True, (200, 200, 200))
        self.surface.blit(score, (380, 10))

    def display_message(self, message):
        font = pygame.font.SysFont('arial', 20)
        score = font.render(f"{message}", True, (200, 200, 200))
        self.surface.blit(score, (100, 10))

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
        self.render_background()
        self.snake.move()
        self.apple.draw()
        self.display_score()
        self.display_message(self.message)
        self.iterations_without_rewards += 1
        self.reward = 0
        # self.reward = -0.1  # Small negative reward for each step

        # time.sleep(0.15)

        if self.eat(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.snake.increase()
            self.apple.move(self.snake)
            self.score += 1
            self.iterations_without_rewards = 0  # reset
            self.reward = 10

        if self.is_collision():
            self.reward = -10
            self.game_over = True

        # if self.iterations_without_rewards > 300 * self.snake.length:
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

    def render_background(self):
        bg = pygame.image.load("resources/background.jpg")
        self.surface.blit(bg, (0, 0))

    def reset(self):
        self.snake = Snake(self.surface)
        self.apple = Apple(self.surface)
        self.score = 0
        self.game_over = False
        self.iterations_without_rewards = 0
        self.position_history.clear()
        self.loop_detect_counter = 0

    def run_step(self, action):

        # action = [up, right, down, left]
        direction = 'up'

        if action == 0:
            direction = 'up'
        elif action == 1:
            direction = 'right'
        elif action == 2:
            direction = 'down'
        elif action == 3:
            direction = 'left'

        if direction == "left":
            # if self.snake.direction != "right":
            self.snake.direction = "left"
        elif direction == "right":
            # if self.snake.direction != "left":
            self.snake.direction = "right"
        elif direction == "down":
            # if self.snake.direction != "up":
            self.snake.direction = "down"
        else:
            # if self.snake.direction != "down":
            self.snake.direction = "up"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.play()
        pygame.display.update()
        pygame.time.Clock().tick(200)

        return self.reward, self.game_over, self.score
