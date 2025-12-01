import sys
import time
import numpy as np
from tqdm import tqdm
import pygame
import matplotlib.pyplot as plt


class TaxiGameEnv:
    ACTION_SOUTH = 0
    ACTION_NORTH = 1
    ACTION_EAST = 2
    ACTION_WEST = 3
    ACTION_PICKUP = 4
    ACTION_DROPOFF = 5

    WEATHER_SUN = 0
    WEATHER_RAIN = 1
    WEATHER_HEAT = 2

    def __init__(self, rows, cols, render_mode="human", fps=8, max_steps=200, seed=None):
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.random_state = np.random.RandomState(seed)
        self.render_mode = render_mode
        self.fps = fps

        self.reward_step = -0.01
        self.reward_illegal = -10
        self.reward_dropoff = 20

        self.weather_states = 3
        self.p_rain = 0.25
        self.p_heat = 0.25
        self.p_auto_move_rain = 0.2
        self.extra_penalty_heat = -0.02

        self.shaping_factor = 0.1

        self.locations = {
            0: ("R", (0, 0)),
            1: ("G", (0, self.cols - 1)),
            2: ("Y", (self.rows - 1, 0)),
            3: ("B", (self.rows - 1, self.cols - 2)),
        }

        self.taxi_row = 0
        self.taxi_col = 0
        self.passenger_location = 0
        self.destination = 0
        self.step_count = 0
        self.weather = self.WEATHER_SUN

        self.current_episode = 1

        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.cell_size = 80
        self.margin = 10
        self.width_px = self.cols * self.cell_size + 2 * self.margin
        self.height_px = self.rows * self.cell_size + 2 * self.margin + 60

    @property
    def n_states(self):
        return self.rows * self.cols * 5 * 4 * self.weather_states

    @property
    def n_actions(self):
        return 6

    def encode(self, row, col, passenger_location, destination, weather):
        v = row
        v = v * self.cols + col
        v = v * 5 + passenger_location
        v = v * 4 + destination
        v = v * self.weather_states + weather
        return int(v)

    def decode(self, state_value):
        w = state_value % self.weather_states
        state_value //= self.weather_states
        d = state_value % 4
        state_value //= 4
        p = state_value % 5
        state_value //= 5
        c = state_value % self.cols
        r = state_value // self.cols
        return r, c, p, d, w

    def _sample_weather(self):
        r = self.random_state.rand()
        if r < self.p_rain:
            return self.WEATHER_RAIN
        elif r < self.p_rain + self.p_heat:
            return self.WEATHER_HEAT
        else:
            return self.WEATHER_SUN

    def _random_move(self):
        actions = []
        if self.taxi_row < self.rows - 1:
            actions.append("S")
        if self.taxi_row > 0:
            actions.append("N")
        if self.taxi_col < self.cols - 1:
            actions.append("E")
        if self.taxi_col > 0:
            actions.append("W")
        if not actions:
            return
        a = self.random_state.choice(actions)
        if a == "S":
            self.taxi_row += 1
        elif a == "N":
            self.taxi_row -= 1
        elif a == "E":
            self.taxi_col += 1
        elif a == "W":
            self.taxi_col -= 1

    def _manhattan_distance(self, r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2)

    def reset(self):
        self.taxi_row = self.random_state.randint(0, self.rows)
        self.taxi_col = self.random_state.randint(0, self.cols)
        self.passenger_location = self.random_state.randint(0, 4)
        self.destination = self.random_state.randint(0, 4)
        while self.destination == self.passenger_location:
            self.destination = self.random_state.randint(0, 4)
        self.step_count = 0
        self.weather = self._sample_weather()

        if self.render_mode == "human":
            self._init_pygame()
            self._draw()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps
        reward = 0.0

        if self.passenger_location != 4:
            _, (trow, tcol) = self.locations[self.passenger_location]
        else:
            _, (trow, tcol) = self.locations[self.destination]
        old_dist = self._manhattan_distance(self.taxi_row, self.taxi_col, trow, tcol)

        if action == self.ACTION_SOUTH:
            if self.taxi_row < self.rows - 1:
                self.taxi_row += 1
            reward += self.reward_step

        elif action == self.ACTION_NORTH:
            if self.taxi_row > 0:
                self.taxi_row -= 1
            reward += self.reward_step

        elif action == self.ACTION_EAST:
            if self.taxi_col < self.cols - 1:
                self.taxi_col += 1
            reward += self.reward_step

        elif action == self.ACTION_WEST:
            if self.taxi_col > 0:
                self.taxi_col -= 1
            reward += self.reward_step

        elif action == self.ACTION_PICKUP:
            if self.passenger_location != 4:
                _, (pr, pc) = self.locations[self.passenger_location]
                if (self.taxi_row, self.taxi_col) == (pr, pc):
                    self.passenger_location = 4
                    reward += self.reward_step
                else:
                    reward += self.reward_illegal
            else:
                reward += self.reward_illegal

        elif action == self.ACTION_DROPOFF:
            if self.passenger_location == 4:
                _, (dr, dc) = self.locations[self.destination]
                if (self.taxi_row, self.taxi_col) == (dr, dc):
                    reward += self.reward_dropoff
                    terminated = True
                    self.passenger_location = self.destination
                else:
                    reward += self.reward_illegal
            else:
                reward += self.reward_illegal

        else:
            reward += self.reward_step

        if self.weather == self.WEATHER_RAIN:
            if self.random_state.rand() < self.p_auto_move_rain:
                self._random_move()
                reward += -0.05
        elif self.weather == self.WEATHER_HEAT:
            reward += self.extra_penalty_heat

        if self.passenger_location != 4:
            _, (trow2, tcol2) = self.locations[self.passenger_location]
        else:
            _, (trow2, tcol2) = self.locations[self.destination]
        new_dist = self._manhattan_distance(self.taxi_row, self.taxi_col, trow2, tcol2)

        reward += self.shaping_factor * (old_dist - new_dist)

        obs = self._get_obs()

        if self.render_mode == "human":
            self._handle_events()
            self._draw()
            self.clock.tick(self.fps)

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            self._handle_events()
            self._draw()
            self.clock.tick(self.fps)

    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False

    def _get_obs(self):
        return self.encode(
            self.taxi_row,
            self.taxi_col,
            self.passenger_location,
            self.destination,
            self.weather,
        )

    def _init_pygame(self):
        if self.pygame_initialized:
            return
        pygame.init()
        self.screen = pygame.display.set_mode((self.width_px, self.height_px))
        pygame.display.set_caption("Taxi Game")
        self.clock = pygame.time.Clock()
        self.pygame_initialized = True

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit(0)

    def _draw(self):
        if not self.pygame_initialized:
            return

        bg = (18, 18, 26)
        grid = (40, 40, 60)
        taxi_c = (255, 215, 0)
        pas_c = (80, 200, 120)
        dst_c = (80, 160, 255)
        txt_c = (230, 230, 230)

        self.screen.fill(bg)

        for r in range(self.rows):
            for c in range(self.cols):
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                pygame.draw.rect(self.screen, grid,
                                 pygame.Rect(x, y, self.cell_size, self.cell_size), 1)

        font = pygame.font.SysFont("DejaVu Sans", 20, bold=True)
        for idx, (name, (rr, cc)) in self.locations.items():
            x = self.margin + cc * self.cell_size
            y = self.margin + rr * self.cell_size
            rect = pygame.Rect(x + 5, y + 5, self.cell_size - 10, self.cell_size - 10)
            pygame.draw.rect(self.screen, (50, 50, 80), rect)
            text_surface = font.render(name, True, txt_c)
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)

        _, (dr, dc) = self.locations[self.destination]
        dx = self.margin + dc * self.cell_size
        dy = self.margin + dr * self.cell_size
        pygame.draw.rect(self.screen, dst_c,
                         (dx + 10, dy + 10, self.cell_size - 20, self.cell_size - 20), 2)

        if self.passenger_location != 4:
            _, (pr, pc) = self.locations[self.passenger_location]
            px = self.margin + pc * self.cell_size + self.cell_size // 2
            py = self.margin + pr * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, pas_c, (px, py), 10)

        tx = self.margin + self.taxi_col * self.cell_size + 10
        ty = self.margin + self.taxi_row * self.cell_size + 10
        taxi_rect = pygame.Rect(tx, ty, self.cell_size - 20, self.cell_size - 20)
        pygame.draw.rect(self.screen, taxi_c, taxi_rect)
        if self.passenger_location == 4:
            pygame.draw.circle(self.screen, pas_c, taxi_rect.center, 8)

        hud_y = self.margin + self.rows * self.cell_size + 10
        pygame.draw.rect(self.screen, (30, 30, 40),
                         (0, hud_y, self.width_px, 50))

        weather_str = {
            self.WEATHER_SUN: "Sol",
            self.WEATHER_RAIN: "Lluvia",
            self.WEATHER_HEAT: "Calor"
        }[self.weather]

        info = f"Ep={self.current_episode}  State={self._get_obs()}  Steps={self.step_count}  Tiempo={weather_str}"
        hud_font = pygame.font.SysFont("DejaVu Sans", 18)
        hud_surface = hud_font.render(info, True, txt_c)
        self.screen.blit(hud_surface, (self.margin, hud_y + 10))

        pygame.display.flip()


def train_q_learning(env, episodes=20000, alpha=0.1, gamma=0.99,
                     epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.9995):
    q_table = np.zeros((env.n_states, env.n_actions))
    epsilon = epsilon_start
    rewards = []
    eps_history = []

    for _ in tqdm(range(episodes), desc="Entrenando"):
        state, _ = env.reset()
        done = False
        total_r = 0.0

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(q_table[state])

            next_state, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = 0.0 if done else np.max(q_table[next_state])
            q_table[state, action] += alpha * (r + gamma * best_next - q_table[state, action])

            state = next_state
            total_r += r

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_r)
        eps_history.append(epsilon)

    return q_table, rewards, eps_history


def run_trained_agent(q_table, episodes=5, fps=5):
    env = TaxiGameEnv(render_mode="human", cols=6, rows=6, fps=fps, seed=0, max_steps=100)

    for ep in range(1, episodes + 1):
        env.current_episode = ep
        state, _ = env.reset()
        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            time.sleep(0.1)

        time.sleep(0.5)

    env.close()


if __name__ == "__main__":
    print("1. Entrenar con Q-learning")
    print("2. Ejecutar con modelo entrenado")
    option = input("> ")

    if option == "1":
        env = TaxiGameEnv(render_mode=None, rows=6, cols=6, seed=0, max_steps=100)
        q_table, rewards, eps_history = train_q_learning(env)

        np.save("qtable_taxi.npy", q_table)

        n_episodes = len(rewards)
        window = 100
        if n_episodes >= window:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        else:
            moving_avg = None

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        axes[0].plot(rewards)
        axes[0].set_xlabel("Episodio")
        axes[0].set_ylabel("Recompensa")
        axes[0].set_title("Recompensa por episodio")

        if moving_avg is not None:
            axes[1].plot(range(window - 1, n_episodes), moving_avg)
        axes[1].set_xlabel("Episodio")
        axes[1].set_ylabel("Recompensa media")
        axes[1].set_title(f"Media móvil ({window} episodios)")

        axes[2].plot(eps_history)
        axes[2].set_xlabel("Episodio")
        axes[2].set_ylabel("Epsilon")
        axes[2].set_title("Exploración (epsilon)")

        fig.tight_layout()
        fig.savefig("entrenamiento_qlearning.png")

    else:
        q_table = np.load("qtable_taxi.npy")
        try:
            n_episodes = int(input("Número de episodios a visualizar: "))
        except ValueError:
            n_episodes = 5
        run_trained_agent(q_table, episodes=n_episodes, fps=5)
