import sys
import time
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import pygame
import matplotlib.pyplot as plt
from gymnasium.wrappers import TransformAction, TransformObservation, RecordEpisodeStatistics, TimeLimit, TransformReward, RecordVideo
 
#create grids
N_EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.003
SEED = 42

# Visualización
CELL = 80
MARGIN = 8
FPS_TRAIN = 14
FPS_DEMO = 8


def epsilon_after(ep: int) -> float:
    return max(EPS_END, EPS_START - ep * EPS_DECAY)


class TaxiViewer:
    def __init__(self, env: gym.Env, title: str = "Taxi-v3 (Q-Learning)"):
        pygame.init()
        self.env = env
        self.clock = pygame.time.Clock()
        self.rows = 5
        self.cols = 5
        w = self.cols * CELL + 2 * MARGIN
        h = self.rows * CELL + 2 * MARGIN
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("consolas", 18)
        self.C_BG   = (24, 24, 28)
        self.C_GRID = (70, 70, 80)
        self.C_TAXI = (255, 210, 64)
        self.C_PAX  = (116, 220, 105)
        self.C_DEST = (98, 186, 255)
        self.C_TEXT = (220, 220, 235)
        self.locs = [tuple(x) for x in env.unwrapped.locs]

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def draw(self, state: int, info_text: str = ""):
        taxi_r, taxi_c, pass_loc, dest_idx = self.env.unwrapped.decode(state)
        self.screen.fill(self.C_BG)
        for r in range(self.rows):
            for c in range(self.cols):
                x = MARGIN + c * CELL
                y = MARGIN + r * CELL
                pygame.draw.rect(self.screen, self.C_GRID, (x, y, CELL, CELL), width=1)
        for i, (r, c) in enumerate(self.locs):
            x = MARGIN + c * CELL
            y = MARGIN + r * CELL
            rect = pygame.Rect(x+8, y+8, CELL-16, CELL-16)
            pygame.draw.rect(self.screen, (60,60,70), rect, border_radius=8)
            label = self.font.render("RGYB"[i], True, (180, 180, 190))
            self.screen.blit(label, (x + CELL//2 - 6, y + CELL//2 - 8))
        dest_r, dest_c = self.locs[dest_idx]
        dest_rect = pygame.Rect(MARGIN + dest_c*CELL + 12, MARGIN + dest_r*CELL + 12, CELL-24, CELL-24)
        pygame.draw.rect(self.screen, self.C_DEST, dest_rect, width=2, border_radius=6)
        if pass_loc < 4:
            pr, pc = self.locs[pass_loc]
            pax_rect = pygame.Rect(MARGIN + pc*CELL + 24, MARGIN + pr*CELL + 24, CELL-48, CELL-48)
            pygame.draw.rect(self.screen, self.C_PAX, pax_rect, border_radius=4)
        tx = MARGIN + taxi_c * CELL + 8
        ty = MARGIN + taxi_r * CELL + 8
        taxi_rect = pygame.Rect(tx, ty, CELL-16, CELL-16)
        pygame.draw.rect(self.screen, self.C_TAXI, taxi_rect, border_radius=10)
        if info_text:
            txt = self.font.render(info_text, True, self.C_TEXT)
            self.screen.blit(txt, (MARGIN, self.rows*CELL + MARGIN - 24))
        pygame.display.flip()

    def tick(self, fps: int):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()


def plot_training_png(rewards, lengths, td_errors, filename="taxi_training.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Recompensas por episodio
    axes[0].plot(rewards)
    axes[0].set_title("Recompensas por episodio")
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Recompensa total")

    # Duración de episodios
    axes[1].plot(lengths)
    axes[1].set_title("Duración de episodios")
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylabel("Pasos")

    # Error TD (por paso de entrenamiento)
    axes[2].plot(td_errors)
    axes[2].set_title("Error de entrenamiento (TD)")
    axes[2].set_xlabel("Actualización")
    axes[2].set_ylabel("TD Error (abs)")

    fig.suptitle("Entrenamiento del agente Taxi-v3")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def train_with_optional_view(n_episodes: int = N_EPISODES, visualize: bool = False, visualize_every: int = 0):
    base = gym.make("Taxi-v3", render_mode="rgb_array")
    base.reset(seed = SEED)
    #env  = TransformReward(base, lambda r: np.clip(0, 10))
    env = TimeLimit(base, max_episode_steps=100)
    env = RecordVideo(env, video_folder="taxi_videos", episode_trigger=lambda ep: ep % 1000 == 0)
    env = TransformReward(env, lambda r: r if r != -1 else -0.01)
    env = TransformObservation(env, lambda obs: obs, env.observation_space)
    env = RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    #sticky action no sirve ya que es un entorno discreto por lo que no hay acciones continuas
   
   
    #Record video, sticky actions,transform rewards, etc. could be added here
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    viewer = TaxiViewer(env, title="Taxi-v3 (training)") if visualize else None

    # --- Métricas ---
    td_errors = []          # por paso (para el gráfico 3)

    for ep in tqdm(range(n_episodes), desc="Entrenando Taxi-v3 (Q-Learning)"):
        s, _ = env.reset()
        done = False
        eps = epsilon_after(ep)

        show_this_ep = visualize and (visualize_every == 0 or ep % visualize_every == 0)

        while not done:
            if viewer and show_this_ep:
                if viewer._handle_events():
                    viewer.close(); env.close(); sys.exit(0)
                viewer.draw(s, f"ep {ep}  eps {eps:.2f}  return ")
                viewer.tick(FPS_TRAIN)
            
            # ε-greedy
            if np.random.random() < eps:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[s]))

            s2, r, term, trunc, _ = env.step(a)
            #s2 -> Estado siguiente
            #r -> Recompensa recibida
            #term -> Si el episodio ha terminado
            #trunc -> Si se ha truncado el episodio
            #r = -1 -> Siempre que no exista otra reward 
            #r = -10 -> Si hace pickup o drop cuando no corresponde
            #r = 20 -> Deja al pasajeero en el destion
            done = term or trunc

            # TD target y error
            best_next = 0.0 if done else float(np.max(Q[s2]))
            td = r + GAMMA * best_next - Q[s, a]
            td_errors.append(abs(td))                 # guardamos |TD|
            # r = -1 tf 0 -> td = -1 + 0 - Q[s,a] = -1 +0.01 = -0.99
            # r =  20 td = 20 +0.01 = 20.01
            # td_erros <- abs(-0-.99) = 0.99. abs(20.01) = 20.01


            # update
            Q[s, a] += ALPHA * td

            s = s2
        #print(f"Estadisticas:{list(env.return_queue)}")

    if viewer:
        viewer.close()
    env.close()

    # --- Guardar PNG con los 3 gráficos ---
    plot_training_png(list(env.return_queue), list(env.length_queue), td_errors, filename="taxi_training.png")

    return Q, list(env.return_queue),list(env.length_queue), td_errors


def demo_greedy(Q: np.ndarray, fps: int = FPS_DEMO):
    env = gym.make("Taxi-v3")
    s, _ = env.reset(seed=SEED)
    viewer = TaxiViewer(env, title="Taxi-v3 (demo greedy)")

    done = False
    total_r = 0.0
    steps = 0

    while not done:
        if viewer._handle_events():
            viewer.close(); env.close(); return
        viewer.draw(s, f"demo | steps {steps}  return {total_r:.1f}")
        viewer.tick(fps)
        a = int(np.argmax(Q[s]))
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        total_r += r
        steps += 1

    for _ in range(12):
        if viewer._handle_events():
            break
        viewer.draw(s, f"FINAL | steps {steps}  return {total_r:.1f}")
        viewer.tick(fps)

    viewer.close(); env.close()

if __name__ == "__main__":
    # Entrena (puedes poner visualize=False para ir más rápido)
    Q, rewards, lengths, td = train_with_optional_view(N_EPISODES, visualize=False, visualize_every=50)
    # Demo final
    #demo_greedy(Q)
