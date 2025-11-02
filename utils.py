import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# Funciones para visualización
# -----------------------------
def plot_training(env, agent, plot_save=False, rolling_length=500):
 
       
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    fig.suptitle("Entrenamiento del agente spaceInvaders", fontsize=16)

    # Recompensas
    reward_ma = np.convolve(np.array(env.return_queue).flatten(),
                            np.ones(rolling_length), mode="valid") / rolling_length
    axs[0].plot(range(len(reward_ma)), reward_ma)
    axs[0].set_title("Recompensas por episodio")

    # Duración de episodios
    length_ma = np.convolve(np.array(env.length_queue).flatten(),
                            np.ones(rolling_length), mode="same") / rolling_length
    axs[1].plot(range(len(length_ma)), length_ma)
    axs[1].set_title("Duración de episodios")

    # Error de entrenamiento
    td_ma = np.convolve(np.array(agent.training_error),
                        np.ones(rolling_length), mode="same") / rolling_length
    axs[2].plot(range(len(td_ma)), td_ma)
    axs[2].set_title("Error de entrenamiento (TD)")

    plt.tight_layout()
    if plot_save:
        plt.savefig("spaceInvaders.png")
    else:
        plt.show()

def create_grids(agent, usable_ace=False):
    """Create grids for state values and policies.

    Args:
        agent (SpaceInvaders): The trained Blackjack agent.
        usable_ace (bool, optional): Whether to include usable ace in the state. Defaults to False.

    Returns:
        tuple: A tuple containing the value grid and policy grid.
    """

    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(np.arange(12, 22), np.arange(1, 11))
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(player_count, dealer_count, value, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    plt.xticks(range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title("Valores de estado")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    ax2 = fig.add_subplot(1, 2, 2)
    sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False, ax=ax2)
    ax2.set_title("Política")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig