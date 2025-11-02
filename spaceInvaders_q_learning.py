import numpy as np
from tqdm import tqdm
import gymnasium as gym
from collections import defaultdict
from utils import plot_training 

class SpaceInvaders_agent:
    def __init__ (
        self,
        env,
        learning_rate: float,
        intial_epsilon: float,
        epsilon_decay: float,
        final_epsilon:float,
        discount_factor: float = 0.95
    ):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = intial_epsilon #Probabilidad con la que cogemos una acción aleatoria
        self.epsilon_decay = epsilon_decay 
        self.final_epsilon = final_epsilon
        self.training_error = []
    
    def get_action(self,env,obs: tuple[int,int,bool]) -> int:
        if np.random.rand() < self.epsilon: #Epsilon es la probabilidad de exploración
            return env.action_space.sample()
        return np.argmax(self.q_values[obs])
        
    def update(self, obs, action,reward,terminated,next_obs):
        best_next_action = (not terminated) * np.argmax(self.q_values[next_obs])
        td_error = reward + self.discount_factor * best_next_action -self.q_values[obs][action]      
        self.q_values[obs][action] += self.lr * td_error
        self.training_error.append(abs(td_error))
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def train_agent(n_repetitions=10):
        env = gym.make("CartPole-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_repetitions)
        agent = SpaceInvaders_agent(
            env= env,
            learning_rate=0.001,
            intial_epsilon=1.0,
            epsilon_decay=0.01,
            final_epsilon=0.1,
        )
        
        for episode in tqdm(range(n_repetitions)):
            obs, info = env.reset()
            obs = tuple(obs)
            terminated = False
            
            while not terminated:
                action = agent.get_action(env, obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = tuple(next_obs)
     
                agent.update(obs, action, reward, terminated, next_obs)
                obs = next_obs
            
            agent.decay_epsilon()
        
        return env, agent 
    





if __name__ == "__main__":
    env, trained_agent = train_agent(n_repetitions=10_000)   
    plot_training(env, trained_agent, plot_save=True)    
    env.close()
    
    


