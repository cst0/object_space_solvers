from tqdm import tqdm
import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env_hoop import HoopGripperPushEnv

# Create the environment
env = HoopGripperPushEnv()


def train_and_eval(model, env, eval_every=1000, total_learning_time=100000):
    # Train the agent
    n_steps = total_learning_time // eval_every
    training_results = pd.DataFrame()
    for step in tqdm(range(n_steps)):
        model.learn(total_timesteps=eval_every)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        training_results.loc[step, "mean_reward"] = mean_reward
        training_results.loc[step, "std_reward"] = std_reward
    return training_results

start_time = time.time()
all_results = pd.DataFrame()
for n in range(10):
    print(f"Training run {n}")
    model = PPO("MlpPolicy", env, verbose=1)
    results = train_and_eval(model, env)
    all_results[f"run_{n}"] = results["mean_reward"]

# save the results
all_results.to_csv(f"hoop_ppo_objectspace_{start_time}.csv")
