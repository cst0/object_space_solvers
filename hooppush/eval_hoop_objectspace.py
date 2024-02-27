from tqdm import tqdm
import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env_hoop_objectspace import HoopObjectEnv

# Create the environment
env = HoopObjectEnv()
starttime = time.time()

def train_and_eval(
    model, env, eval_every=1000, total_learning_time=1000000, current_iter=0, df=None
):
    if df is None:
        df = pd.DataFrame()
    # Train the agent
    n_steps = total_learning_time // eval_every
    training_results = pd.DataFrame()
    for step in tqdm(range(n_steps)):
        model.learn(total_timesteps=eval_every)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=3)
        training_results.loc[step, "mean_reward"] = mean_reward
        # training_results.loc[step, "std_reward"] = std_reward
        df[f"run_{n}"] = training_results["mean_reward"]
        df.to_csv(f"hoop_ppo_objectspace_{starttime}.csv")
    return training_results


all_results = pd.DataFrame()
for n in range(10):
    print(f"Training run {n}")
    model = PPO("MlpPolicy", env, verbose=1)
    results = train_and_eval(model, env, current_iter=n, df=all_results)
