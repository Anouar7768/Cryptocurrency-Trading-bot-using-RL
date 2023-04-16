import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from RLGlue.rl_glue import RLGlue

from src.environment import Environment
from src.agents import GreedyAgent, EpsilonGreedyAgent
import matplotlib.pyplot as plt

# Parameters to play with
with_sentiment = False
epsilon_greedy = True
data = pd.read_csv("./data/all_data.csv").to_dict() # full data with price

if with_sentiment:
        predictions = pd.read_csv("./data/all_data_with_sentiment.csv")
else:
        predictions = pd.read_csv("./data/all_data_without_sentiment.csv")

epsilons = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1]
for epsilon in epsilons:


        num_obs = max([k for k in data['Date'].keys()])



        env = Environment
        if epsilon_greedy:
                agent = EpsilonGreedyAgent
                agent_info = {'model': pd.read_csv("./data/all_data_with_sentiment.csv").to_dict(), 'cash': 100000,
                              'crypto': ['BTC', 'BNB', 'ETH'], 'epsilon': epsilon}
        else:
                agent = GreedyAgent
                agent_info = {'model': pd.read_csv("./data/all_data_with_sentiment.csv").to_dict(), 'cash': 100000,
                              'crypto': ['BTC', 'BNB', 'ETH']}


        env_info = {'max': num_obs, 'data': data}



        rl_glue = RLGlue(env, agent)  # Creates a new RLGlue experiment with the env and agent we chose
        rl_glue.rl_init(agent_info, env_info) # Pass RLGlue what it needs to initialize the agent and environment
        rl_glue.rl_start()


        num_steps = 200
        total_nupl = []
        date = np.array(predictions.Date)
        print(date)
        actions = np.zeros(3)
        for i in range(num_steps):
                reward, obs, action, done = rl_glue.rl_step()
                print("iteration:",i)
                print("Total NUPL:", reward)
                print("New observation:", obs[0], obs[1])
                print("Agent Portfolio:", action[0])
                print("Action taken:", action[1])

                total_nupl.append(reward)
                actions = np.vstack((actions,action[1]))
                if done:
                        break

        total_nupl_df = pd.DataFrame(total_nupl)

        total_nupl_df.to_csv(f"./data/NUPLS_sentiment_{with_sentiment}_{epsilon}_{epsilon_greedy}.csv", index=False)

