import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from RLGlue.rl_glue import RLGlue

from src.environment import Environment
from src.agents import Agent


data = pd.read_csv("./data/all_data.csv").to_dict() # full data with price and sentiment
num_obs = max([k for k in data['Date'].keys()])

env = Environment
agent = Agent

env_info = {'max' : num_obs , 'data':data}
agent_info = {'model' : pd.read_csv("./data/all_data_with_sentiment.csv").to_dict(), 'cash':100000, 'crypto': ['BNB','BTC','ETH']}


rl_glue = RLGlue(env, agent)  # Creates a new RLGlue experiment with the env and agent we chose 
rl_glue.rl_init(agent_info, env_info) # Pass RLGlue what it needs to initialize the agent and environment
rl_glue.rl_start() 




num_steps = 1

total_nupl = []
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

