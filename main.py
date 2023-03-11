import numpy as np
import pandas as pd
from RLGlue.rl_glue import RLGlue

from src.environment import Environment
from src.agents import Agent


data = pd.read_csv("./data/Data.csv").to_dict() # full data with price and sentiment
num_obs = max([k for k in data['Date'].keys()])

env = Environment
agent = Agent

env_info = {'max' : num_obs , 'data':data}

rl_glue = RLGlue(env, agent)  # Creates a new RLGlue experiment with the env and agent we chose 
rl_glue.rl_init({}, env_info) # Pass RLGlue what it needs to initialize the agent and environment
rl_glue.rl_start() 

num_steps = 100

for i in range(num_steps):
        reward, obs, action, done = rl_glue.rl_step()
        print("iteration:",i)
        print(reward, obs, action, done)
        if done:
                break


