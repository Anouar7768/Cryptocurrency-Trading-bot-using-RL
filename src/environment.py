#!/usr/bin/env python

from RLGlue.environment import BaseEnvironment

import numpy as np


class Environment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    actions = [0]

    def __init__(self):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)
        self.count = 0
        self.data = None
        self.seed = None
        self.time = 0
        self.max_time = None

    def get_full_obs(self):
        infos = ['BTC', 'BNB', 'ETH'] ##TODO add real prices of cryptos on the same row

        return [self.data[key][self.time] for key in infos]

    def get_obs(self):
        infos = ['BTC', 'BNB', 'ETH']

        return [self.data[key][self.time] for key in infos]

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        self.max_time = env_info['max']
        self.data = env_info['data']
        local_observation = []
        for _ in range(32):
            local_observation += [self.get_full_obs()]
            self.time += 1

        self.reward_obs_term = (0.0, np.array(local_observation), self.get_obs(), False)

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment. Gives 5 points to agent
        """
        return self.reward_obs_term

    def NUPL(self, portfolio, current):
        """Method to get NUPLS fro each crypto according to agent portfolio
        Args:
            portfolio : agent portfolio
            current : the current market value for each crypto
        Returns:
            NUPLs
        """
        NUPLs = []
        for i, crypto in enumerate(['BTC', 'BNB', 'ETH']):
            total = sum([nb for nb, value in portfolio[crypto]])
            if portfolio[crypto] != []:
                NUPLs.append(sum([(current[i] - value) * nb for nb, value in portfolio[crypto]]) / total * current[i])
            else:
                NUPLs.append(0)
        return NUPLs

    def update_agent_portfolio(self, market_values, action, portfolio, cash):
        """Method update virtually the agent portfolio according to the action he chose
        Args:
            portfolio : agent portfolio
            market_values : the current market value for each crypto
            action: the actions taken by the agent 
            cash : agent cash for teh buy action

        Returns:
            portfolio 
        """
        for i, crypto in enumerate(['BTC', 'BNB', 'ETH']):
            if action[i] == 1:
                portfolio[crypto].append((cash / market_values[i], market_values[i]))
            if action[i] == 2:
                portfolio[crypto] = []
        return portfolio

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        current = self.get_full_obs()  ##TODO add curent price in order to compute NUPL
        portfolio, action_per_crypto, cash = action
        portfolio = self.update_agent_portfolio(current, action_per_crypto, portfolio, cash)
        reward = sum(self.NUPL(portfolio, current))
        self.time += 1

        obs = self.get_obs()

        if self.time != self.max_time:
            self.reward_obs_term = (reward, (obs, current), False)
        else:
            self.reward_obs_term = (reward, (obs, current), True)
        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_obs_term[0])

        # else
        return "I don't know how to respond to your message"
