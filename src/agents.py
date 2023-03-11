#!/usr/bin/env python

from RLGlue.agent import BaseAgent

import numpy as np

class Agent(BaseAgent):
    """agent does *no* learning, selects action 0 always"""
    def __init__(self):
        self.action = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.portfolio = agent_info

        self.action = 0

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        ### hold = 0, sell = 1, buy = 2
        self.action = 0  # set first action to 0

        return self.action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        # implement policy 
        #observation is a dict with keys ['weighted_positive_score','weighted_neutral_score','weighted_negative_score',
        # 'total','Date','Open','High','Low','Close','Adj Close','Volume']
        # reward is NUPL
        self.action = 0

        return self.action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        pass