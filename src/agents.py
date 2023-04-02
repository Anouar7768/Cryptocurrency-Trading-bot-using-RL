#!/usr/bin/env python

from RLGlue.agent import BaseAgent

import numpy as np

class Agent(BaseAgent):
    """agent does *no* learning, selects action 0 always"""
    def __init__(self):
        self.action = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.portfolio = agent_info['portfolio']
        self.model = agent_info["model"]
        self.possible_action = [0,-1,1]

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
        reward, info, obs,term = observation
        self.obs = obs
        ### hold = 0, sell = 1, buy = 2
        self.info_market = info #np array 
        self.action = 0  # set first action to 0
        return self.portfolio
    
    def NUPL2current(self,reward):
        return -self.portfolio/(reward -1)

    def get_action(self,prediction):
        reals = [(self.portfolio + action)*prediction  for action in [0,-1,1]]
        NUPL = [(prediction - r)/prediction for r in reals]
        return np.argmax(NUPL)
    
    def agent_step(self, reward,observation):
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
        obs = observation
        market_value = self.NUPL2current(reward)
        result = np.array(obs.copy()+[market_value])
        self.info_market = np.vstack((self.info_market,result))
        X = np.copy(self.info_market[:,:-1])
        y = self.info_market[:,-1]
        self.model.fit(X,y)
        prediction = self.model.predict(np.array(obs).reshape(1, -1))
        self.obs = obs

        self.action = self.get_action(prediction)
        self.portfolio += self.possible_action[self.action]

        return self.portfolio

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