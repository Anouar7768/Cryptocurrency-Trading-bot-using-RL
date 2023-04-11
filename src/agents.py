#!/usr/bin/env python

from RLGlue.agent import BaseAgent

import numpy as np


class Agent(BaseAgent):
    """agent does *no* learning, selects action 0 always"""

    def __init__(self):
        self.action = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.cash = agent_info['cash']  # total cash of the agent
        self.model = agent_info["model"]  # model for prediction
        self.crypto_name = agent_info['crypto']  # name of the considered crypto

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        reward, info, obs, term = observation
        self.obs = obs
        ### hold = 0, sell = 1, buy = 2
        self.info_market = info  # np array
        # Initiate the portfolio, buy for each crypto 1/4 of the total money amount
        last_values = self.info_market[0, :]  ##TODO change this to get the real last values
        self.portfolio = {}
        for i, name in enumerate(self.crypto_name):
            self.portfolio[name] = [((self.cash / 4) / last_values[i], last_values[i])]  # (num crypto , purchase price)

        self.cash = self.cash / 4  # update cash

        self.action = [0, 0, 0]  # hold everything

        return self.portfolio, self.action, self.cash / 4

    def get_possible_portfolio(self, crypto, current):
        """ Method to get the associated portfolios of the different possible action according to the price prediction 
        Args:
            crypto: the crypto name to consider
            current: its price prediction
        Returns:
            list of possible portfolio
        """

        hold_portolio = self.portfolio[crypto].copy()  # just the same
        if self.cash != 0:  # enough money to buy
            buy_portfolio = self.portfolio[crypto].copy() + [((self.cash / 3) / current, current)]
        else:
            buy_portfolio = self.portfolio[crypto].copy()  # like hold

        return [hold_portolio, buy_portfolio]

    def get_action(self, prediction):
        """Method to get the action that max the NUPL for each crypto
        Args:
            prediction : an array that gives the price prediction for each crypto
        Returns:
            Action taken : a list of values 0,1 or 2 (hold, buy, sell)
        """
        action = []
        for i, crypto in enumerate(self.crypto_name):  # compute the 3 possible portfolio
            NUPLs = []
            total = sum([nb for nb, value in self.portfolio[crypto]])  # nb of crypto
            current = prediction[0][i]  # prediction for the crypto
            possible_portfolio = self.get_possible_portfolio(crypto, current)
            for portfolio in possible_portfolio:
                if total != 0:
                    NUPLs.append(sum([(current - value) * nb for nb, value in portfolio]) / total * current)
                else:
                    NUPLs.append(0)
            NUPLs.append(0)  # NUPL for sell
            if NUPLs == [0, 0, 0]:
                action.append(1)  # force to buy
            else:
                action.append(np.argmax(NUPLs))

        return action

    def update(self, market_values):
        """Method to update cash and portfolio accorrding to the trye current market value given by the env
        Args:
            market_values : the true value
        """
        for i, crypto in enumerate(self.crypto_name):
            action = self.action[i]
            if action == 1:  # buy
                self.portfolio[crypto].append((self.cash / 4 / market_values[i], market_values[i]))
                self.cash -= self.cash / 4
            if action == 2:  # sell
                total = sum([nb for nb, value in self.portfolio[crypto]])
                self.portfolio[crypto] = []
                self.cash += total * market_values[i]

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward : the reward received for taking the last action taken, sum of the NUPL for each crypto
            observation: the state observation from the
                environment's step based, where the agent ended up after the
                last step, new sentiment value and past true value
        Returns:
            The action the agent is taking.
        """

        obs, market_values = observation
        total_NUPL = reward
        self.update(market_values)

        ############ Prediction step ##########
        X = np.copy(self.info_market) ##TODO adapt this code line depending on how the real prices are added to data

        prediction = X

        self.obs = obs

        self.action = self.get_action(prediction)

        return self.portfolio, self.action, self.cash / 4

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
