# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:05:58 2017

@author: Gregory
"""

import datetime
import pandas as pd
import time
import requests
import os
import numpy as np
import Queue
from Queue import Queue
from collections import deque

import getpass


## This class contains the components necessary for running and evaluating a historical simulation.

class SimulationEnv(object):
    
    def __init__(self, wealth, assets, 
                 path_to_data, start_date, end_date,
                 agent_type, expert_type):

        self.init_wealth = wealth
        self.wealth = wealth
        self.assets = assets
        self.agent_type = agent_type
        self.expert_type = expert_type
        self.path_to_data = path_to_data
        ## TODO: 
        ## - specify start and end date for simulation
        ## - collect statistics on simulation results
        
        return
    
    
    ## To be called before running a simulation - initialize experts, agents, and initial position, etc.

    def setup_params(self, agent_args={}, expert_args={}):
    
        self.experts = [self.expert_type(a, self.path_to_data, **expert_args) for a in self.assets]
        self.agent = self.agent_type(self.experts, **agent_args)
        self.positions = [weight * self.wealth for weight in self.agent.weights]
        
        ## TODO: 
        # do various other stuff here like for select for high-volatility stocks or something
        # exception handling
        # Need to make sure data files are in sync
        
        return
        
    ## Run simulation
    def run(self, log=False, logpath=os.getcwd()):
        
        if log:
            runtime = datetime.datetime.now()
            runuser = getpass.getuser()
            logname = runuser + "_" + runtime.strftime('%Y-%m-%d_%H-%M-%S')
            if not os.path.exists(os.path.join(logpath, logname)):
                os.makedirs(os.path.join(logpath, logname))
          
            
        ## Warmup period: 
        ## i.e. for strategies involving moving average indicators, wait until we have enough data to calculate MA
        for expert in self.agent.experts:
            expert.warmup()

        ## Simlation: Go until data iterators reach the end
        while True:
            try:
                print self.positions
                ## Update experts with last period's rewards
                for expert in self.agent.experts:
                    expert.update()
                ## Update agent accordingly (i.e. for Hedge, update weights according to each expert's reward in the last period)
                self.agent.update()
                
                ## Update position with returns from last period
                self.positions = self.positions * (1 + self.agent.returns)
                self.wealth = np.sum(self.positions)
                
                ## Rebalance according to new, updated weights
                ## TODO: only allow non-fractional shares to be purchased (?)
                self.positions = [weight * self.wealth for weight in self.agent.weights]
                
            except StopIteration:
                break
        



from abc import ABCMeta, abstractmethod

class Agent(object):
    __metaclass__ =ABCMeta
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def update(self):
        pass
    
    @abstractmethod
    def act(self):
        pass
    
class Expert(object):
    __metaclass__=ABCMeta
    
    @abstractmethod
    def __init__(self, reward_data):
        pass
    
    @abstractmethod
    def update(self):
        pass
    
class Context(object):
    __metaclass__=ABCMeta
    
    @abstractmethod
    def __init__(self, context_data):
        pass
    
    @abstractmethod
    def observe(self):
        pass
    
## Hedge. (http://www.cis.upenn.edu/~mkearns/finread/helmbold98line.pdf)

class Hedge(Agent):
    
    ## Initialize with a set of experts
    def __init__(self, experts, eta):
        self.eta = eta
        self.experts = experts
        self.weights = np.ones(len(self.experts))/len(self.experts)
        self.rewards = None
        self.returns = None
        return
    
    ## Update the agent's rewards and its weights for each expert
    def update(self):
        self.rewards = np.asarray([e.reward for e in self.experts])
        self.returns = np.asarray([e.returns for e in self.experts])
        multipliers = np.exp(self.eta * self.rewards/np.sum(self.weights * self.rewards))
        self.weights = (self.weights * multipliers)/ np.sum(self.weights * multipliers)
        self.weights = np.nan_to_num(self.weights)

        return
    
    def act(self):
        return self.weights

## Dummy expert that always pick the same asset
class Dummy(Expert):
    
    ## Expert has a reward associated with its pick
    def __init__(self, name, path_to_data):
        self.reward = 0.
        self.names = name
        self.pick = True ## Might not be necessary
        self.data = pd.read_csv(path_to_data + name + ".csv", iterator=True, chunksize=1)
        self.last_price = float(self.data.get_chunk(1)["adj_close"])
        return
    
    ## Expert updates its reward 
    def update(self):
        current_price = float(self.data.get_chunk(1)["adj_close"])
        self.reward = (current_price - self.last_price)/self.last_price
        self.returns = self.reward
        self.last_price = current_price
        return
    
    def warmup(self):
        pass
    

class MeanReversion(Expert):
    
    def __init__(self, name, path_to_data, window_size, threshold):
        self.reward = 0.
        self.pick = False
        self.data = pd.read_csv(path_to_data + name + ".csv", iterator=True, chunksize=1)
        self.last_price = float(self.data.get_chunk(1)["adj_close"])
        self.window_size = window_size
        self.avg = 0.0
        self.std = 0.0
        self.threshold = threshold
        self.last_n_prices = Queue(maxsize=10)
        self.returns = 0.
        return
    
    def warmup(self):
        n = 1
        while n <= self.window_size:
            self.last_n_prices.put(self.last_price)
            self.last_price = float(self.data.get_chunk(1)["adj_close"])
            n += 1
        return
        
    def update(self):
        _ = self.last_n_prices.get()
            
        self.last_n_prices.put(self.last_price)
        self.avg = np.mean(list(self.last_n_prices.queue))
        self.std = np.std(list(self.last_n_prices.queue))
        
        current_price = float(self.data.get_chunk(1)["adj_close"])

        ## If self.pick is True, we bought the stock and our reward is whatever the return was in the last period
        if self.pick:
            self.reward = (current_price - self.last_price)/self.last_price
            self.returns = self.reward
        else:
            self.reward = -(current_price - self.last_price)/self.last_price
            self.returns = 0
        self.last_price = current_price

        if self.last_price <= self.avg - self.threshold * self.std:
            self.pick = True
        else:
            self.pick = False

        return
    
    
class VolContext(Context):
    
    def __init__(self, data_file, threshold):
        self.data = pd.read_csv(data_file, iterator=True, chunksize=1)
        self.threshold = threshold
        self.contexts = ["HighVol", "LowVol"]
        return
    
    # Returns a string giving the name of the context
    def observe(self):
        if float(self.data.get_chunk(1)["adj_close"]) > self.threshold:
            return self.contexts[0]
        else:
            return self.contexts[1]
      
## Some other examples of agents to use as benchmarks 

class NaiveBuyHold(Agent):
    
    def __init__(self, experts):
        self.experts = experts
        self.weights = np.ones(len(self.experts))/len(self.experts)
        self.rewards = None
        return
    
    def update(self):
        self.rewards = np.asarray([e.reward for e in self.experts])
        self.returns = self.rewards
        return
    
    def act(self):
        return self.weights
    
## TODO
class WeightedBuyHold(Agent):
    
    def __init__(self):
        return
    
    def update(self):
        return
    
    def act(self):
        return
    

## TODO
class ConstantRebalancer(Agent):
    
    def __init__(self):
        return
    
    def update(self):
        return
    
    def act(self):
        return

stocks = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE',
 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 
'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM']

## Run a simulation with Hedge, each expert is a proxy for buying a given stock

s = SimulationEnv(100000, stocks, "data/djia_20150101_20171101/", "2015-01-01", "2017-11-01", Hedge, MeanReversion)
s.setup_params(agent_args = {"eta": -0.01}, expert_args={"window_size": 10, "threshold": .5})
s.run(log=True, logpath="logs")
print s.wealth