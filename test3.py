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
import xlsxwriter
import pandas.io.formats.excel
import getpass
from matplotlib import rc
import matplotlib.pyplot as plt


## This class contains the components necessary for running and evaluating a historical simulation.

class SimulationEnv(object):
    
    loggables = ['positions']
    
    def __init__(self, wealth, assets, 
                 path_to_data, start_date, end_date,
                 agent_type, expert_type, reinvest):

        self.init_wealth = wealth
        self.wealth = wealth
        self.assets = assets
        self.agent_type = agent_type
        self.expert_type = expert_type
        self.path_to_data = path_to_data
        self.reinvest = reinvest
        ## TODO: 
        ## - specify start and end date for simulation
        ## - collect statistics on simulation results
        
        return
    
    
    ## To be called before running a simulation - initialize experts, agents, and initial position, etc.

    def setup_params(self, agent_args={}, expert_args={}):
    
        self.experts = [self.expert_type(a, self.path_to_data, **expert_args) for a in self.assets]
        self.agent = self.agent_type(self.experts, **agent_args)
        self.positions = np.array([weight * self.wealth for weight in self.agent.weights])
        
        ## TODO: 
        # do various other stuff here like for select for high-volatility stocks or something
        # exception handling
        # Need to make sure data files are in sync
        
        return
        
    ## Run simulation
    def run(self, log=False, logpath=os.getcwd()):
        
        ## Start period counter and log
        self.period = 1
        self.finallog = []
        self.logpath = logpath
        self.log = log
        
        ## Warmup period: 
        ## i.e. for strategies involving moving average indicators, wait until we have enough data to calculate MA
        for expert in self.agent.experts:
            expert.warmup()
        
        ## Simulation: Go until data iterators reach the end
        while True:
            try:

                ## Log this period's results
                if log:
                    self.logperiod()

                    
                ## Update experts with last period's rewards
                for i, expert in enumerate(self.agent.experts):
                    expert.update()
                    if self.reinvest:# below is to redistribute unallocated money
                        if expert.pick == False:
                            extra_money = self.positions[i]
                            self.positions[i] = 0
                            distribute = np.greater(self.positions, 0)
                            if not np.any(distribute): # this is to avoid divide by zero if nothing is invested
                                #print("all experts say not to buy")
#                                print self.positions
#                                print [e.pick for e in self.agent.experts]
                                self.positions = np.array([weight * self.wealth for weight in self.agent.weights])
#                                print type(self.positions)
                            else:
#                                print self.positions
#                                print [e.pick for e in self.agent.experts]
                                distribute_extra_money = distribute * (extra_money/ np.sum(distribute))
                                self.positions = np.add(self.positions, distribute_extra_money)
#                                print type(self.positions)
                            assert(abs(np.sum(self.positions)-self.wealth) <= 1)# make sure all money is accounted for
                      
                ## Update agent accordingly (i.e. for Hedge, update weights according to each expert's reward in the last period)
                self.agent.update()
                

                
                ## Update position with returns from last period
                self.positions = self.positions * (1 + self.agent.returns)
                self.wealth = np.sum(self.positions)
                
                ## Rebalance according to new, updated weights
                ## TODO: only allow non-fractional shares to be purchased (?)
                if np.sum(self.agent.weights) == 1:
                    self.positions = np.array([weight * self.wealth for weight in self.agent.weights])
                    
                ## Advance period
                self.period += 1
                
            except StopIteration:

                break
        
        ## Write the log file
        if log:
            self.savelog()

    
    def logperiod(self):
        row = [self.period] + [self.wealth]
        nrow = []
        for loggable in (self.loggables):
            if getattr(self,loggable) is None:
                nrow += [None] * len(self.experts)
            else:
                nrow += getattr(self,loggable).tolist()
        arow = []
        for loggable in (self.agent_type.loggables):
            if getattr(self.agent,loggable) is None:
                arow += [None] * len(self.experts)
            else:
                arow += getattr(self.agent,loggable).tolist()
        erow = []
        for loggable in (self.expert_type.loggables):
            if [getattr(e,loggable) for e in self.experts] is None:
                erow += [None] * len(self.experts)
            else:
                erow += [getattr(e,loggable) for e in self.experts]
        row += nrow + arow + erow
        self.finallog.append(row)
    
    def savelog(self):
        runtime = datetime.datetime.now()
        runuser = getpass.getuser()
        logname = runuser + "_" + runtime.strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(os.path.join(self.logpath, logname)):
            os.makedirs(os.path.join(self.logpath, logname))
            os.makedirs(os.path.join(self.logpath, logname, 'plots'))
        col = ['period', 'wealth'] + \
            [x+'.'+y for x in self.loggables for y in self.assets] + \
            [x+'.'+y for x in self.agent_type.loggables for y in self.assets] + \
            [x+'.'+y for x in self.expert_type.loggables for y in self.assets]
        df = pd.DataFrame(self.finallog, columns=col)
        df.set_index('period', inplace=True)
        
        
        writer = pd.ExcelWriter(os.path.join(self.logpath, logname, logname+'.xlsx'), engine='xlsxwriter')
        pd.io.formats.excel.header_style = None
        df.to_excel(writer,'run_log')
        
        workbook  = writer.book
        worksheet = writer.sheets['run_log']
        
        header_format = workbook.add_format({'bold': True,'text_wrap': True})
        worksheet.set_row(0, None, header_format)
        writer.save()
        
        # Set up matplotlib
        rc('font', family = 'serif', serif = 'cmr12')
        
        plot = ['wealth', 'weights','positions','rewards']
        for p in plot:
            for legend in [True, False]:
                plotdf = df.filter(regex='period|'+p)
                plotlab = [l.split(p+'.',1)[1] if l != 'wealth' else l.title() for l in list(plotdf.columns.values)]
                plt.plot(plotdf)
                plt.ylabel(p.title())
                plt.xlabel('Round')
                if legend:
                    plt.legend(plotlab, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=4)
                    plt.savefig(os.path.join(self.logpath, logname, 'plots', p+'_legend.png'), bbox_inches='tight', dpi=300)
                else:
                    plt.savefig(os.path.join(self.logpath, logname, 'plots', p+'.png'), bbox_inches='tight', dpi=300)
                plt.close()


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

    loggables = ['returns','rewards','weights']
    
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
    
    loggables = ['avg','std']
    
    def __init__(self, name, path_to_data, window_size, threshold):
        self.reward = 0.
        self.pick = False
        self.data = pd.read_csv(path_to_data + name + ".csv", iterator=True, chunksize=1)
        self.last_price = float(self.data.get_chunk(1)["adj_close"])
        self.window_size = window_size
        self.avg = 0.0
        self.std = None
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

s = SimulationEnv(100000, stocks, "data/djia_20150101_20171101/", "2015-01-01", "2017-11-01", Hedge, MeanReversion, True)
s.setup_params(agent_args = {"eta": -0.01}, expert_args={"window_size": 10, "threshold": .5})
s.run(log=True, logpath="logs")
print s.wealth