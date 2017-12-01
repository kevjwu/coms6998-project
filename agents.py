import datetime
#from datetime import datetime
import pandas as pd
import time
import requests
import os
import numpy as np
import xlsxwriter
from abc import ABCMeta, abstractmethod
from Queue import Queue
from collections import deque
import getpass
import pandas.io.formats.excel
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt


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

## EG. (http://www.cis.upenn.edu/~mkearns/finread/helmbold98line.pdf)
class EG(Agent):
    
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


class EGRecent(Agent):
    
    loggables = ['returns','rewards','weights']
    
    ## Initialize with a set of experts
    def __init__(self, experts, eta, n):
        self.eta = eta
        self.experts = experts
        #self.weights = Queue(maxsize=n)
        self.weights = np.ones(len(self.experts))/len(self.experts)
        self.rewards = deque(maxlen=n)
        self.returns = None
        self.t = 0
        self.n = n
        #self.rewards = None
        #self.returns = None
        return
    
    ## Update the agent's rewards and its weights for each expert
    def update(self):
           
        self.rewards.append(np.asarray([e.reward for e in self.experts]))
        self.returns = np.asarray([e.returns for e in self.experts])
        
        self.weights = np.ones(len(self.experts))/len(self.experts)
        #print self.t
        #print len(self.rewards)
        for rewards in self.rewards:
            rewards = np.asarray(rewards)
            multipliers = np.exp(self.eta * rewards/np.sum(self.weights * rewards))
            self.weights = (self.weights * multipliers)/ np.sum(self.weights * multipliers)
            self.weights = np.nan_to_num(self.weights)
        self.t += 1
        #print self.t
        
        return
    
    def act(self):
        return self.weights
