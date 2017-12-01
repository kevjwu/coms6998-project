import datetime
from datetime import datetime
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
from matplotlib import rc
import matplotlib.pyplot as plt


class Expert(object):
    __metaclass__=ABCMeta
    
    ## Required parameters for initializing experts
    @abstractmethod
    def __init__(self, name, path_to_data, start_date, end_date):
        self.reward = 0.
        self.pick = False ## Need to activate experts in child classes
        self.data = pd.read_csv(path_to_data + name + ".csv", iterator=True, chunksize=1)
        self.current_date = datetime.strptime("1000-01-01", "%Y-%m-%d")
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        while self.current_date < self.start_date:
            self.last_row = self.data.get_chunk(1)
            self.last_price = float(self.last_row["adj_close"])
            self.current_date = datetime.strptime(self.last_row["date"].item(), "%Y-%m-%d")
        
    @abstractmethod
    def update(self):
        pass
## Dummy expert that always pick the same asset
class Dummy(Expert):
    
    loggables = ['reward']
    
    ## Expert has a reward associated with its pick
    def __init__(self, name, path_to_data, start_date, end_date):
        super(Dummy, self).__init__(name, path_to_data, start_date, end_date)
        self.pick = True
        return
    
    ## Expert updates its reward 
    def update(self):
        self.last_row = self.data.get_chunk(1)
        self.current_date = datetime.strptime(self.last_row["date"].item(), "%Y-%m-%d")
        
        if self.current_date > self.end_date:
            raise StopIteration
            
        current_price = float(self.last_row["adj_close"])
        self.reward = current_price/self.last_price
        self.returns = self.reward - 1.
        self.last_price = current_price
        return
    
    def warmup(self):
        pass
    

class MeanReversion(Expert):
    
    loggables = ['avg','std']
    
    def __init__(self, name, path_to_data, start_date, end_date, window_size, threshold):
        super(MeanReversion, self).__init__(name, path_to_data, start_date, end_date)
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
            
            self.last_row = self.data.get_chunk(1)
            self.current_date = datetime.strptime(self.last_row["date"].item(), "%Y-%m-%d")

            self.last_price = float(self.last_row["adj_close"])
            n += 1
        return
        
    def update(self):
        _ = self.last_n_prices.get()
        
        self.last_n_prices.put(self.last_price)
        self.avg = np.mean(list(self.last_n_prices.queue))
        self.std = np.std(list(self.last_n_prices.queue))
        
        self.last_row = self.data.get_chunk(1)
        self.current_date = datetime.strptime(self.last_row["date"].item(), "%Y-%m-%d")
        
        if self.current_date > self.end_date:
            raise StopIteration
            
        current_price = float(self.last_row["adj_close"])
        
        ## If self.pick is True, we bought the stock and our reward is whatever the return was in the last period
        if self.pick:
            self.reward = current_price/self.last_price
            self.returns = self.reward - 1.
        else:
            self.reward = self.last_price/current_price
            self.returns = 0
            
        self.last_price = current_price

        if self.last_price <= self.avg - self.threshold * self.std:
            self.pick = True
        else:
            self.pick = False

        return

    
class Momentum(Expert):
    
    loggables = ['avg','std']
    
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
            self.reward = current_price/self.last_price
            self.returns = self.reward - 1.
        else:
            self.reward = -current_price/self.last_price
            self.returns = 0
        self.last_price = current_price

        if self.last_price >= self.avg - self.threshold * self.std:
            self.pick = True
        else:
            self.pick = False

        return
