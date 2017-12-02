from agents import *
from experts import * 
from base import *

stocks = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE',
         'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 
         'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'VZ', 'WMT', 'XOM']

stocks = [stock.lower() for stock in stocks]

agent_arglist = [
            {"eta": 0.01},
            {"eta": 0.02}, 
            {"eta": 0.05},
            {"eta": 0.1},
            {"eta": 0.2},
            {"eta": 0.5}
        ]

expert_arglist = [
            {"window_size": 10, "threshold": 0.25},
            {"window_size": 10, "threshold": 0.5},
            {"window_size": 10, "threshold": 1.0},
            {"window_size": 10, "threshold": 1.5},
            {"window_size": 10, "threshold": 2.0}, 
            {"window_size": 5, "threshold": 0.25},
            {"window_size": 5, "threshold": 0.5},
            {"window_size": 5, "threshold": 1.0},
            {"window_size": 5, "threshold": 1.5},
            {"window_size": 5, "threshold": 2.0}
       ]

GridSearch(EG, Momentum, stocks, reinvest=True, full_data=True, agent_args=agent_arglist, expert_args=expert_arglist, pickle_file="eg_momentum_reinvest_all")


