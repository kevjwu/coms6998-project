from agents import *
from experts import * 
from base import *

stocks = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE',
         'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 
         'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'VZ', 'WMT', 'XOM']

stocks = [stock.lower() for stock in stocks]

GridSearch(EG, Momentum, stocks, reinvest=True, full_data=False, agent_args=[{"eta":0.5}], expert_args=[{"window_size": 10, "threshold": 0.5}])


