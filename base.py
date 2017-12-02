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


class SimulationEnv(object):
    
    loggables = ['positions']
    
    def __init__(self, wealth, assets, 
                 path_to_data, start_date, end_date,
                 agent_type, expert_type, reinvest, max_assets=100):

        self.init_wealth = wealth
        self.wealth = wealth
        self.assets = assets
        self.agent_type = agent_type
        self.expert_type = expert_type
        self.path_to_data = path_to_data
        self.reinvest = reinvest
        self.max_assets = max_assets
        self.start_date = start_date
        self.end_date = end_date
        
        return
    
    ## To be called before running a simulation - initialize experts, agents, and initial position, etc.
    def setup_params(self, agent_args={}, expert_args={}):
    
        stdevs = {}
        for f in os.listdir(self.path_to_data):
            df = pd.read_csv(self.path_to_data+f)
            df = df[df["date"] < self.end_date]
            df["return"] = (df["adj_close"] - df["adj_close"].shift(1))/df["adj_close"].shift(1)
            stdevs[f] = df["return"].std()
            self.assets = [f.split(".")[0].upper() for f in sorted(stdevs, key=stdevs.get, reverse=True)[:self.max_assets]]
        
        print self.assets
        
        self.experts = [self.expert_type(a, self.path_to_data, self.start_date, self.end_date, **expert_args) for a in self.assets]
        self.agent = self.agent_type(self.experts, **agent_args)
        self.positions = np.array([weight * self.wealth for weight in self.agent.weights])
        
        return
        
    ## Run simulation
    def run(self, log=False, logpath=os.getcwd()):
        
        ## Start period counter and log
        self.period = 1
        self.finallog = []
        self.log = log
        self.logpath = logpath
        
        ## Warmup period: 
        ## i.e. for strategies involving moving average indicators, wait until we have enough data to calculate MA
        for expert in self.agent.experts:
            expert.warmup()
        
        ## Simulation: Go until data iterators reach the end
        #print "Initial weights:"
        #print np.array([weight for weight in self.agent.weights])
        
        while True:
            try:
#                print "PERIOD {}".format(self.period)
#                print "dates:"
                dates = [e.current_date for e in self.agent.experts]
                print list(set(dates))
                ## NEED TO MAKE THIS TRUE
                ## assert len(set(dates)) == 1
              


#                print "---------------------"
                ## Log this period
                if log:
                    self.logperiod()
                    
                ## Update experts with last period's rewards
                for i, expert in enumerate(self.agent.experts):
                    expert.update()

                ## Update agent accordingly (i.e. for Hedge, update weights according to each expert's reward in the last period)
                self.agent.update()
                           
                ## Rewards accrue to agent
                self.positions = self.positions * (1 + self.agent.returns)
                self.wealth = np.sum(self.positions)
                
                ## Rebalance according to new, updated weights
                ## TODO: only allow non-fractional shares to be purchased (?)
                if np.sum(self.agent.weights) == 1:
                    self.positions = np.array([weight * self.wealth for weight in self.agent.weights])
                
#                 print "weights:"
#                 print np.array([weight for weight in self.agent.weights])

#                 print "rewards:"
#                 print np.array([r for r in self.agent.rewards])
#                 print "returns:"
#                 print np.array([r for r in self.agent.returns])
                
#                 print "positions:"
#                 print np.array([weight * self.wealth for weight in self.agent.weights])
#                 print "positions invested:"
                positions_invested = np.array([e.pick for e in self.agent.experts])*self.positions
#                 print positions_invested
                if self.reinvest and np.any(positions_invested):
                    uninvested_wealth = self.wealth - np.sum(positions_invested)
                    #print "Need to reallocate {} cash amongst experts that are investing".format(uninvested_wealth)
                    #print "Total wealth: {}".format(self.wealth)
                    #print "Reinvestments:"
                    #print uninvested_wealth * positions_invested/np.sum(positions_invested)
                    #print "Final position"
                    self.positions = uninvested_wealth * positions_invested/np.sum(positions_invested) + positions_invested
                    #print self.positions
                    new_wealth = np.sum(self.positions)
                    assert(new_wealth-self.wealth <= 0.00001)
                
                
                
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
        # Set up log file structure
        print("saving log")
        runtime = datetime.datetime.now()
        runuser = getpass.getuser()
        logname = runuser + "_" + runtime.strftime('%Y-%m-%d_%H-%M-%S')
        if not os.path.exists(os.path.join(self.logpath, logname)):
            os.makedirs(os.path.join(self.logpath, logname))
            os.makedirs(os.path.join(self.logpath, logname, 'plots'))
        
        # Create dataframe from log
        col = ['period', 'wealth'] + \
            [x+'.'+y for x in self.loggables for y in self.assets] + \
            [x+'.'+y for x in self.agent_type.loggables for y in self.assets] + \
            [x+'.'+y for x in self.expert_type.loggables for y in self.assets]
        df = pd.DataFrame(self.finallog, columns=col)
        df.set_index('period', inplace=True)
        
        # Write xlsx of log data
        writer = pd.ExcelWriter(os.path.join(self.logpath, logname, logname+'.xlsx'), engine='xlsxwriter')
        pd.io.formats.excel.header_style = None
        df.to_excel(writer,'run_log')
        
        workbook  = writer.book
        worksheet = writer.sheets['run_log']
        
        header_format = workbook.add_format({'bold': True,'text_wrap': True})
        worksheet.set_row(0, None, header_format)
        writer.save()
        
        # Set up matplotlib. Loop through loggables and save a plot with and without legend for each.
        rc('font', family = 'serif', serif = 'cmr10')
        plots = ['wealth'] + self.loggables + self.agent_type.loggables + self.expert_type.loggables
        
        for p in plots:
            for legend in [True, False]:
                plotdf = df.filter(regex='period|'+p)
                plotlab = [l.split(p+'.',1)[1] if l != 'wealth' else l.title() for l in list(plotdf.columns.values)]
                plt.plot(plotdf)
                plt.ylabel(p.title())
                plt.xlabel('Round')
                plt.ticklabel_format(useOffset=False)
                if legend:
                    plt.legend(plotlab, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=4)
                    plt.savefig(os.path.join(self.logpath, logname, 'plots', p+'_legend.png'), bbox_inches='tight', dpi=300)
                else:
                    plt.savefig(os.path.join(self.logpath, logname, 'plots', p+'.png'), bbox_inches='tight', dpi=300)
                plt.close()


def GridSearch(Agent, Expert, stocks, reinvest=False, full_data=False, agent_args=[], expert_args=[]):
    ## Run simulation 
    start = time.time()
    initial_wealth = 100000
    if full_data:
        data = "data/djia_20000101_20170831/"
        start_date = "2000-01-01"
        end_date = "2017-08-31"
        years = 17. + 9./12
    else:
        data = "data/djia_20150101_20171101/"
        start_date = "2015-01-01"
        end_date = "2017-11-01"
        years = 2. + 11./12
    best_wealth = 0
    best_params = {}
    for agent_arguments in agent_args:
        for expert_arguments in expert_args:
            
            s = SimulationEnv(
                initial_wealth, 
                stocks, 
                data, 
                start_date,
                "2017-11-01", 
                Agent, 
                Expert, 
                reinvest
            )
            s.setup_params(
                agent_args=agent_arguments,
                expert_args=expert_arguments
            )            s.run(log=True)            s.run(log=True)
            s.run(log=True, logpath="logs")
            ar = ((s.wealth)/initial_wealth)**(1/years) - 1
            end = time.time()

            if s.wealth > best_wealth:
                best_wealth = s.wealth
                best_params = {
                    "agent_args": agent_arguments,
                    "expert_args": expert_arguments,
                    "Initial wealth": initial_wealth,
                    "Final wealth": s.wealth,
                    "Annualized return": ar,
                    "Time": int(end-start),
                }
    print("Best params:", best_params)
 
