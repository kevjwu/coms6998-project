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
import pickle

class SimulationEnv(object):
    
    loggables = ['positions']
    
    def __init__(self, wealth, assets, 
                 path_to_data, start_date, end_date,
                 agent_type, expert_type, reinvest, max_assets=30):

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
    
    # To be called before running a simulation - initialize experts, agents, and initial position, etc.
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
        
    # Run simulation
    def run(self, log=False, plot=False, logpath=os.getcwd()):
        
        self.log = log
        self.plot = plot
        self.logpath = logpath
        keeplog = False
        self.period_return = None
        
        # Start period counter and timer
        self.period = 1
        self.starttime = datetime.datetime.now()
        
        # Keep a log if either logging or plotting is true
        if log == True or plot == True:
            keeplog = True
            self.finallog = []
            self.runtime = datetime.datetime.now()
            self.runuser = getpass.getuser()
            self.logname = self.runuser + "_" + self.runtime.strftime('%Y-%m-%d_%H-%M-%S')
        
        # Warmup period: 
        # i.e. for strategies involving moving average indicators, wait until we have enough data to calculate MA
        for expert in self.agent.experts:
            expert.warmup()
        
        # Simulation: Go until data iterators reach the end
        while True:
            try:
                dates = [e.current_date for e in self.agent.experts]
                # NEED TO MAKE THIS TRUE
                # assert len(set(dates)) == 1
                
                prior_wealth = self.wealth

                # Log this period
                if keeplog:
                    self.logperiod()
                    
                # Update experts with last period's rewards
                for i, expert in enumerate(self.agent.experts):
                    expert.update()

                # Update agent accordingly (i.e. for Hedge, update weights according to each expert's reward in the last period)
                self.agent.update()
                           
                # Rewards accrue to agent
                self.positions = self.positions * (1 + self.agent.returns)
                self.wealth = np.sum(self.positions)
                
                # Rebalance according to new, updated weights
                # TODO: only allow non-fractional shares to be purchased (?)
                if np.sum(self.agent.weights) == 1:
                    self.positions = np.array([weight * self.wealth for weight in self.agent.weights])
                
                positions_invested = np.array([e.pick for e in self.agent.experts])*self.positions
                
                if self.reinvest and np.any(positions_invested):
                    uninvested_wealth = self.wealth - np.sum(positions_invested)
                    self.positions = uninvested_wealth * positions_invested/np.sum(positions_invested) + positions_invested
                    new_wealth = np.sum(self.positions)
                    assert(new_wealth-self.wealth <= 0.00001)
                    
                # Calculate return for period
                self.period_return = (self.wealth - prior_wealth)/prior_wealth
                
                # Advance period
                self.period += 1
                
            except StopIteration:
                break
        
        self.runduration = str((datetime.datetime.now()-self.starttime).seconds) + ' seconds'
        
        # If logging, convert log to DF
        if keeplog:
            cols = ['period', 'wealth', 'period_return'] + \
                [x+'.'+y for x in self.loggables for y in self.assets] + \
                [x+'.'+y for x in self.agent_type.loggables for y in self.assets] + \
                [x+'.'+y for x in self.expert_type.loggables for y in self.assets]
            self.logdf = pd.DataFrame(self.finallog, columns=cols)
            self.logdf.set_index('period', inplace=True)
        
        # Write the log file if logging
        if log:
            self.savelog()
            
        # Write the plots if plotting
        if plot:
            self.saveplots()

    def logperiod(self):
        row = [self.period] + [self.wealth] + [self.period_return]
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
        if not os.path.exists(os.path.join(self.logpath, self.logname)):
            os.makedirs(os.path.join(self.logpath, self.logname))
        
        # Write simulation metadata
        writer = pd.ExcelWriter(os.path.join(self.logpath, self.logname, self.logname+'.xlsx'), engine='xlsxwriter')
        pd.io.formats.excel.header_style = None
        
        sim_meta = self.__dict__.copy()
        nologkeys = ['agent','experts','finallog','logdf','positions','runtime','runuser','period_return']
        for k in nologkeys:
            sim_meta.pop(k, None)
        sim_meta['agent_type'] = self.agent_type.__name__
        sim_meta['expert_type'] = self.expert_type.__name__
        sim_meta['runuser'] = self.runuser
        sim_meta['rundate'] = self.runtime.strftime('%Y-%m-%d')
        sim_meta['runtime'] = self.runtime.strftime('%H:%M:%S')
        sim_meta['annual_return'] = ((self.wealth)/self.init_wealth)**(252./self.period) - 1
        sim_meta['sharpe'] = (sim_meta.get('annual_return')-0.01)/((252**(1/2.0))*np.std(self.logdf['period_return']))
        
        sim_metadf = pd.DataFrame.from_dict(sim_meta, orient='index')
        sim_metadf.columns = ['value']
        sim_metadf.index.names = ['attrib']
        sim_metadf.sort_index(inplace=True)
        sim_metadf.to_excel(writer,'env')
        
        workbook  = writer.book
        worksheet = writer.sheets['env']
        header_format = workbook.add_format({'bold': True,'text_wrap': True})
        left = workbook.add_format({'align': 'left'})
        worksheet.set_column('B:B', None, left)
        worksheet.set_row(0, None, header_format)
        
        # Write xlsx of log data
        self.logdf.to_excel(writer,'run_log')
        
        worksheet = writer.sheets['run_log']
        worksheet.set_row(0, None, header_format)
        
        writer.save()
        
    def saveplots(self):
        # Set up plot file structure
        print("saving plots")
        if not os.path.exists(os.path.join(self.logpath, self.logname, 'plots')):
            os.makedirs(os.path.join(self.logpath, self.logname, 'plots'))
            
        # Set up matplotlib. Loop through loggables and save a plot with and without legend for each.
        rc('font', family = 'serif', serif = 'cmr10')
        plots = ['wealth', 'period_return'] + self.loggables + self.agent_type.loggables + self.expert_type.loggables
        
        for p in plots:
            for legend in [True, False]:
                plotdf = self.logdf.filter(regex='^period$|'+p)
                plotlab = [l.split(p+'.',1)[1] if l not in ['wealth', 'period_return'] else l.title() for l in list(plotdf.columns.values)]
                plt.plot(plotdf)
                plt.ylabel(p.title())
                plt.xlabel('Round')
                plt.ticklabel_format(useOffset=False)
                if legend:
                    plt.legend(plotlab, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=4)
                    plt.savefig(os.path.join(self.logpath, self.logname, 'plots', p+'_legend.png'), bbox_inches='tight', dpi=300)
                else:
                    plt.savefig(os.path.join(self.logpath, self.logname, 'plots', p+'.png'), bbox_inches='tight', dpi=300)
                plt.close()


def GridSearch(Agent, Expert, stocks, max_assets = 30, reinvest=False, full_data=False, agent_args=[], expert_args=[], pickle_file=None):
    ## Run simulation 
    start = time.time()
    initial_wealth = 1.
    if full_data:
        data = "data/djia_20000101_20170831/"
        start_date = "2000-01-01"
        end_date = "2017-08-31"
    else:
        data = "data/djia_20150101_20171101/"
        start_date = "2015-01-01"
        end_date = "2017-11-01"
    
    all_params = []
    best_wealth = 0
    best_params = {}
    for agent_arguments in agent_args:
        for expert_arguments in expert_args:
            s = SimulationEnv(
                initial_wealth, 
                stocks, 
                data, 
                start_date,
                end_date, 
                Agent, 
                Expert, 
                reinvest,
                max_assets
            )
            s.setup_params(
                agent_arguments,
                expert_arguments
            )           
            #s.run(log=True, logpath="logs")
            s.run()
            ar = ((s.wealth)/initial_wealth)**(252./s.period) - 1
            end = time.time()

                           
            params = {
                    "agent_args": agent_arguments,
                    "expert_args": expert_arguments,
                    "Initial wealth": initial_wealth,
                    "Final wealth": s.wealth,
                    "Annualized return": ar
                }
            print params
            print "\n\n\n"
            all_params.append(params)

            if s.wealth > best_wealth:
                best_wealth = s.wealth
                best_params = params

    print("Best params:", best_params)
    if pickle_file:
        print("Pickling results....")
        pickle.dump(all_params, open(pickle_file, "wb" ) )
