import datetime as dt
import pandas as pd
import numpy as np
import util as ut

import datetime as dt
import QLearner as ql
import pandas as pd
import numpy as np
import util as ut

class StrategyLearner(object):

    qty = 0
    discretization_intervals = {}
    indicators = ['MFI','WilliamR','StDev'] #'RSI','ATR','ADX'

    # constructor this method should create a QLearner
    def __init__(self, sym, sv, verbose = False):
        self.sv = sv
        self.sym = sym
        self.data = ut.get_data(sym)
        self.create_discretization_intervals()
        self.qty = sv/self.data['Close'][-1]
        self.total_states = 3*(10**len(self.indicators))
        self.verbose = verbose

        self.learner = ql.QLearner(num_states=self.total_states, \
                                   num_actions=3, \
                                   rar=0.98, \
                                   radr=0.9999, \
                                   dyna=0, \
                                   verbose=self.verbose)  # initialize the learner

    # train qlearner for trading
    def addEvidence(self, symbol, dates):
        iterations = 50     # set the numberof iterations to go through the training set
        trainX, daily_returns = self.get_features_data(dates)
        states = self.discretize(trainX)
        trading_dates = trainX.index    # do not use states index as dropna has been applied on it
        orders = self.perform_qlearning(states['States'], symbol, self.sv, trading_dates, daily_returns, iterations, process = 'TRAIN')
        return orders

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol, dates):
        sym=[symbol]
        testX, daily_returns = self.get_features_data(dates)
        states = self.discretize(testX)
        trading_dates = testX.index
        orders = self.perform_qlearning(states['States'], symbol, self.sv, trading_dates, daily_returns, iterations = 1, process = 'TEST')
        return orders

    # States 0-999 No Holding
    # States 1000-1999 LONG
    # States 2000-2999 SHORT
    # Action 0 Do Nothing
    # Action 1 BUY
    # Action 2 SELL
    def perform_qlearning(self, states, sym, sv, dates, dr,  iterations = 1, process = 'TEST'):
        print process
        lastIter = False
        trades = pd.DataFrame()
        for iter in range(0,iterations):
            if iter == iterations-1:
                lastIter = True
            initial_state = states[0]
            holding = 0
            holding_state_factor = (10**len(self.indicators))
            action = self.learner.querysetstate(holding_state_factor*holding + initial_state) #set the state and get first action
            orders = pd.DataFrame(columns=['Date','Symbol','Order','Shares'])
            trades = pd.DataFrame(data= np.zeros(len(dates)), columns=['Trades'], index = dates)

            for date, current_state in states[1:].iteritems():

                if process == 'TRAIN':
                    reward = 0
                    if holding == 1:
                        reward = dr.ix[date]*100
                    elif holding == 2:
                        reward = dr.ix[date]*-100
                    action = self.learner.query(holding_state_factor*holding + current_state, reward)
                elif process == 'TEST':
                    action = self.learner.querysetstate(holding_state_factor*holding + current_state)

                if action == 1 and holding != 1:    #BUY
                    if holding == 0:    #Long Entry
                        holding = 1
                        orders = orders.append({'Date': date, 'Symbol': sym, 'Order': 'BUY', 'Shares': self.qty},
                                               ignore_index=True)
                    elif holding == 2:  #SHORT Exit
                        holding = 0
                        orders = orders.append({'Date': date, 'Symbol': sym, 'Order': 'COVER', 'Shares': self.qty},
                                               ignore_index=True)
                    trades.ix[date] = self.qty

                if action == 2 and holding != 2:     #SELL
                    if holding == 0:    #Short Entry
                        holding = 2
                        orders = orders.append({'Date': date, 'Symbol': sym, 'Order': 'SHORT', 'Shares': self.qty},
                                               ignore_index=True)
                    elif holding == 1:  #LONG Exit
                        holding = 0
                        orders = orders.append({'Date': date, 'Symbol': sym, 'Order': 'SELL', 'Shares': self.qty},
                                               ignore_index=True)
                    trades.ix[date] = -self.qty

            profit = self.compute_portvals(orders)
            print str(iter+1) + '. final portfolio value =',sv + profit

            # if process == 'TEST':
            #     print orders

        return orders

    def get_features_data(self, dates):
        # data = data/data.ix[0,:]
        period_data = self.data.loc[dates]
        dataX = period_data[self.indicators]
        daily_returns = period_data['ROC']
        # print daily_returns
        # print dataX
        # print prices.shape
        return dataX, daily_returns

    def create_discretization_intervals(self):
        for indicator in self.indicators:
            indicator_data = self.data[indicator]
            self.discretization_intervals[indicator] = np.percentile(indicator_data, range(10,100,10))

    def discretize(self, data):
        states = data*0
        s = np.zeros(states.shape[0], dtype=int)
        for indicator in self.indicators:
            states[indicator] = np.digitize(data[indicator], self.discretization_intervals[indicator])
            s = s*10 + states[indicator]
        s.astype(np.int64)
        states['States'] = s
        # freq = np.unique(s.values, return_counts=True)
        # j = 0
        # for i in range(0,self.total_states/3):
        #     if freq[0][j] > i:
        #         print i, 0
        #     if freq[0][j] == i:
        #         print freq[0][j], freq[1][j]
        #         j+=1
        return states

    def compute_portvals(self, df_orders):
        prices = self.data['Close']
        df_orders = df_orders.set_index('Date')
        df_orders_sell = df_orders.where((df_orders['Order'] == 'SELL') | (df_orders['Order'] == 'SHORT'))
        df_orders_sell['Shares'] = df_orders_sell['Shares']*-1
        df_orders.update(df_orders_sell)    #Sold shares will be represented as negative

        if df_orders['Shares'].sum(axis=0) != 0:
            df_orders = df_orders[:-1]

        df_account = pd.DataFrame(index=df_orders.index)
        df_account['Cash']= -1 * df_orders['Shares']* prices
        # df_account['Shares'] = df_orders['Shares']

        profit = df_account['Cash'].sum()

        return profit


if __name__=="__main__":
    sym = "COPPERM-I"
    # instantiate the strategy learner
    verb = False
    learner = StrategyLearner(sym=sym, sv=10000, verbose=verb)

    # set parameters for traning
    stdate = dt.datetime(2018, 11, 1)
    enddate = dt.datetime(2019, 3, 31) + dt.timedelta(days=1)
    training_dates = [i for i in learner.data.index if i > stdate and i < enddate]

    # train the learner
    train_orders = learner.addEvidence(symbol=sym, dates=training_dates)

    # set parameters for testing
    stdate = dt.datetime(2019, 4, 1)
    enddate = dt.datetime(2019, 4, 30) + dt.timedelta(days=1)
    testing_dates = [i for i in learner.data.index if i > stdate and i < enddate]

    # test the learner
    test_orders = learner.testPolicy(symbol=sym, dates=testing_dates)

    orders = train_orders.append(test_orders)
    orders.to_csv(ut.get_path(sym + '_Trades'), index=False)
