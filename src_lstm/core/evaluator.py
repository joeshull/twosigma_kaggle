import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.dates import date2num


class ReturnsEvaluator():
	'''A stock prediction evaluator.
	Requires a dataframe with returns and predictions in the form of the
	Kaggle/Two-Sigma Competition circa fall 2018.
	'''

	def __init__(self, 
				timecol = 'time',
				targetcol = 'returnsOpenNextMktres10',
				universecol = 'universe', 
				predcol='confidenceValue', 
				rawcol='returnsOpenNextRaw10'):
		self.timecol = timecol
		self.targetcol = targetcol
		self.universecol = universecol
		self.predcol = predcol
		self.rawcol = rawcol

	def get_kaggle_mean_variance(self, df, model=True, universe=False):
		'''Returns the Mean-variance metric used in the Kaggle competition.
 
		Input: Dataframe with columns defined in object instantiation

		Output: (float) The model's performance as evaluated by Kaggle
		'''
		if universe:
			if model:
				df = self._create_model_returns(df, self.targetcol)
				daily_returns = df.groupby(self.timecol).model_returns.mean()
			else:
				df = self._create_market_returns(df, self.targetcol)
				daily_returns = df.groupby(self.timecol).market_returns.mean()
		else:
			if model:
				df['model_returns'] = df[self.predcol] * df[returnscol]
				daily_returns = df.groupby(self.timecol).model_returns.mean()
			else:
				df['market_returns'] = df[returnscol]
				daily_returns = df.groupby(self.timecol).market_returns.mean()
		mean_return = daily_returns.mean()
		std_return = daily_returns.std()
		return mean_return/std_return



	def get_returns(self,df):
		'''
		Input: Dataframe with columns defined in object instantiation

		Output: (dict)
		1. 'model_raw' - (array) Cumulative Raw Return over the evaluation period
		2. 'model_res' - (array) Cumulative Market Residualized return over the evaluation period
		3. 'market_raw' - (array) The avg daily residualized return of the entire market
		4. 'market_res' - (array) The avg daily residualized return of the entire market
		5. 'dates' - (pd.Series) The series of dates for prediction time period
		'''

		model_raw = self._calc_raw(df)
		model_res = self._calc_res(df)
		market_raw = self._calc_market_raw(df)
		market_res = self._calc_market_res(df)
		dates = self._get_date_series(df)


		return {'model_raw' : model_raw, 'model_res' : model_res,
				'market_raw' :  market_raw, 'market_res' : market_res,
				'dates' : dates}


	def _create_model_returns(self, df, returnscol):
		df['model_returns'] = df[self.predcol] * df[returnscol] * df[self.universecol]
		return df

	def _create_market_returns(self, df, returnscol):
		df['market_returns'] = df[returnscol] * df[self.universecol]
		return df


	def groupby_time(self, df):
		df = df.groupby(self.timecol).mean()
		df.reset_index(level=self.timecol, inplace=True)
		df.sort_values(self.timecol, inplace=True)
		return df


	def get_cumulative_return(self, df, returnscol):
		model_returns = df[returnscol].values
		invest = np.ones(len(model_returns))
		principal_return = np.zeros((len(model_returns)))
		raw_returns = np.zeros(len(model_returns))

		for i in range(len(model_returns)):
			if i-11 < 0:
				raw_returns[i] = model_returns[i]
				continue

			invest[i] = invest[i-11] + ((invest[i-11] - principal_return[i-11]) * model_returns[i-11])
			raw_returns[i] = invest[i] * model_returns[i]
			principal_return[i] = invest[i-11]

		portfolio_return = raw_returns/11
		portfolio_return[:11] = 0 

		return portfolio_return.cumsum()



	def _calc_raw(self, df):
		'''
		Hidden Function that calculates the cumulative return of the model.
		'''
		df = self._create_model_returns(df, self.rawcol)
		df = self.groupby_time(df)
		return self.get_cumulative_return(df, 'model_returns')

	def _calc_res(self, df):
		'''
		Hidden Function that calculates the cumulative return of the model.
		'''
		df = self._create_model_returns(df, self.targetcol)
		df = self.groupby_time(df)
		return self.get_cumulative_return(df, 'model_returns')


	def _calc_market_raw(self, df):
		'''
		Hidden Function that calculates the cumulative return of the market.
		'''
		df = self._create_market_returns(df, self.rawcol)
		df = self.groupby_time(df)
		return self.get_cumulative_return(df, 'market_returns')	

	def _calc_market_res(self, df):
		'''
		Hidden Function that calculates the cumulative return of the market.
		'''
		df = self._create_market_returns(df, self.targetcol)
		df = self.groupby_time(df)
		return self.get_cumulative_return(df, 'market_returns')	

	def _get_date_series(self, df):
		'''
		Hidden function that returns the series of dates for prediction time-period
		'''
		df = self.groupby_time(df)
		df['DateTime'] = pd.to_datetime(df[self.timecol].astype(str), format='%Y%m%d')

		return df['DateTime']


def plot_model_vs_market(dates, model_returns, market_returns, ax, title='Model vs Market'):
	X = date2num(dates.values)

	ax.plot_date(X, model_returns, 
				linestyle='-',
				linewidth=2,
				markersize=.1, 
				label=f'Model Returns : {round(model_returns[-1],2)}')	
	ax.plot_date(X, market_returns, 
				linestyle='-',
				linewidth=2,
				markersize=.2,
				label=f'Market Returns : {round(market_returns[-1],2)}')
	for tick in ax.get_xticklabels():
		tick.set_rotation(90)

	ax.set_xlabel('Dates')
	ax.set_ylabel('Cumulative Return')
	ax.set_title(title)	
	ax.legend()




if __name__ == '__main__':
	#test the class
	np.random.seed(31)
	df_test = pd.read_pickle('../data/5dayapple_pred.pkl')
	df_test = df_test.loc[(df_test.time>=20170101) & (df_test.time<=20181101)]
	test_data = {'time': np.arange(60),
			'confidenceValue' : np.ones(60),
			'universe' : np.ones(60),
			'returnsOpenNextRaw10' : (np.ones(60)*.03),
			'returnsOpenNextMktres10' : (np.ones(60)*.03)
			}

	test = pd.DataFrame.from_dict(test_data, orient='columns')


	evaluator = ReturnsEvaluator()

	print(evaluator.get_kaggle_mean_variance(test))


	portfolio_return = evaluator._calc_raw(test)


	metrics_dict = evaluator.get_returns(df_test)
	dates = metrics_dict['dates']
	model_returns = metrics_dict['model_raw']
	market_returns = metrics_dict['market_raw']

	fig, ax = plt.subplots()
	plot_model_vs_market(dates, model_returns, market_returns, ax)
	plt.show()











