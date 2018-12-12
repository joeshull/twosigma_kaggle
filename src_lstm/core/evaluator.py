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
				predcol='confidenceLevel', 
				rawcol='returnsOpenNextRaw10'):
		self.timecol = timecol
		self.targetcol = targetcol
		self.predcol = predcol
		self.rawcol = rawcol

	def get_kaggle_mean_variance(self, df):
		'''Returns the Mean-variance metric used in the Kaggle competition.

		Input: Dataframe with columns defined in object instantiation

		Output: (float) The model's performance as evaluated by Kaggle
		'''
		df['model_returns'] = df[self.predcol] * df[self.targetcol]
		mean_return = df['model_returns'].mean()
		std_return = df['model_returns'].std()
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

	def _calc_raw(self, df):
		'''
		Hidden Function that calculates the cumulative return of the model.
		'''
		df['model_returns'] = df[self.predcol] * df[self.rawcol]
		df = df.groupby(self.timecol).mean()
		df.reset_index(level='time', inplace=True)
		df.sort_values(self.timecol)
		model_raw = df['model_returns'].cumsum().values

		return model_raw

	def _calc_res(self, df):
		'''
		Hidden Function that calculates the cumulative return of the model.
		'''
		df['model_returns'] = df[self.predcol] * df[self.targetcol]
		df = df.groupby(self.timecol).mean()
		df.reset_index(level='time', inplace=True)
		df.sort_values(self.timecol)
		model_res = df['model_returns'].cumsum().values

		return model_res


	def _calc_market_raw(self, df):
		'''
		Hidden Function that calculates the cumulative return of the market.
		'''
		df = df.groupby(self.timecol).mean()
		df.reset_index(level='time', inplace=True)
		df.sort_values(self.timecol)
		model_raw = df[self.rawcol].cumsum().values

		return model_raw	

	def _calc_market_res(self, df):
		'''
		Hidden Function that calculates the cumulative return of the market.
		'''
		df = df.groupby(self.timecol).mean()
		df.reset_index(level='time', inplace=True)
		df.sort_values(self.timecol)
		model_res = df[self.predcol].cumsum().values

		return model_res

	def _get_date_series(self, df):
		'''
		Hidden function that returns the series of dates for prediction time-period
		'''
		df = df.groupby(self.timecol).mean()
		df.reset_index(level='time', inplace=True)
		df.sort_values(self.timecol)
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
	df_test = pd.read_pickle('../data/test_data.pkl')
	df_test['confidenceLevel'] = np.random.uniform(-1,1, len(df_test))
	df_test = df_test.loc[df_test.time<20181201]

	evaluator = ReturnsEvaluator()

	# print(evaluator.get_kaggle_mean_variance(df_test))

	metrics_dict = evaluator.get_returns(df_test)

	dates = metrics_dict['dates']
	model_returns = metrics_dict['model_res']
	market_returns = metrics_dict['market_res']

	fig, ax = plt.subplots()
	plot_model_vs_market(dates, model_returns, market_returns, ax)
	plt.show()










