import numpy as np
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import Matern
from core.data_processor import DataLoader
from core.model import Model

from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from gp import expected_improvement, sample_next_hyperparameter

import os
import json
import time
import math

class BayesOptimizerLSTM(object):

	def __init__(self, configs, pred_model, datahandler, random_search = False, param_range=[1,30], num_iterations = 10, init_params=None, alpha=1e-5, epsilon=1e-7):
		self.configs = configs
		self.pred_model = pred_model
		self.datahandler = datahandler
		self.param_range = np.array(param_range)
		self.num_iterations = num_iterations
		self.random_search = random_search
		self.alpha = alpha
		self.epsilon = epsilon
		if init_params is not None:
			self.xp = np.array(init_params[0])
			self.yp = np.array(init_params[1])
		else:
			self.init_params = None
		self.kernel = Matern()
		self.gp = gp.GaussianProcessRegressor(kernel=self.kernel,
											alpha=alpha,
											n_restarts_optimizer=10,
											normalize_y=True)

	def fit(self):
		# n_params = self.param_range.shape[0]

		for i in range(self.num_iterations):
			self.gp.fit(self.xp.reshape(-1,1), self.yp)


			if self.random_search:
				x_random = np.random.randint(self.param_range[:, 0], self.param_range[:, 1], size=(random_search, n_params))
				ei = -1 * expected_improvement(x_random, self.gp, self.yp, greater_is_better=True, n_params=n_params)
				next_sample = x_random[np.argmax(ei), :]            
			else:
				next_sample = sample_next_hyperparameter(expected_improvement, self.gp, self.yp, greater_is_better=True)


			if np.any(np.abs(next_sample - self.xp) <= self.epsilon):
				next_sample = np.random.randint(self.param_range[0], self.param_range[1], size=1)

			print(f'next sample is {next_sample}')
			data = self.datahandler.make_windows(np.ceil(next_sample))

			x_train, y_train = data.get_train_data(
								seq_len=self.configs['data']['sequence_length'],
								normalise=self.configs['data']['normalise'])

			self.pred_model.train(x_train,y_train,
						epochs = self.configs['training']['epochs'],
						batch_size = self.configs['training']['batch_size'],
						save_dir = self.configs['model']['save_dir'])

			x_test, y_test = data.get_test_data(
								seq_len=self.configs['data']['sequence_length'],
								normalise=self.configs['data']['normalise'])

			cv_score = self.pred_model.model.evaluate(x_test,y_test)
			print(f'cv score {cv_score}')
			self.xp = np.append(self.xp,next_sample)
			self.yp = np.append(self.yp,cv_score)

			print(f'current xp,yp {self.xp, self.yp}')

	def get_optimal_params(self):
		return self.xp[np.argmax(self.yp)]

def main():
	pass



if __name__ == '__main__':
	main()
	configs = json.load(open('config.json', 'r'))
	if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

	data = DataLoader(
		os.path.join('data', configs['data']['filename']),
		configs['data']['train_test_split'],
		configs['data']['columns'])


	model = Model()
	model.build_model(configs)
	x, y = data.get_train_data(
		seq_len=configs['data']['sequence_length'],
		normalise=configs['data']['normalise']
	)


	bolstm = BayesOptimizerLSTM(configs, model, data, init_params=[[1],[.1]])
	bolstm.fit()

	plt.plot(bolstm.xp, bolstm.yp)
	plt.show()

