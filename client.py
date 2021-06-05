import time
import numpy as np

class Client(object):
	def __init__(self, comp_const):
		'''
		Args:
			comp_const: computation constraint of client
		'''
		self.comp_const = comp_const

	def update(self, data, current_weight, deadline_time):
		gradient, run_time = self.compute_gradient(data, current_weight)

		if run_time > deadline_time:
			return 0
		return gradient, run_time

	def _differentiate(self, X, y, W, b):
		h = np.dot(X, W) + b
		dW = np.dot(X.T, (h - y))
		db = np.sum(h-y)
		return dW, db

	def compute_gradient(self, data, current_weight):
		X, y = data
		W, b = current_weight

		if X.shape[0] > self.comp_const:
			dW, db = 0, 0
			
		start_time = time.time()
		
		run_time = time.time() - start_time

		return [dW, db], run_time

