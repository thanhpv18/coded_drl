import time
import numpy as np

class Client(object):
	def __init__(self, comp_const):
		'''
		Args:
			comp_const: computation constraint of client
		'''
		self.comp_const = comp_const
		self.dW = 0
		self.db = 0

	def update(self, data, current_weight, deadline_time):
		gradient, run_time = self.compute_gradient(data, current_weight)

		if run_time > deadline_time:
			return 0
		return gradient, run_time

	def _differentiate(self, X, y, W, b):
		h = np.dot(X, W) + b
		self.dW += np.dot(X.T, (h - y))
		self.db += np.sum(h-y)

	def compute_gradient(self, data, current_weight):
		start_time = time.time()

		X, y = data
		W, b = current_weight

		self.dW, self.db = 0, 0
		if X.shape[0] > self.comp_const:
			t = 0
			while t < X.shape[0]:
				batch_X, batch_y = X[t:t + self.comp_const], y[t:t + self.comp_const]
				self._differentiate(batch_X, batch_y, W, b)
				t += self.comp_const
			if X.shape[0] % self.comp_const:
				batch_X, batch_y = X[t:], y[t:]
				self._differentiate(batch_X, batch_y, W, b)

		else:
			self._differentiate(X, y, W, b)

		run_time = time.time() - start_time

		return [self.dW, self.db], run_time

