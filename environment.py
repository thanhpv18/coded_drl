import random
import numpy as np 

from client import Client

# define all set of states
# define reward
# define client

class ClientSystem(object):
	def __init__(self, config):
		self.config = config
		self.create_system()

	def create_system(self):
		self.clients = []
		computation_constraint = config["computation_constraint"]
		n = len(computation_constraint)
		for i in range(self.config["n_client"]):
			cons_ = computation_constraint[np.random.randint(n)]
			single_client = Client(cons_)
			self.clients.append(single_client)

class LinearRegression(object):
	def __init__(self, w_size, lr = 0.001):
		self.W = np.random.randn((w_size, 1))
		self.b = 0
		self.lr = lr
		self.n_update = 0

	def compute_loss(self, data):
		X, y = data
		h = np.dot(X.T, self.W) + self.b

		return np.sum(np.power(y - h), 2)

	def update(self, dW, db, m):
		self.n_update += 1
		self.W = self.W - self.lr * dW/m
		self.b = self.b - self.lr * db/m

	def get_weight(self):
		return (self.W, self.b) 


class DistributedSystem(object):
	def __init__(self, config, data):
		self.config = config
		self.system = ClientSystem(config)
		self.X, self.y = data
		self.m, self.d = X.shape
		self.model = LinearRegression(self.d)

	def create_random_matrix(self, c = None):
		if c is None
			c = np.random.randint(m, m + max(10, m//10))
		self.G = np.random.normal(size = (c, self.m))

	def reset(self):
		state = self.get_state()
		self.create_random_matrix()
		return state 

	def get_state(self):

	def step(self):
		self.coded_X = np.dot(self.G, self.X.T)
		self.coded_y = np.dot(self.G, self.y.T)

		total_gradient = []
		for i in range(config["n_client"]):
			st, ed, deadline = self.assign_info[i]
			X = self.coded_X[st:ed]
			y = self.coded_y[st:ed]
			single_client = self.system.clients[i]

			gradient, run_time = single_client.update([X, y], self.model.get_weight(), deadline)
			if run_time > deadline:
				gradient = 0
			else:
				total_gradient.append(gradient)

		self.update_model(total_gradient)

		return reward

	def update_model(self, gradient):
		grad_W, grad_b = 0, 0
		total_data_used = 0
		for i in range(self.config['n_client']):
			st, ed = self.assign_info[i][:2]
			grad_W += gradient[i][0] * (ed - st)
			grad_b += gradient[i][1] * (ed - st)
			total_data_used += ed - st 

		if total_data_used != 0:
			self.model.update(grad_W, grad_b, total_data_used)


