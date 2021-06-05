import random
import numpy as np 

from client import Client

# define all set of states
# define reward
# define client

class ClientSystem(object):
	def __init__(self, n_client, config):
		self.n_client = n_client
		self.config = config
		self.create_system()

	def create_system(self):
		self.clients = []
		computation_constraint = config["computation_constraint"]
		n = len(computation_constraint)
		for i in range(self.n_client):
			cons_ = computation_constraint[np.random.randint(n)]
			single_client = Client(cons_)
			self.clients.append(single_client)

class DistributedSystem(object):
	def __init__(self):

	def reset(self):
		state = self.get_state()
		return state 

	def get_state(self):

	def step(self):
		return reward