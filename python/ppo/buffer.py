import numpy as np



class BufferException(Exception):
    """
    Related to errors with the Buffer.
    """
    pass

class Buffer(object):
	class AgentBuffer(object):
		class AgentBufferField(object):
			def __init__(self):
				self._list = []
			def __len__(self):
				return len(self._list)
			def __str__(self):
				return str(np.array(self._list).shape)
			def __getitem__(self, index):
				return self._list[index]
			def reorder(self, order):
				"""
		        reorder will rearange the list of np.arrays according to order.
		        :param order: The order of the permutation.
		        """
				self._list = [self._list[i] for i in order]
				#TODO : Find better names for the methods
			def append_element(self, data):
				"""
				Ads an element to the end of the list of np.arrays.
				:param data: The np.array or value to append.
				"""
				#TODO: need to handle the case the data is not the right size
				self._list += [np.array(data)]
			def append_list(self, data):
				"""
				Ads a list of np.arrays to the end of the list of np.arrays.
				:param data: The np.array list to append.
				"""
				#TODO: need to handle the case the data is not the right size
				self._list += list(np.array(data))
			def set(self, data):
				"""
				Sets the list of np.array to the input data
				:param data: The np.array list to be set.
				"""
				del self._list
				self._list = list(np.array(data))
				#TODO: find a better way to set the np.array list
			def get_batch(self, batch_size = None, training_length = None):
				"""
				Retrieve the last batch_size elements of length training_length
				from the list of np.array
				:param batch_size: The number of elements to retrieve. If None: 
				All elements will be retrieved.
				:param training_length: The length of the sequence to be retrieved. If
				None: only takes one element.
				"""
				#TODO: Decide what to do if there enough points to retrieve 
				if training_length is None:
					if batch_size is None:
						#return all of them
						return np.array(self._list)
					else:
						# return the batch_size last elements
						if batch_size > len(self._list):
							raise BufferException("Batch size requested is too large")
						return np.array(self._list[-batch_size:])
				else:
					if batch_size is None:
						# retrieve the maximum number of elements
						batch_size = len(self._list) - training_length + 1
					if (len(self._list) - training_length + 1) < batch_size :
						raise BufferException("The batch size and training length requested for get_batch where" 
							" too large given the current number of data points.")
						return 
					tmp_list = []
					for end in range(len(self._list)-batch_size+1, len(self._list)+1):
						tmp_list += [np.array(self._list[end-training_length:end])]
					return np.array(tmp_list)
					# TODO: decide if we need to reset the local buffer now ?	
			def reset_field(self):
				"""
				Resets the AgentBufferField
				"""
				del self._list
				self._list = []
		def __init__(self):
			self._data = {}
		def __str__(self):
			return ", ".join(["'{0}' : {1}".format(k, str(self._data[k])) for k in self._data.keys()])
		def reset_agent(self):
			"""
			Resets the AgentBuffer
			"""
			#TODO: There might be some garbage collection issues ?
			field_list = self._data.keys()
			for k in field_list:
				del self._data[k]
			del self._data
			self._data = {}
		def __len__(self):
			return len(self._data)
		def __getitem__(self, key):
			if key not in self._data:
				self._data[key] = self.AgentBufferField()
			return self._data[key]
		def __setitem__(self, key, value):
			self._data[key] = value    
		def keys(self):
			return self._data.keys()
		def __contains__(self, key):
			return key in self.keys()
		def __iter__(self):
			for key in self.keys():
				yield key
		def check_length(self, key_list):
			"""
			Some methods will require that some fields have the same length.
			check_length will return true if the fields in key_list 
			have the same length.
			:param key_list: The fields which length will be compared
			"""
			if len(key_list) < 2:
				return True
			l = None
			for key in key_list:
				if key not in self._data:
					return False
				if ((l != None) and (l!=len(self._data[key]))):
					return False
				l = len(self._data[key])
			return True
		def shuffle(self, key_list = None):
			"""
			Shuffles the fields in key_list in a consistent way: The reordering will
			be the same accross fields.
			:param key_list: The fields that must be shuffled.
			"""
			if key_list is None:
				key_list = list(self.keys())
			if not self.check_length(key_list):
				raise BufferException("Unable to shuffle if the fields are not of same length")
				return
			s = np.arange(len(self._data[key_list[0]]))
			np.random.shuffle(s)
			for key in key_list:
				self._data[key].reorder(s)
	def __init__(self):
		self.global_buffer = self.AgentBuffer() 
		#Should we have a global buffer? what if the system is distributed ?
		self.local_buffers = {}
	def __str__(self):
		return "global buffer :\n\t{0}\nlocal_buffers :\n{1}".format(str(self.global_buffer), 
			'\n'.join(['\tagent {0} :{1}'.format(k, str(self.local_buffers[k])) for k in self.local_buffers.keys()]))
	def __getitem__(self, key):
		if key not in self.local_buffers:
			self.local_buffers[key] = self.AgentBuffer()
		return self.local_buffers[key]
	def __setitem__(self, key, value):
		self.local_buffers[key] = value
	def keys(self):
		return self.local_buffers.keys()
	def __contains__(self, key):
		return key in self.keys()
	def __iter__(self):
		for key in self.keys():
			yield key
	def append_BrainInfo(self, info):
		"""
		Appends the information in a BrainInfo element to the buffer.
		:param info: The BrainInfo from which the information will be extracted.
		"""
		raise BufferException("This method is not yet implemented")
		# TODO: Find how useful this would be
		# TODO: Implementation
	def reset_global(self):
		"""
		Resets the global buffer
		"""
		del self.global_buffer
		self.global_buffer = self.AgentBuffer() #Is it efficient for garbage collection ?
	def reset_all(self):
		"""
		Resets the global buffer and all the local local_buffers
		"""
		# TODO: Need a better name
		del self.global_buffer
		self.global_buffer = self.AgentBuffer() 
		agent_ids = list(self.keys())
		for k in agent_ids:
			del self.local_buffers[k]
		del self.local_buffers
		self.local_buffers = {}
		# gc.collect()
	def append_global(self, agent_id ,key_list = None,  batch_size = None, training_length = None):
		"""
		Appends the buffer of an agent to the global buffer.
		:param agent_id: The id of the agent which data will be appended
		:param key_list: The fields that must be added. If None: all fields will be appended.
		:param batch_size: The number of elements that must be appended. If None: All of them will be.
		:param training_length: The length of the samples that must be appended. If None: only takes one element.
		"""
		if key_list is None:
			key_list = self.local_buffers[agent_id].keys()
		if not self.local_buffers[agent_id].check_length(key_list):
			raise BufferException("The length of the fields {0} for agent {1} where not of comparable length"
				.format(key_list, agent_id))
		for field_key in key_list:
			self.global_buffer[field_key].append_list(
				self.local_buffers[agent_id][field_key].get_batch(batch_size =batch_size, training_length =training_length)
			)
	def append_all_agent_batch_to_global(self, key_list = None,  batch_size = None, training_length = None):
		"""
		Appends the buffer of all agents to the global buffer.
		:param key_list: The fields that must be added. If None: all fields will be appended.
		:param batch_size: The number of elements that must be appended. If None: All of them will be.
		:param training_length: The length of the samples that must be appended. If None: only takes one element.
		"""
		#TODO: Maybe no batch_size, only training length and a flag corresponding to "get only last of training_length"
		for agent_id in self.local_buffers.keys():
			self.append_global(agent_id ,key_list,  batch_size, training_length)


#TODO: Put these functions into a utils class
def discount_rewards(r, gamma=0.99, value_next=0.0):
    """
    Computes discounted sum of future rewards for use in updating value estimate.
    :param r: List of rewards.
    :param gamma: Discount factor.
    :param value_next: T+1 value estimate for returns calculation.
    :return: discounted sum of future rewards as list.
    """
    discounted_r = np.zeros_like(r)
    running_add = value_next
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def get_gae(rewards, value_estimates, value_next=0.0, gamma=0.99, lambd=0.95):
    """
    Computes generalized advantage estimate for use in updating policy.
    :param rewards: list of rewards for time-steps t to T.
    :param value_next: Value estimate for time-step T+1.
    :param value_estimates: list of value estimates for time-steps t to T.
    :param gamma: Discount factor.
    :param lambd: GAE weighing factor.
    :return: list of advantage estimates for time-steps t to T.
    """
    value_estimates = np.asarray(value_estimates.tolist() + [value_next])
    delta_t = rewards + gamma * value_estimates[1:] - value_estimates[:-1]
    advantage = discount_rewards(r=delta_t, gamma=gamma*lambd)
    return advantage

