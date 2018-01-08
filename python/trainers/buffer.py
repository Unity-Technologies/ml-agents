import numpy as np

class BufferException(Exception):
    """
    Related to errors with the Buffer.
    """
    pass

class Buffer(dict):
	class AgentBuffer(dict):
		class AgentBufferField(list):
			def __str__(self):
				return str(np.array(self).shape)
			def extend(self, data):
				"""
				Ads a list of np.arrays to the end of the list of np.arrays.
				:param data: The np.array list to append.
				"""
				#TODO: need to handle the case the data is not the right size
				self += list(np.array(data))
			def set(self, data):
				"""
				Sets the list of np.array to the input data
				:param data: The np.array list to be set.
				"""
				self[:] = []
				self[:] = list(np.array(data))
			def get_batch(self, batch_size = None, training_length = None, sequential = True):
				"""
				Retrieve the last batch_size elements of length training_length
				from the list of np.array
				:param batch_size: The number of elements to retrieve. If None: 
				All elements will be retrieved.
				:param training_length: The length of the sequence to be retrieved. If
				None: only takes one element.
				:param sequential: If true and training_length is not None: the elements 
				will not repeat in the sequence. [a,b,c,d,e] with training_length = 2 and
				sequential=True gives [[a,b],[c,d],[0,e]]. If sequential=False gives
				[[a,b],[b,c],[c,d],[d,e]]
				"""
				#TODO: Decide what to do if there enough points to retrieve 
				if training_length is None:
					if batch_size is None:
						#return all of them
						return np.array(self)
					else:
						# return the batch_size last elements
						if batch_size > len(self):
							raise BufferException("Batch size requested is too large")
						return np.array(self[-batch_size:])
				else:
					if not sequential:
						if batch_size is None:
							# retrieve the maximum number of elements
							batch_size = len(self) - training_length + 1
						if (len(self) - training_length + 1) < batch_size :
							raise BufferException("The batch size and training length requested for get_batch where" 
								" too large given the current number of data points.")
							return 
						tmp_list = []
						for end in range(len(self)-batch_size+1, len(self)+1):
							tmp_list += [np.array(self[end-training_length:end])]
						return np.array(tmp_list)
					if sequential:
						# No padding at all
						leftover = len(self) % training_length
						if batch_size is None:
							# retrieve the maximum number of elements
							batch_size = len(self) // training_length +1 *(leftover != 0)
						if batch_size > (len(self) // training_length +1 *(leftover != 0)):
							raise BufferException("The batch size and training length requested for get_batch where" 
								" too large given the current number of data points.")
							return 
						tmp_list = []
						padding = np.array(self[-1]) * 0 
						for end in range(len(self), len(self) % training_length , -training_length)[:batch_size]:
							tmp_list += [np.array(self[end-training_length:end])]
						if (leftover != 0) and (len(tmp_list) < batch_size):
							tmp_list +=[np.array([padding]*(training_length - leftover)+self[:leftover])]
						return np.array(tmp_list)
					# TODO: decide if we need to reset the local buffer now ?	
			def reset_field(self):
				"""
				Resets the AgentBufferField
				"""
				self[:] = []
		def __str__(self):
			return ", ".join(["'{0}' : {1}".format(k, str(self[k])) for k in self.keys()])
		def reset_agent(self):
			"""
			Resets the AgentBuffer
			"""
			#TODO: There might be some garbage collection issues ?
			for k in self.keys():
				try:
					self[k].reset_field()
				except:
					print(k)
		def __getitem__(self, key):
			if key not in self.keys():
				self[key] = self.AgentBufferField()
			return super(Buffer.AgentBuffer, self).__getitem__(key)
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
				if key not in self.keys():
					return False
				if ((l != None) and (l!=len(self[key]))):
					return False
				l = len(self[key])
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
			s = np.arange(len(self[key_list[0]]))
			np.random.shuffle(s)
			for key in key_list:
				self[key][:] = [self[key][i] for i in s]
				#memory leak here ?
				# with l as [self[key][i] for i in s]:
				# 	self[key].reset_field()
				# 	self[key] += l
				# self[key] = Buffer.AgentBuffer.AgentBufferField([self[key][i] for i in s])
				# self[key].reorder(s)
	def __init__(self):
		self.global_buffer = self.AgentBuffer() 
		super(Buffer, self).__init__()
	def __str__(self):
		return "global buffer :\n\t{0}\nlocal_buffers :\n{1}".format(str(self.global_buffer), 
			'\n'.join(['\tagent {0} :{1}'.format(k, str(self[k])) for k in self.keys()]))
	def __getitem__(self, key):
		if key not in self.keys():
			self[key] = self.AgentBuffer()
		return super(Buffer, self).__getitem__(key)
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
		self.global_buffer.reset_agent()
	def reset_all(self):
		"""
		Resets the global buffer and all the local local_buffers
		"""
		# TODO: Need a better name
		self.global_buffer.reset_agent()
		agent_ids = list(self.keys())
		for k in agent_ids:
			self[k].reset_agent()
	def append_global(self, agent_id ,key_list = None,  batch_size = None, training_length = None):
		"""
		Appends the buffer of an agent to the global buffer.
		:param agent_id: The id of the agent which data will be appended
		:param key_list: The fields that must be added. If None: all fields will be appended.
		:param batch_size: The number of elements that must be appended. If None: All of them will be.
		:param training_length: The length of the samples that must be appended. If None: only takes one element.
		"""
		if key_list is None:
			key_list = self[agent_id].keys()
		if not self[agent_id].check_length(key_list):
			raise BufferException("The length of the fields {0} for agent {1} where not of comparable length"
				.format(key_list, agent_id))
		for field_key in key_list:
			self.global_buffer[field_key].extend(
				self[agent_id][field_key].get_batch(batch_size =batch_size, training_length =training_length)
			)
	def append_all_agent_batch_to_global(self, key_list = None,  batch_size = None, training_length = None):
		"""
		Appends the buffer of all agents to the global buffer.
		:param key_list: The fields that must be added. If None: all fields will be appended.
		:param batch_size: The number of elements that must be appended. If None: All of them will be.
		:param training_length: The length of the samples that must be appended. If None: only takes one element.
		"""
		#TODO: Maybe no batch_size, only training length and a flag corresponding to "get only last of training_length"
		for agent_id in self.keys():
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

