import numpy as np

from unityagents.exception import UnityException


class BufferException(UnityException):
    """
    Related to errors with the Buffer.
    """
    pass


class Buffer(dict):
    """
    Buffer contains a dictionary of AgentBuffer. The AgentBuffers are indexed by agent_id.
    Buffer also contains an update_buffer that corresponds to the buffer used when updating the model.
    """

    class AgentBuffer(dict):
        """
        AgentBuffer contains a dictionary of AgentBufferFields. Each agent has his own AgentBuffer.
        The keys correspond to the name of the field. Example: state, action
        """

        class AgentBufferField(object):
            """
            AgentBufferField is a list of numpy arrays. When an agent collects a field, you can add it to his 
            AgentBufferField with the append method.
            """
            def __init__(self):
                self._array = None

            def __str__(self):
                return str(self._array.shape)

            def __len__(self):
                if self._array is None:
                    return 0
                else:
                    return len(self._array)

            def __getitem__(self, item):
                return self._array[item]

            def extend(self, data):
                """
                Ads a list of np.arrays to the end of the list of np.arrays.
                :param data: The np.array list to append.
                """
                if self._array is None:
                    self._array = data
                else:
                    self._array = np.concatenate([self._array, data], axis=0)

            def append(self, data):
                extended_data = np.expand_dims(data, 0)

                if self._array is None:
                    self._array = extended_data
                else:
                    self._array = np.concatenate([self._array, extended_data])

            def set(self, data):
                """
                Sets the list of np.array to the input data
                :param data: The np.array list to be set.
                """
                self._array = np.array(data)

            def re_index(self, indices):
                self._array = self._array[indices]

            def get_batch(self, batch_size=None, training_length=None, sequential=True):
                """
                Retrieve the last batch_size elements of length training_length
                from the list of np.array
                :param batch_size: The number of elements to retrieve. If None: 
                All elements will be retrieved.
                :param training_length: The length of the sequence to be retrieved. If
                None: only takes one element.
                :param sequential: If true and training_length is not None: the elements 
                will not repeat in the sequence. [a,b,c,d,e] with training_length = 2 and
                sequential=True gives [[0,a],[b,c],[d,e]]. If sequential=False gives
                [[a,b],[b,c],[c,d],[d,e]]
                """
                if training_length is None:
                    # When the training length is None, the method returns a list of elements,
                    # not a list of sequences of elements.
                    if batch_size is None:
                        # If batch_size is None : All the elements of the AgentBufferField are returned.
                        return self._array
                    else:
                        # return the batch_size last elements
                        if batch_size > len(self._array):
                            raise BufferException("Batch size requested is too large")
                        return self._array[-batch_size:]
                else:
                    # The training_length is not None, the method returns a list of SEQUENCES of elements
                    if not sequential:
                        # The sequences will have overlapping elements
                        if batch_size is None:
                            # retrieve the maximum number of elements
                            batch_size = len(self._array) - training_length + 1
                        # The number of sequences of length training_length taken from a list of len(self) elements
                        # with overlapping is equal to batch_size
                        if (len(self._array) - training_length + 1) < batch_size:
                            raise BufferException("The batch size and training length requested for get_batch where"
                                                  " too large given the current number of data points.")
                        tmp_list = []
                        for end in range(len(self._array) - batch_size + 1, len(self._array) + 1):
                            tmp_list.append(self._array[end - training_length:end])
                        return np.stack(tmp_list, axis=0)
                    if sequential:
                        # The sequences will not have overlapping elements (this involves padding)
                        leftover = len(self._array) % training_length
                        # leftover is the number of elements in the first sequence (this sequence might need 0 padding)
                        if batch_size is None:
                            # retrieve the maximum number of elements
                            batch_size = len(self._array) // training_length + 1 * (leftover != 0)
                        # The maximum number of sequences taken from a list of length len(self) without overlapping
                        # with padding is equal to batch_size
                        if batch_size > (len(self._array) // training_length + 1 * (leftover != 0)):
                            raise BufferException("The batch size and training length requested for get_batch where"
                                                  " too large given the current number of data points.")
                        tmp_list = []
                        padding = self._array[-1] * 0
                        # The padding is made with zeros and its shape is given by the shape of the last element
                        for end in range(len(self._array), len(self._array) % training_length, -training_length)[:batch_size]:
                            tmp_list.append(self._array[end - training_length:end])
                        if (leftover != 0) and (len(tmp_list) < batch_size):
                            # tmp_list += [np.array([padding] * (training_length - leftover) + self._array[:leftover])]
                            lover = self._array[:leftover]
                            if len(lover.shape) > 1:
                                tiled = np.tile(padding, [training_length - leftover, 1])
                            else:
                                tiled = np.tile(padding, [training_length - leftover])
                            tmp_list.append(np.concatenate([tiled, lover], axis=0))
                        tmp_list.reverse()
                        return np.stack(tmp_list, axis=0)

            def reset_field(self):
                """
                Resets the AgentBufferField
                """
                self._array = None

        def __init__(self):
            self.last_brain_info = None
            self.last_take_action_outputs = None
            super(Buffer.AgentBuffer, self).__init__()

        def __str__(self):
            return ", ".join(["'{0}' : {1}".format(k, str(self[k])) for k in self.keys()])

        def reset_agent(self):
            """
            Resets the AgentBuffer
            """
            for k in self.keys():
                self[k].reset_field()
            self.last_brain_info = None
            self.last_take_action_outputs = None

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
                if (l is not None) and (l != len(self[key])):
                    return False
                l = len(self[key])
            return True

        def shuffle(self, key_list=None):
            """
            Shuffles the fields in key_list in a consistent way: The reordering will
            be the same across fields.
            :param key_list: The fields that must be shuffled.
            """
            if key_list is None:
                key_list = list(self.keys())
            if not self.check_length(key_list):
                raise BufferException("Unable to shuffle if the fields are not of same length")
            s = np.arange(len(self[key_list[0]]))
            np.random.shuffle(s)
            for key in key_list:
                self[key].re_index(s)

    def __init__(self):
        self.update_buffer = self.AgentBuffer()
        super(Buffer, self).__init__()

    def __str__(self):
        return "Update Buffer :\n\t{0}\nLocal_Buffers :\n{1}".format(str(self.update_buffer),
                                                                     '\n'.join(
                                                                         ['\tagent {0} :{1}'.format(k, str(self[k])) for
                                                                          k in self.keys()]))

    def __getitem__(self, key):
        if key not in self.keys():
            self[key] = self.AgentBuffer()
        return super(Buffer, self).__getitem__(key)

    def reset_update_buffer(self):
        """
        Resets the update buffer
        """
        self.update_buffer.reset_agent()

    def reset_all(self):
        """
        Resets all the local local_buffers
        """
        agent_ids = list(self.keys())
        for k in agent_ids:
            self[k].reset_agent()

    def append_update_buffer(self, agent_id, key_list=None, batch_size=None, training_length=None):
        """
        Appends the buffer of an agent to the update buffer.
        :param agent_id: The id of the agent which data will be appended
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        if key_list is None:
            key_list = self[agent_id].keys()
        if not self[agent_id].check_length(key_list):
            raise BufferException("The length of the fields {0} for agent {1} where not of same length"
                                  .format(key_list, agent_id))
        for field_key in key_list:
            self.update_buffer[field_key].extend(
                self[agent_id][field_key].get_batch(batch_size=batch_size, training_length=training_length))

    def append_all_agent_batch_to_update_buffer(self, key_list=None, batch_size=None, training_length=None):
        """
        Appends the buffer of all agents to the update buffer.
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        for agent_id in self.keys():
            self.append_update_buffer(agent_id, key_list, batch_size, training_length)
