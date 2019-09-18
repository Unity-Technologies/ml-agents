import numpy as np
import h5py

from mlagents.envs.exception import UnityException


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

        class AgentBufferField(list):
            """
            AgentBufferField is a list of numpy arrays. When an agent collects a field, you can add it to his
            AgentBufferField with the append method.
            """

            def __init__(self):
                self.padding_value = 0
                super(Buffer.AgentBuffer.AgentBufferField, self).__init__()

            def __str__(self):
                return str(np.array(self).shape)

            def append(self, element, padding_value=0):
                """
                Adds an element to this list. Also lets you change the padding
                type, so that it can be set on append (e.g. action_masks should
                be padded with 1.)
                :param element: The element to append to the list.
                :param padding_value: The value used to pad when get_batch is called.
                """
                super(Buffer.AgentBuffer.AgentBufferField, self).append(element)
                self.padding_value = padding_value

            def extend(self, data):
                """
                Adds a list of np.arrays to the end of the list of np.arrays.
                :param data: The np.array list to append.
                """
                self += list(np.array(data))

            def set(self, data):
                """
                Sets the list of np.array to the input data
                :param data: The np.array list to be set.
                """
                self[:] = []
                self[:] = list(np.array(data))

            def get_batch(self, batch_size=None, training_length=1, sequential=True):
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
                if sequential:
                    # The sequences will not have overlapping elements (this involves padding)
                    leftover = len(self) % training_length
                    # leftover is the number of elements in the first sequence (this sequence might need 0 padding)
                    if batch_size is None:
                        # retrieve the maximum number of elements
                        batch_size = len(self) // training_length + 1 * (leftover != 0)
                    # The maximum number of sequences taken from a list of length len(self) without overlapping
                    # with padding is equal to batch_size
                    if batch_size > (
                        len(self) // training_length + 1 * (leftover != 0)
                    ):
                        raise BufferException(
                            "The batch size and training length requested for get_batch where"
                            " too large given the current number of data points."
                        )
                    if batch_size * training_length > len(self):
                        padding = np.array(self[-1]) * self.padding_value
                        return np.array(
                            [padding] * (training_length - leftover) + self[:],
                            dtype=np.float32,
                        )
                    else:
                        return np.array(
                            self[len(self) - batch_size * training_length :],
                            dtype=np.float32,
                        )
                else:
                    # The sequences will have overlapping elements
                    if batch_size is None:
                        # retrieve the maximum number of elements
                        batch_size = len(self) - training_length + 1
                    # The number of sequences of length training_length taken from a list of len(self) elements
                    # with overlapping is equal to batch_size
                    if (len(self) - training_length + 1) < batch_size:
                        raise BufferException(
                            "The batch size and training length requested for get_batch where"
                            " too large given the current number of data points."
                        )
                    tmp_list = []
                    for end in range(len(self) - batch_size + 1, len(self) + 1):
                        tmp_list += self[end - training_length : end]
                    return np.array(tmp_list, dtype=np.float32)

            def reset_field(self):
                """
                Resets the AgentBufferField
                """
                self[:] = []

        def __init__(self):
            self.last_brain_info = None
            self.last_take_action_outputs = None
            super(Buffer.AgentBuffer, self).__init__()

        def __str__(self):
            return ", ".join(
                ["'{0}' : {1}".format(k, str(self[k])) for k in self.keys()]
            )

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
            length = None
            for key in key_list:
                if key not in self.keys():
                    return False
                if (length is not None) and (length != len(self[key])):
                    return False
                length = len(self[key])
            return True

        def shuffle(self, sequence_length, key_list=None):
            """
            Shuffles the fields in key_list in a consistent way: The reordering will
            be the same across fields.
            :param key_list: The fields that must be shuffled.
            """
            if key_list is None:
                key_list = list(self.keys())
            if not self.check_length(key_list):
                raise BufferException(
                    "Unable to shuffle if the fields are not of same length"
                )
            s = np.arange(len(self[key_list[0]]) // sequence_length)
            np.random.shuffle(s)
            for key in key_list:
                tmp = []
                for i in s:
                    tmp += self[key][i * sequence_length : (i + 1) * sequence_length]
                self[key][:] = tmp

        def make_mini_batch(self, start, end):
            """
            Creates a mini-batch from buffer.
            :param start: Starting index of buffer.
            :param end: Ending index of buffer.
            :return: Dict of mini batch.
            """
            mini_batch = {}
            for key in self:
                mini_batch[key] = self[key][start:end]
            return mini_batch

        def sample_mini_batch(self, batch_size, sequence_length=1):
            """
            Creates a mini-batch from a random start and end.
            :param batch_size: number of elements to withdraw.
            :param sequence_length: Length of sequences to sample.
                Number of sequences to sample will be batch_size/sequence_length.
            """
            num_seq_to_sample = batch_size // sequence_length
            mini_batch = Buffer.AgentBuffer()
            buff_len = len(next(iter(self.values())))
            num_sequences_in_buffer = buff_len // sequence_length
            start_idxes = (
                np.random.randint(num_sequences_in_buffer, size=num_seq_to_sample)
                * sequence_length
            )  # Sample random sequence starts
            for i in start_idxes:
                for key in self:
                    mini_batch[key].extend(self[key][i : i + sequence_length])
            return mini_batch

        def save_to_file(self, file_object):
            """
            Saves the AgentBuffer to a file-like object.
            """
            with h5py.File(file_object) as write_file:
                for key, data in self.items():
                    write_file.create_dataset(
                        key, data=data, dtype="f", compression="gzip"
                    )

        def load_from_file(self, file_object):
            """
            Loads the AgentBuffer from a file-like object.
            """
            with h5py.File(file_object) as read_file:
                for key in list(read_file.keys()):
                    self[key] = Buffer.AgentBuffer.AgentBufferField()
                    # extend() will convert the numpy array's first dimension into list
                    self[key].extend(read_file[key][()])

    def __init__(self):
        self.update_buffer = self.AgentBuffer()
        super(Buffer, self).__init__()

    def __str__(self):
        return "update buffer :\n\t{0}\nlocal_buffers :\n{1}".format(
            str(self.update_buffer),
            "\n".join(
                ["\tagent {0} :{1}".format(k, str(self[k])) for k in self.keys()]
            ),
        )

    def __getitem__(self, key):
        if key not in self.keys():
            self[key] = self.AgentBuffer()
        return super(Buffer, self).__getitem__(key)

    def reset_update_buffer(self):
        """
        Resets the update buffer
        """
        self.update_buffer.reset_agent()

    def truncate_update_buffer(self, max_length, sequence_length=1):
        """
        Truncates the update buffer to a certain length.

        This can be slow for large buffers. We compensate by cutting further than we need to, so that
        we're not truncating at each update. Note that we must truncate an integer number of sequence_lengths
        param: max_length: The length at which to truncate the buffer.
        """
        current_length = len(next(iter(self.update_buffer.values())))
        # make max_length an integer number of sequence_lengths
        max_length -= max_length % sequence_length
        if current_length > max_length:
            for _key in self.update_buffer.keys():
                self.update_buffer[_key] = self.update_buffer[_key][
                    current_length - max_length :
                ]

    def reset_local_buffers(self):
        """
        Resets all the local local_buffers
        """
        agent_ids = list(self.keys())
        for k in agent_ids:
            self[k].reset_agent()

    def append_update_buffer(
        self, agent_id, key_list=None, batch_size=None, training_length=None
    ):
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
            raise BufferException(
                "The length of the fields {0} for agent {1} where not of same length".format(
                    key_list, agent_id
                )
            )
        for field_key in key_list:
            self.update_buffer[field_key].extend(
                self[agent_id][field_key].get_batch(
                    batch_size=batch_size, training_length=training_length
                )
            )

    def append_all_agent_batch_to_update_buffer(
        self, key_list=None, batch_size=None, training_length=None
    ):
        """
        Appends the buffer of all agents to the update buffer.
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        for agent_id in self.keys():
            self.append_update_buffer(agent_id, key_list, batch_size, training_length)
