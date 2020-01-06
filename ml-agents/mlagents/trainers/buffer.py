import numpy as np
import h5py
from typing import List, BinaryIO

from mlagents_envs.exception import UnityException


class BufferException(UnityException):
    """
    Related to errors with the Buffer.
    """

    pass


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
            super().__init__()

        def __str__(self):
            return str(np.array(self).shape)

        def append(self, element: np.ndarray, padding_value: float = 0.0) -> None:
            """
            Adds an element to this list. Also lets you change the padding
            type, so that it can be set on append (e.g. action_masks should
            be padded with 1.)
            :param element: The element to append to the list.
            :param padding_value: The value used to pad when get_batch is called.
            """
            super().append(element)
            self.padding_value = padding_value

        def extend(self, data: np.ndarray) -> None:
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
            # Make sure we convert incoming data to float32 if it's a float
            dtype = None
            if data is not None and len(data) and isinstance(data[0], float):
                dtype = np.float32
            self[:] = []
            self[:] = list(np.array(data, dtype=dtype))

        def get_batch(
            self,
            batch_size: int = None,
            training_length: int = 1,
            sequential: bool = True,
        ) -> np.ndarray:
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
                if batch_size > (len(self) // training_length + 1 * (leftover != 0)):
                    raise BufferException(
                        "The batch size and training length requested for get_batch where"
                        " too large given the current number of data points."
                    )
                if batch_size * training_length > len(self):
                    padding = np.array(self[-1], dtype=np.float32) * self.padding_value
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
                tmp_list: List[np.ndarray] = []
                for end in range(len(self) - batch_size + 1, len(self) + 1):
                    tmp_list += self[end - training_length : end]
                return np.array(tmp_list, dtype=np.float32)

        def reset_field(self) -> None:
            """
            Resets the AgentBufferField
            """
            self[:] = []

    def __init__(self):
        self.last_brain_info = None
        self.last_take_action_outputs = None
        super().__init__()

    def __str__(self):
        return ", ".join(["'{0}' : {1}".format(k, str(self[k])) for k in self.keys()])

    def reset_agent(self) -> None:
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
        return super().__getitem__(key)

    def check_length(self, key_list: List[str]) -> bool:
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

    def shuffle(self, sequence_length: int, key_list: List[str] = None) -> None:
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
            tmp: List[np.ndarray] = []
            for i in s:
                tmp += self[key][i * sequence_length : (i + 1) * sequence_length]
            self[key][:] = tmp

    def make_mini_batch(self, start: int, end: int) -> "AgentBuffer":
        """
        Creates a mini-batch from buffer.
        :param start: Starting index of buffer.
        :param end: Ending index of buffer.
        :return: Dict of mini batch.
        """
        mini_batch = AgentBuffer()
        for key in self:
            mini_batch[key] = self[key][start:end]
        return mini_batch

    def sample_mini_batch(
        self, batch_size: int, sequence_length: int = 1
    ) -> "AgentBuffer":
        """
        Creates a mini-batch from a random start and end.
        :param batch_size: number of elements to withdraw.
        :param sequence_length: Length of sequences to sample.
            Number of sequences to sample will be batch_size/sequence_length.
        """
        num_seq_to_sample = batch_size // sequence_length
        mini_batch = AgentBuffer()
        buff_len = self.num_experiences
        num_sequences_in_buffer = buff_len // sequence_length
        start_idxes = (
            np.random.randint(num_sequences_in_buffer, size=num_seq_to_sample)
            * sequence_length
        )  # Sample random sequence starts
        for i in start_idxes:
            for key in self:
                mini_batch[key].extend(self[key][i : i + sequence_length])
        return mini_batch

    def save_to_file(self, file_object: BinaryIO) -> None:
        """
        Saves the AgentBuffer to a file-like object.
        """
        with h5py.File(file_object, "w") as write_file:
            for key, data in self.items():
                write_file.create_dataset(key, data=data, dtype="f", compression="gzip")

    def load_from_file(self, file_object: BinaryIO) -> None:
        """
        Loads the AgentBuffer from a file-like object.
        """
        with h5py.File(file_object, "r") as read_file:
            for key in list(read_file.keys()):
                self[key] = AgentBuffer.AgentBufferField()
                # extend() will convert the numpy array's first dimension into list
                self[key].extend(read_file[key][()])

    def truncate(self, max_length: int, sequence_length: int = 1) -> None:
        """
        Truncates the buffer to a certain length.

        This can be slow for large buffers. We compensate by cutting further than we need to, so that
        we're not truncating at each update. Note that we must truncate an integer number of sequence_lengths
        param: max_length: The length at which to truncate the buffer.
        """
        current_length = self.num_experiences
        # make max_length an integer number of sequence_lengths
        max_length -= max_length % sequence_length
        if current_length > max_length:
            for _key in self.keys():
                self[_key] = self[_key][current_length - max_length :]

    def resequence_and_append(
        self,
        target_buffer: "AgentBuffer",
        key_list: List[str] = None,
        batch_size: int = None,
        training_length: int = None,
    ) -> None:
        """
        Takes in a batch size and training length (sequence length), and appends this AgentBuffer to target_buffer
        properly padded for LSTM use. Optionally, use key_list to restrict which fields are inserted into the new
        buffer.
        :param target_buffer: The buffer which to append the samples to.
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        if key_list is None:
            key_list = list(self.keys())
        if not self.check_length(key_list):
            raise BufferException(
                "The length of the fields {0} were not of same length".format(key_list)
            )
        for field_key in key_list:
            target_buffer[field_key].extend(
                self[field_key].get_batch(
                    batch_size=batch_size, training_length=training_length
                )
            )

    @property
    def num_experiences(self) -> int:
        """
        The number of agent experiences in the AgentBuffer, i.e. the length of the buffer.

        An experience consists of one element across all of the fields of this AgentBuffer.
        Note that these all have to be the same length, otherwise shuffle and append_to_update_buffer
        will fail.
        """
        if self.values():
            return len(next(iter(self.values())))
        else:
            return 0
