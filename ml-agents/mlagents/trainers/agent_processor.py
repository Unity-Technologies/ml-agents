from typing import List, Union

from mlagents.trainers.buffer import AgentBuffer, BufferException


class ProcessingBuffer(dict):
    """
    ProcessingBuffer contains a dictionary of AgentBuffer. The AgentBuffers are indexed by agent_id.
    """

    def __str__(self):
        return "local_buffers :\n{0}".format(
            "\n".join(["\tagent {0} :{1}".format(k, str(self[k])) for k in self.keys()])
        )

    def __getitem__(self, key):
        if key not in self.keys():
            self[key] = AgentBuffer()
        return super().__getitem__(key)

    def reset_local_buffers(self) -> None:
        """
        Resets all the local AgentBuffers.
        """
        for buf in self.values():
            buf.reset_agent()

    def append_to_update_buffer(
        self,
        update_buffer: AgentBuffer,
        agent_id: Union[int, str],
        key_list: List[str] = None,
        batch_size: int = None,
        training_length: int = None,
    ) -> None:
        """
        Appends the buffer of an agent to the update buffer.
        :param update_buffer: A reference to an AgentBuffer to append the agent's buffer to
        :param agent_id: The id of the agent which data will be appended
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        if key_list is None:
            key_list = self[agent_id].keys()
        if not self[agent_id].check_length(key_list):
            raise BufferException(
                "The length of the fields {0} for agent {1} were not of same length".format(
                    key_list, agent_id
                )
            )
        for field_key in key_list:
            update_buffer[field_key].extend(
                self[agent_id][field_key].get_batch(
                    batch_size=batch_size, training_length=training_length
                )
            )

    def append_all_agent_batch_to_update_buffer(
        self,
        update_buffer: AgentBuffer,
        key_list: List[str] = None,
        batch_size: int = None,
        training_length: int = None,
    ) -> None:
        """
        Appends the buffer of all agents to the update buffer.
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        for agent_id in self.keys():
            self.append_to_update_buffer(
                update_buffer, agent_id, key_list, batch_size, training_length
            )
