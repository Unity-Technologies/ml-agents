import logging
import tensorflow as tf
import numpy as np

from mlagents.trainers.models import LearningModel
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.buffer import Buffer

logger = logging.getLogger("mlagents.trainers")


class DemoHelper(object):
    sequence_length_name = 'sequence_length'
    use_recurrent_name = 'use_recurrent'
    memory_size = 'memory_size'
    # TODO : Implement recurrent
    # TODO : Implement Discrete Control
    # TODO : Pretrain the critic ? All of the critics ? Use a gradient gate ?
    # TODO : tune lambdas during the training process (start at 1 and slowly go to right value)
    #  TODO : check the equality between brain and brain_params

    def __init__(self, brain, parameters):
        self.n_sequences = 128
        self.n_epoch = 500
        self.batches_per_epoch = 10

        self.use_recurrent = parameters[self.use_recurrent_name]
        self.sequence_length = 1
        self.m_size = 0
        if self.use_recurrent:
            self.sequence_length = parameters[self.sequence_length_name]
            self.m_size = parameters[self.memory_size]

        self.brain = brain

        buffer_name = parameters["demo_path"]
        brain_params, self.demonstration_buffer = demo_to_buffer(
            buffer_name,
            self.sequence_length)

        self.n_sequences = min(self.n_sequences, len(self.demonstration_buffer.update_buffer['actions']))
        
    def update_policy(self):
        """
        Updates the policy.
        """
        for iteration in range(self.n_epoch):
            self.demonstration_buffer.update_buffer.shuffle()
            batch_losses = []
            num_batches = min(len(self.demonstration_buffer.update_buffer['actions']) //
                              self.n_sequences, self.batches_per_epoch)
            for i in range(num_batches):
                update_buffer = self.demonstration_buffer.update_buffer
                start = i * self.n_sequences
                end = (i + 1) * self.n_sequences
                mini_batch = update_buffer.make_mini_batch(start, end)
                loss = self._update(mini_batch, self.n_sequences)
                batch_losses.append(loss)
            logger.info("Pre-Training loss at iteration "+str(iteration)+" : "+str(np.mean(batch_losses)))

    def concat_demos(self, external_buffer: Buffer):
        """
        Takes in a mini batch, loads some demonstrations from a demo file, and concatenates them together. 
        Sets the Advantages for the demos appropriately so that it is weighed less over time. 
        """
        print(external_buffer.update_buffer)
        update_buffer = self.demonstration_buffer.update_buffer
        update_buffer.shuffle()
        update_buffer['masks'] = Buffer.AgentBuffer.AgentBufferField()
        update_buffer['masks'].extend(len(update_buffer['actions'])*[1])
        update_buffer['prev_action'] = Buffer.AgentBuffer.AgentBufferField()
        update_buffer['prev_action'].extend(len(update_buffer['actions'])*[1])
        update_buffer['advantages'] = Buffer.AgentBuffer.AgentBufferField()
        update_buffer['advantages'].extend(len(update_buffer['actions'])*[1])
        external_buffer.append_update_buffer_with_ext(update_buffer, self.n_sequences, self.sequence_length)
        print("After appending")
        print(external_buffer.update_buffer)
