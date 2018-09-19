import json
import pathlib
from mlagents.trainers.buffer import Buffer
from mlagents.envs.brain import BrainParameters, BrainInfo


class DemonstrationLoader(object):
    @staticmethod
    def load(file_path, brain_name, sequence_length):
        """
        Loads and parses a demonstration file.
        :param sequence_length: Desired sequence length for buffer.
        :param file_path: Location of demonstration file (.demo).
        :param brain_name: Name of the brain this file corresponds to.
        :return: BrainParameter and Buffer objects containing demonstration data.
        """
        file_extension = pathlib.Path(file_path).suffix
        if file_extension != '.demo':
            raise ValueError("The file is not a '.demo' file. Please provide a file with the "
                             "correct extension.")

        # Parse demonstration file.
        experiences = []
        brain_params_dict = {}
        with open(file_path, "r") as read_file:
            for idx, line in enumerate(read_file):
                if idx == 0:
                    brain_params_dict = json.loads(line)
                else:
                    json_obj = json.loads(line)
                    if 'storedVectorActions' in json_obj.keys():
                        experiences.append(json_obj)

        brain_params = BrainParameters(brain_name, brain_params_dict)

        # Create and populate buffer using experiences
        demo_buffer = Buffer()
        for idx, experience in enumerate(experiences):
            if idx > len(experiences) - 2:
                break
            current_brain_info = DemonstrationLoader._make_brain_info(experiences[idx])
            next_brain_info = DemonstrationLoader._make_brain_info(experiences[idx + 1])
            demo_buffer[0].last_brain_info = current_brain_info
            for i in range(brain_params.number_visual_observations):
                demo_buffer[0]['visual_obs%d' % i] \
                    .append(current_brain_info.visual_observations[i][0])
            if brain_params.vector_observation_space_size > 0:
                demo_buffer[0]['vector_obs'] \
                    .append(current_brain_info.vector_observations[0])
            demo_buffer[0]['actions'].append(next_brain_info.previous_vector_actions[0])
            if next_brain_info.local_done[0]:
                demo_buffer.append_update_buffer(0, batch_size=None,
                                                 training_length=sequence_length)
        demo_buffer.append_update_buffer(0, batch_size=None,
                                         training_length=sequence_length)
        return brain_params, demo_buffer

    @staticmethod
    def _make_brain_info(experience):
        """
        Helper function which creates a BrainInfo object from an experience dictionary.
        :param experience: Experience dictionary.
        :return: BrainInfo.
        """
        brain_info = BrainInfo([experience["visualObservations"]], [experience["vectorObservation"]],
                  [experience["textObservation"]], [experience["memories"]],
                  [experience["reward"]], [experience["id"]], [experience["done"]],
                  [experience["storedVectorActions"]], [experience["storedTextActions"]],
                  [experience["maxStepReached"]], [experience["actionMasks"]])
        return brain_info
