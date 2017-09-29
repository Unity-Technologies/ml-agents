import numpy as np

history_keys = ['states', 'observations', 'actions', 'rewards', 'action_probs', 'epsilons',
                'value_estimates', 'advantages', 'discounted_returns']


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


def empty_local_history(agent_dict):
    """
    Empties the experience history for a single agent.
    :param agent_dict: Dictionary of agent experience history.
    :return: Emptied dictionary (except for cumulative_reward and episode_steps).
    """
    for key in history_keys:
        agent_dict[key] = []
    return agent_dict


def vectorize_history(agent_dict):
    """
    Converts dictionary of lists into dictionary of numpy arrays.
    :param agent_dict: Dictionary of agent experience history.
    :return: dictionary of numpy arrays.
    """
    for key in history_keys:
        agent_dict[key] = np.array(agent_dict[key])
    return agent_dict


def empty_all_history(agent_info):
    """
    Clears all agent histories and resets reward and episode length counters.
    :param agent_info: a BrainInfo object.
    :return: an emptied history dictionary.
    """
    history_dict = {}
    for agent in agent_info.agents:
        history_dict[agent] = {}
        history_dict[agent] = empty_local_history(history_dict[agent])
        history_dict[agent]['cumulative_reward'] = 0
        history_dict[agent]['episode_steps'] = 0
    return history_dict


def append_history(global_buffer, local_buffer=None):
    """
    Appends agent experience history to global history buffer.
    :param global_buffer: Global buffer for all agents experiences.
    :param local_buffer: Local history for individual agents experiences.
    :return: Global buffer with new experiences added.
    """
    for key in history_keys:
        global_buffer[key] = np.concatenate([global_buffer[key], local_buffer[key]], axis=0)
    return global_buffer


def set_history(global_buffer, local_buffer=None):
    """
    Creates new global_buffer from existing local_buffer
    :param global_buffer: Global buffer for all agents experiences.
    :param local_buffer: Local history for individual agents experiences.
    :return: Global buffer with new experiences.
    """
    for key in history_keys:
        global_buffer[key] = np.copy(local_buffer[key])
    return global_buffer


def shuffle_buffer(global_buffer):
    """
    Randomizes experiences in global_buffer
    :param global_buffer: training_buffer to randomize.
    :return: Randomized buffer
    """
    s = np.arange(global_buffer[history_keys[2]].shape[0])
    np.random.shuffle(s)
    for key in history_keys:
        if len(global_buffer[key]) > 0:
            global_buffer[key] = global_buffer[key][s]
    return global_buffer
