from urllib.parse import urlparse, parse_qs


def _behavior_to_agent_id(behavior_name: str, unique_id: int) -> str:
    return f"{behavior_name}?agent_id={unique_id}"


def _agent_id_to_behavior(agent_id: str) -> str:
    return agent_id.split("?agent_id=")[0]


def _unwrap_batch_steps(batch_steps, behavior_name):
    decision_batch, termination_batch = batch_steps
    decision_id = [
        _behavior_to_agent_id(behavior_name, i) for i in decision_batch.agent_id
    ]
    termination_id = [
        _behavior_to_agent_id(behavior_name, i) for i in termination_batch.agent_id
    ]
    agents = decision_id + termination_id
    obs = {
        agent_id: [batch_obs[i] for batch_obs in termination_batch.obs]
        for i, agent_id in enumerate(termination_id)
    }
    if decision_batch.action_mask is not None:
        obs.update(
            {
                agent_id: {
                    "observation": [batch_obs[i] for batch_obs in decision_batch.obs],
                    "action_mask": [mask[i] for mask in decision_batch.action_mask],
                }
                for i, agent_id in enumerate(decision_id)
            }
        )
    else:
        obs.update(
            {
                agent_id: [batch_obs[i] for batch_obs in decision_batch.obs]
                for i, agent_id in enumerate(decision_id)
            }
        )
    obs = {k: v if len(v) > 1 else v[0] for k, v in obs.items()}
    dones = {agent_id: True for agent_id in termination_id}
    dones.update({agent_id: False for agent_id in decision_id})
    rewards = {
        agent_id: termination_batch.reward[i]
        for i, agent_id in enumerate(termination_id)
    }
    rewards.update(
        {agent_id: decision_batch.reward[i] for i, agent_id in enumerate(decision_id)}
    )
    cumulative_rewards = {k: v for k, v in rewards.items()}
    infos = {}
    for i, agent_id in enumerate(decision_id):
        infos[agent_id] = {}
        infos[agent_id]["behavior_name"] = behavior_name
        infos[agent_id]["group_id"] = decision_batch.group_id[i]
        infos[agent_id]["group_reward"] = decision_batch.group_reward[i]
    for i, agent_id in enumerate(termination_id):
        infos[agent_id] = {}
        infos[agent_id]["behavior_name"] = behavior_name
        infos[agent_id]["group_id"] = termination_batch.group_id[i]
        infos[agent_id]["group_reward"] = termination_batch.group_reward[i]
        infos[agent_id]["interrupted"] = termination_batch.interrupted[i]
    id_map = {agent_id: i for i, agent_id in enumerate(decision_id)}
    return agents, obs, dones, rewards, cumulative_rewards, infos, id_map


def _parse_behavior(full_behavior):
    parsed = urlparse(full_behavior)
    name = parsed.path
    ids = parse_qs(parsed.query)
    team_id: int = 0
    if "team" in ids:
        team_id = int(ids["team"][0])
    return name, team_id
