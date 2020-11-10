# Sample observations, dones, rewards from experience replay buffer
observations, next_observations, dones, rewards = sample_batch()

# Evaluate current policy on sampled observations
(
    sampled_actions,
    log_probs,
    entropies,
    sampled_values,
) = policy.sample_actions(observations)

# Evaluate Q networks on observations and actions
q1p_out, q2p_out = value_network(observations, sampled_actions)
q1_out, q2_out = value_network(observations, actions)

# Evaluate target network on next observations
with torch.no_grad():
    target_values = target_network(next_observations)

# Evaluate losses
q1_loss, q2_loss = sac_q_loss(q1_out, q2_out, target_values, dones, rewards)
value_loss = sac_value_loss(log_probs, sampled_values, q1p_out, q2p_out)
policy_loss = sac_policy_loss(log_probs, q1p_out)
entropy_loss = sac_entropy_loss(log_probs)

total_value_loss = q1_loss + q2_loss + value_loss

# Backprop and weights update
policy_optimizer.zero_grad()
policy_loss.backward()
policy_optimizer.step()

value_optimizer.zero_grad()
total_value_loss.backward()
value_optimizer.step()

entropy_optimizer.zero_grad()
entropy_loss.backward()
entropy_optimizer.step()