from mlagents.envs import AllBrainInfo, Dict
from mlagents.trainers.trainer import Trainer

K = 1 # Constant for rating changes, higher is less stable but converges faster


# Derived from https://metinmediamath.wordpress.com/2013/11/27/how-to-calculate-the-elo-rating-including-example/
def compute_elo_rating_changes(rating1, rating2, result):
    r1 = pow(10, rating1 / 400)
    r2 = pow(10, rating2 / 400)

    sum = r1 + r2
    e1 = r1 / sum

    s1 = 1 if result == "win" else 0

    change = K * (s1 - e1)

    return change


class EloTracker(object):

    def __init__(self, run_id):
        self.run_id = run_id
        self.previous_agent_brains = {}

    def update_elo_ratings(self, info: AllBrainInfo, trainers: Dict[str, Trainer]):
        agent_policies = {}
        should_update_rating = {}
        processed_agents = []
        for brain_name, brain_info in info.items():
            for agent in brain_info.agent_ids:
                """
                The match result we get at Frame F(n) is the result of the match played during the previous frames F(n-1), F(n-2), ...
                The brain of agents could have changed between F(n) and F(n-1) 
                    and we want to make sure to assign the victory to the brain that the winning agent was using during the match.
                So we assign the match results we got at F(n) to the brains that the agents had at F(n-1)
                """
                if agent in self.previous_agent_brains:
                    trainer = trainers[self.previous_agent_brains[agent]]
                    if hasattr(trainer, 'agent_policies'):
                        agent_policies[agent] = trainer.agent_policies[agent]
                    else:
                        agent_policies[agent] = trainer.policy
                    should_update_rating[agent] = trainer.should_update_elo_rating()
                else:
                    # If the agent didn't exist on the last frame he can't have been part of a match so we ignore him.
                    processed_agents.append(agent)
                self.previous_agent_brains[agent] = brain_name

        for brain_name, brain_info in info.items():
            for idx, agent in enumerate(brain_info.agent_ids):
                if agent in processed_agents:
                    continue

                text = brain_info.text_observations[idx]
                split = text.split('|')

                if len(split) == 2:
                    opponent = int(split[0])
                    result = split[1]

                    if result != "play":
                        agent_rating = agent_policies[agent].get_elo_rating()
                        opponent_rating = agent_policies[opponent].get_elo_rating()
                        change = compute_elo_rating_changes(agent_rating, opponent_rating, result)

                        if should_update_rating[agent]:
                            agent_policies[agent].increment_elo_rating(change)
                        if should_update_rating[opponent]:
                            agent_policies[opponent].increment_elo_rating(-change)

                        processed_agents.append(agent)
                        processed_agents.append(opponent)
                        