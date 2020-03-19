import logging

from typing import Deque, Dict
from collections import deque
from mlagents.trainers.ghost.trainer import GhostTrainer

logger = logging.getLogger("mlagents.trainers")


class GhostController(object):
    def __init__(self, swap_interval: int, maxlen: int = 10):
        self._swap_interval = swap_interval
        self._last_swap: int = 0
        self._queue: Deque[int] = deque(maxlen=maxlen)
        self._learning_team: int = -1
        self._ghost_trainers: Dict[int, GhostTrainer] = {}

    def subscribe_team_id(self, team_id: int, trainer: GhostTrainer) -> None:
        if team_id not in self._ghost_trainers:
            self._ghost_trainers[team_id] = trainer
            if self._learning_team < 0:
                self._learning_team = team_id
            else:
                self._queue.append(team_id)

    def get_learning_team(self, step: int) -> int:
        if step >= self._swap_interval + self._last_swap:
            self._last_swap = step
            self._queue.append(self._learning_team)
            self._learning_team = self._queue.popleft()
            logger.debug(
                "Learning team {} swapped on step {}".format(
                    self._learning_team, self._last_swap
                )
            )
        return self._learning_team

    # Adapted from https://github.com/Unity-Technologies/ml-agents/pull/1975 and
    # https://metinmediamath.wordpress.com/2013/11/27/how-to-calculate-the-elo-rating-including-example/
    # ELO calculation

    def compute_elo_rating_changes(self, rating: float, result: float) -> float:
        opponent_rating: float = 0.0
        for team_id, trainer in self._ghost_trainers.items():
            if team_id != self._learning_team:
                opponent_rating = trainer.get_opponent_elo()
        r1 = pow(10, rating / 400)
        r2 = pow(10, opponent_rating / 400)

        summed = r1 + r2
        e1 = r1 / summed

        change = result - e1
        for team_id, trainer in self._ghost_trainers.items():
            if team_id != self._learning_team:
                trainer.change_opponent_elo(change)

        return change
