from typing import Deque
from collections import deque


class GhostController(object):
    def __init__(self, swap_interval: int, maxlen: int = 10):
        self._swap_interval = swap_interval
        self._last_swap: int = 0
        self._queue: Deque[int] = deque(maxlen=maxlen)
        self._learning_team: int = 0

    def subscribe_team_id(self, team_id: int) -> None:
        self._queue.append(team_id)

    def get_learning_id(self, step: int) -> int:
        if step >= self._swap_interval + self._last_swap:
            self._last_swap = step
            self.subscribe_team_id(self._learning_team)
            self._learning_team = self._queue.popleft()
        return self._learning_team


# Taken from https://github.com/Unity-Technologies/ml-agents/pull/1975 and
# https://metinmediamath.wordpress.com/2013/11/27/how-to-calculate-the-elo-rating-including-example/
# ELO calculation


def compute_elo_rating_changes(rating1: float, rating2: float, result: float) -> float:
    r1 = pow(10, rating1 / 400)
    r2 = pow(10, rating2 / 400)

    summed = r1 + r2
    e1 = r1 / summed

    change = result - e1
    return change
