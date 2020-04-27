# Training with Self-Play


## Training Statistics

To view training statistics, use TensorBoard. For information on launching and
using TensorBoard, see
[here](./Getting-Started.md#observing-training-progress).

### ELO
In adversarial games, the cumulative environment reward may not be a meaningful metric by which to track learning progress.  This is because cumulative reward is entirely dependent on the skill of the opponent. An agent at a particular skill level will get more or less reward against a worse or better agent, respectively.

We provide an implementation of the ELO rating system, a method for calculating the relative skill level between two players from a given population in a zero-sum game. For more information on ELO, please see [the ELO wiki](https://en.wikipedia.org/wiki/Elo_rating_system).
In a proper training run, the ELO of the agent should steadily increase. The absolute value of the ELO is less important than the change in ELO over training iterations.

Note, this implementation will support any number of teams but ELO is only applicable to games with two teams.  It is ongoing work to implement
a reliable metric for measuring progress in scenarios with three or more teams. These scenarios can still train, though as of now, reward and qualitative observations
are the only metric by which we can judge performance.
