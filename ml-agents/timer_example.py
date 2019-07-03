import time
from mlagents.trainers.trainer_metrics import hierarchical_timer, _global_timer_stack
import json

# root = TimerNode("root")
#
# with root:
#     time.sleep(.1)
#
# print(root)

with hierarchical_timer("foo"):
    time.sleep(0.1)
    for i in range(3):
        with hierarchical_timer("bar"):
            time.sleep(0.05)
    time.sleep(0.1)

print(json.dumps(_global_timer_stack.get_timing_tree(), indent=2))
