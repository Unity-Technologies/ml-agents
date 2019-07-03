import time
from mlagents.trainers.trainer_metrics import hierarchical_timer, _global_timer_stack
import json


with hierarchical_timer("foo"):
    time.sleep(0.1)
    for i in range(3):
        with hierarchical_timer("bar"):
            time.sleep(0.05)

    with hierarchical_timer("raises"):
        raised = False
        try:
            time.sleep(0.05)
            raise RuntimeError("timeout!")
        except RuntimeError:
            raised = True
            print("Got exception as expected.")

    with hierarchical_timer("post_raise"):
        assert raised
        time.sleep(0.1)

print(json.dumps(_global_timer_stack.get_timing_tree(), indent=2))
