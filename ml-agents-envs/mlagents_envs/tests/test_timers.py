from unittest import mock

from mlagents_envs import timers


@timers.timed
def decorated_func(x: int = 0, y: float = 1.0) -> str:
    timers.set_gauge("my_gauge", x + y)
    return f"{x} + {y} = {x + y}"


def test_timers() -> None:
    with mock.patch(
        "mlagents_envs.timers._global_timer_stack", new_callable=timers.TimerStack
    ) as test_timer:
        # First, run some simple code
        with timers.hierarchical_timer("top_level"):
            for i in range(3):
                with timers.hierarchical_timer("multiple"):
                    decorated_func(i, i)

            raised = False
            try:
                with timers.hierarchical_timer("raises"):
                    raise RuntimeError("timeout!")
            except RuntimeError:
                raised = True

            with timers.hierarchical_timer("post_raise"):
                assert raised
                pass

        # We expect the hierarchy to look like
        #   (root)
        #       top_level
        #           multiple
        #               decorated_func
        #           raises
        #           post_raise
        root = test_timer.root
        assert root.children.keys() == {"top_level"}

        top_level = root.children["top_level"]
        assert top_level.children.keys() == {"multiple", "raises", "post_raise"}

        # make sure the scope was closed properly when the exception was raised
        raises = top_level.children["raises"]
        assert raises.count == 1

        multiple = top_level.children["multiple"]
        assert multiple.count == 3

        timer_tree = test_timer.get_timing_tree()

        expected_tree = {
            "name": "root",
            "total": mock.ANY,
            "count": 1,
            "self": mock.ANY,
            "children": [
                {
                    "name": "top_level",
                    "total": mock.ANY,
                    "count": 1,
                    "self": mock.ANY,
                    "children": [
                        {
                            "name": "multiple",
                            "total": mock.ANY,
                            "count": 3,
                            "self": mock.ANY,
                            "children": [
                                {
                                    "name": "decorated_func",
                                    "total": mock.ANY,
                                    "count": 3,
                                    "self": mock.ANY,
                                }
                            ],
                        },
                        {
                            "name": "raises",
                            "total": mock.ANY,
                            "count": 1,
                            "self": mock.ANY,
                        },
                        {
                            "name": "post_raise",
                            "total": mock.ANY,
                            "count": 1,
                            "self": mock.ANY,
                        },
                    ],
                }
            ],
            "gauges": [
                {"name": "my_gauge", "value": 4.0, "max": 4.0, "min": 0.0, "count": 3}
            ],
        }

        assert timer_tree == expected_tree
