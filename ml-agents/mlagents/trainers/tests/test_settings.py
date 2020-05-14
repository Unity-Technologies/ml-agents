import attr

from mlagents.trainers.settings import RunOptions, TrainerSettings


def check_if_different(testobj1: object, testobj2: object) -> None:
    assert testobj1 is not testobj2
    if attr.has(testobj1.__class__) and attr.has(testobj2.__class__):
        for key, val in attr.asdict(testobj1, recurse=False).items():
            if isinstance(val, dict) or isinstance(val, list) or attr.has(val):
                # Note: this check doesn't check the contents of mutables.
                check_if_different(val, attr.asdict(testobj2, recurse=False)[key])


def test_is_new_instance():
    """
    Verify that every instance of RunOptions() and its subclasses
    is a new instance (i.e. all factory methods are used properly.)
    """
    check_if_different(RunOptions(), RunOptions())
    check_if_different(TrainerSettings(), TrainerSettings())
