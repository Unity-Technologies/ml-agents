from mlagents_envs.registry import default_registry
from pettingzoo_unity import UnityToPettingZooWrapper
from typing import Optional
from mlagents_envs.exception import UnityWorkerInUseException

class petting_zoo_env:
    def __init__(self, env_id):
        self.env_id = env_id
    def env(self, seed: Optional[int] = None, **kwargs):
        def make_env(seed: Optional[int] = None, **kwargs):  # some args here
            _env = None
            if "base_port" not in kwargs:
                port = 6000
                while _env is None:
                    print(f"*** increasing port: {port}")
                    try:
                        kwargs["base_port"] = port
                        _env = default_registry[self.env_id].make(**kwargs)
                    except UnityWorkerInUseException:
                        port += 1
                        pass
            else:
                print(kwargs["base_port"])
                _env = default_registry[self.env_id].make(**kwargs)
            return UnityToPettingZooWrapper(_env, seed)

        return make_env()


for key in default_registry:
    #todo: proper value name 3dball not accepted
    locals()[key] = petting_zoo_env(key)
