from mlagents.trainers.torch_entities.networks import SimpleActor
from mlagents.trainers.policy.torch_policy import TorchPolicy
from .network import DroneBody
from .settings import DroneTrainerSettings
from mlagents.trainers.buffer import AgentBuffer

class DroneActor(SimpleActor):
    # só substituímos o corpo pela nossa MLP
    def __init__(self, *args, **kwargs):
        observation_specs = kwargs["observation_specs"]
        action_spec       = kwargs["action_spec"]
        network_settings  = kwargs["network_settings"]
        super().__init__(observation_specs=observation_specs, network_settings=network_settings, action_spec=action_spec)

        # calcula o tamanho das observações vetoriais (ignora sensores visuais aqui)
        vector_size = sum(
            spec.shape[0] for spec in observation_specs if len(spec.shape) == 1
        )

        # troca o backbone padrão pela nossa MLP customizada
        self.network_body  = DroneBody(vector_size, network_settings)
        
        self.encoding_size = network_settings.hidden_units

class DronePolicy(TorchPolicy):
    def __init__(self, seed: int, behavior_spec, trainer_settings: DroneTrainerSettings):

        super().__init__(
            seed=seed,
            behavior_spec=behavior_spec,
            network_settings=trainer_settings.network_settings,
            actor_cls=DroneActor,
            actor_kwargs = {"conditional_sigma": False, "tanh_squash": False}
        )