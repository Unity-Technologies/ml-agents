import pytest
import yaml
import io
from unittest.mock import patch

import mlagents.trainers.trainer_util as trainer_util
from mlagents.trainers.trainer_util import load_config, _load_config
from mlagents.trainers.trainer_metrics import TrainerMetrics
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.exception import TrainerConfigError
from mlagents.trainers.brain import BrainParameters


@pytest.fixture
def dummy_config():
    return yaml.safe_load(
        """
        default:
            trainer: ppo
            batch_size: 32
            beta: 5.0e-3
            buffer_size: 512
            epsilon: 0.2
            gamma: 0.99
            hidden_units: 128
            lambd: 0.95
            learning_rate: 3.0e-4
            max_steps: 5.0e4
            normalize: true
            num_epoch: 5
            num_layers: 2
            time_horizon: 64
            sequence_length: 64
            summary_freq: 1000
            use_recurrent: false
            memory_size: 8
            use_curiosity: false
            curiosity_strength: 0.0
            curiosity_enc_size: 1
            reward_signals:
                extrinsic:
                    strength: 1.0
                    gamma: 0.99
        """
    )


@pytest.fixture
def dummy_config_with_override(dummy_config):
    base = dummy_config
    base["testbrain"] = {}
    base["testbrain"]["normalize"] = False
    return base


@pytest.fixture
def dummy_bad_config():
    return yaml.safe_load(
        """
        default:
            trainer: incorrect_trainer
            brain_to_imitate: ExpertBrain
            batches_per_epoch: 16
            batch_size: 32
            beta: 5.0e-3
            buffer_size: 512
            epsilon: 0.2
            gamma: 0.99
            hidden_units: 128
            lambd: 0.95
            learning_rate: 3.0e-4
            max_steps: 5.0e4
            normalize: true
            num_epoch: 5
            num_layers: 2
            time_horizon: 64
            sequence_length: 64
            summary_freq: 1000
            use_recurrent: false
            memory_size: 8
        """
    )


@patch("mlagents.trainers.brain.BrainParameters")
def test_initialize_trainer_parameters_override_defaults(
    BrainParametersMock, dummy_config_with_override
):
    summaries_dir = "test_dir"
    run_id = "testrun"
    model_path = "model_dir"
    keep_checkpoints = 1
    train_model = True
    load_model = False
    seed = 11
    expected_reward_buff_cap = 1

    base_config = dummy_config_with_override
    expected_config = base_config["default"]
    expected_config["summary_path"] = f"{run_id}_testbrain"
    expected_config["model_path"] = model_path + "/testbrain"
    expected_config["keep_checkpoints"] = keep_checkpoints

    # Override value from specific brain config
    expected_config["normalize"] = False

    brain_params_mock = BrainParametersMock()
    BrainParametersMock.return_value.brain_name = "testbrain"
    external_brains = {"testbrain": brain_params_mock}

    def mock_constructor(
        self,
        brain,
        reward_buff_cap,
        trainer_parameters,
        training,
        load,
        seed,
        run_id,
        multi_gpu,
    ):
        self.trainer_metrics = TrainerMetrics("", "")
        assert brain == brain_params_mock
        assert trainer_parameters == expected_config
        assert reward_buff_cap == expected_reward_buff_cap
        assert training == train_model
        assert load == load_model
        assert seed == seed
        assert run_id == run_id
        assert multi_gpu == multi_gpu

    with patch.object(PPOTrainer, "__init__", mock_constructor):
        trainer_factory = trainer_util.TrainerFactory(
            trainer_config=base_config,
            summaries_dir=summaries_dir,
            run_id=run_id,
            model_path=model_path,
            keep_checkpoints=keep_checkpoints,
            train_model=train_model,
            load_model=load_model,
            seed=seed,
        )
        trainers = {}
        for _, brain_parameters in external_brains.items():
            trainers["testbrain"] = trainer_factory.generate(brain_parameters)
        assert "testbrain" in trainers
        assert isinstance(trainers["testbrain"], PPOTrainer)


@patch("mlagents.trainers.brain.BrainParameters")
def test_initialize_ppo_trainer(BrainParametersMock, dummy_config):
    brain_params_mock = BrainParametersMock()
    BrainParametersMock.return_value.brain_name = "testbrain"
    external_brains = {"testbrain": BrainParametersMock()}
    summaries_dir = "test_dir"
    run_id = "testrun"
    model_path = "model_dir"
    keep_checkpoints = 1
    train_model = True
    load_model = False
    seed = 11
    expected_reward_buff_cap = 1

    base_config = dummy_config
    expected_config = base_config["default"]
    expected_config["summary_path"] = f"{run_id}_testbrain"
    expected_config["model_path"] = model_path + "/testbrain"
    expected_config["keep_checkpoints"] = keep_checkpoints

    def mock_constructor(
        self,
        brain,
        reward_buff_cap,
        trainer_parameters,
        training,
        load,
        seed,
        run_id,
        multi_gpu,
    ):
        self.trainer_metrics = TrainerMetrics("", "")
        assert brain == brain_params_mock
        assert trainer_parameters == expected_config
        assert reward_buff_cap == expected_reward_buff_cap
        assert training == train_model
        assert load == load_model
        assert seed == seed
        assert run_id == run_id
        assert multi_gpu == multi_gpu

    with patch.object(PPOTrainer, "__init__", mock_constructor):
        trainer_factory = trainer_util.TrainerFactory(
            trainer_config=base_config,
            summaries_dir=summaries_dir,
            run_id=run_id,
            model_path=model_path,
            keep_checkpoints=keep_checkpoints,
            train_model=train_model,
            load_model=load_model,
            seed=seed,
        )
        trainers = {}
        for brain_name, brain_parameters in external_brains.items():
            trainers[brain_name] = trainer_factory.generate(brain_parameters)
        assert "testbrain" in trainers
        assert isinstance(trainers["testbrain"], PPOTrainer)


@patch("mlagents.trainers.brain.BrainParameters")
def test_initialize_invalid_trainer_raises_exception(
    BrainParametersMock, dummy_bad_config
):
    summaries_dir = "test_dir"
    run_id = "testrun"
    model_path = "model_dir"
    keep_checkpoints = 1
    train_model = True
    load_model = False
    seed = 11
    bad_config = dummy_bad_config
    BrainParametersMock.return_value.brain_name = "testbrain"
    external_brains = {"testbrain": BrainParametersMock()}

    with pytest.raises(TrainerConfigError):
        trainer_factory = trainer_util.TrainerFactory(
            trainer_config=bad_config,
            summaries_dir=summaries_dir,
            run_id=run_id,
            model_path=model_path,
            keep_checkpoints=keep_checkpoints,
            train_model=train_model,
            load_model=load_model,
            seed=seed,
        )
        trainers = {}
        for brain_name, brain_parameters in external_brains.items():
            trainers[brain_name] = trainer_factory.generate(brain_parameters)


def test_handles_no_default_section(dummy_config):
    """
    Make sure the trainer setup handles a missing "default" in the config.
    """
    brain_name = "testbrain"
    no_default_config = {brain_name: dummy_config["default"]}
    brain_parameters = BrainParameters(
        brain_name=brain_name,
        vector_observation_space_size=1,
        camera_resolutions=[],
        vector_action_space_size=[2],
        vector_action_descriptions=[],
        vector_action_space_type=0,
    )

    trainer_factory = trainer_util.TrainerFactory(
        trainer_config=no_default_config,
        summaries_dir="test_dir",
        run_id="testrun",
        model_path="model_dir",
        keep_checkpoints=1,
        train_model=True,
        load_model=False,
        seed=42,
    )
    trainer_factory.generate(brain_parameters)


def test_raise_if_no_config_for_brain(dummy_config):
    """
    Make sure the trainer setup raises a friendlier exception if both "default" and the brain name
    are missing from the config.
    """
    brain_name = "testbrain"
    bad_config = {"some_other_brain": dummy_config["default"]}
    brain_parameters = BrainParameters(
        brain_name=brain_name,
        vector_observation_space_size=1,
        camera_resolutions=[],
        vector_action_space_size=[2],
        vector_action_descriptions=[],
        vector_action_space_type=0,
    )

    trainer_factory = trainer_util.TrainerFactory(
        trainer_config=bad_config,
        summaries_dir="test_dir",
        run_id="testrun",
        model_path="model_dir",
        keep_checkpoints=1,
        train_model=True,
        load_model=False,
        seed=42,
    )
    with pytest.raises(TrainerConfigError):
        trainer_factory.generate(brain_parameters)


def test_load_config_missing_file():
    with pytest.raises(TrainerConfigError):
        load_config("thisFileDefinitelyDoesNotExist.yaml")


def test_load_config_valid_yaml():
    file_contents = """
this:
  - is fine
    """
    fp = io.StringIO(file_contents)
    res = _load_config(fp)
    assert res == {"this": ["is fine"]}


def test_load_config_invalid_yaml():
    file_contents = """
you:
  - will
- not
  - parse
    """
    with pytest.raises(TrainerConfigError):
        fp = io.StringIO(file_contents)
        _load_config(fp)
