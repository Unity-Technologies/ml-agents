# __precompile__() # this module is safe to precompile
module MLAgentsEnvs

using Base: NamedTuple
using Conda
using PyCall


const mla_envs = PyNULL()
const mla_envs_environment = PyNULL()
const mla_envs_registry = PyNULL()
const default_registry = PyNULL()

# Side Channel objects
const SideChannel = PyNULL()
const OutgoingMessage = PyNULL()
const IncomingMessage = PyNULL()
const SideChannelManager = PyNULL()
const RawBytesChannel = PyNULL()
const FloatPropertiesChannel = PyNULL()
const EngineConfigurationChannel = PyNULL()
const EngineConfig = PyNULL()
const EnvironmentParametersChannel = PyNULL()
const StatsAggregationMethod = PyNULL()
const StatsSideChannel = PyNULL()
const ActionTuple = PyNULL()
const PyUnityEnvironment = PyNULL()

export mla_envs, mla_envs_environment, mla_envs_registry, default_registry
export SideChannel, OutgoingMessage, IncomingMessage, SideChannelManager, RawBytesChannel, FloatPropertiesChannel
export EngineConfig, EngineConfigurationChannel, EnvironmentParametersChannel, StatsAggregationMethod, StatsSideChannel
export PyUnityEnvironment

export ActionTuple
export AbstractUnityEnvironment, UnityEnvironment, SingleBehaviorUnityEnvironment
export reset!, step!, get_steps, set_actions!, close!, get_action_tuple
export make_unityenvironment
include("environment.jl")

export make_basic, make_3dball, make_3dballhard, make_gridworld, make_hallway, make_hallway_visual, make_crawler_dynamic, make_crawler_static
export make_bouncer, make_soccertwos, make_pushblock, make_pushblock_visual, make_walljump, make_tennis, make_reacher, make_pyramids
export make_pyramids_visual, make_walker, make_foodcollector, make_foodcollector_visual, make_strikervsgoalie, make_worm_static, make_worm_dynamic

include("default_envs.jl")

"""
Module initialization function
"""
function __init__()
    # Conda.pip_interop(true)
    # Conda.pip("install", "mlagents-envs")

    copy!(mla_envs, pyimport("mlagents_envs"))
    copy!(mla_envs_environment, pyimport("mlagents_envs.environment"))
    copy!(mla_envs_registry, pyimport("mlagents_envs.registry"))
    copy!(default_registry, mla_envs_registry.default_registry)

    sc = pyimport("mlagents_envs.side_channel")
    copy!(SideChannel, sc.SideChannel)
    copy!(OutgoingMessage, sc.OutgoingMessage)
    copy!(IncomingMessage, sc.IncomingMessage)
    mg = pyimport("mlagents_envs.side_channel.side_channel_manager")
    copy!(SideChannelManager, mg.SideChannelManager)
    rb = pyimport("mlagents_envs.side_channel.raw_bytes_channel")
    copy!(RawBytesChannel, rb.RawBytesChannel)
    fp = pyimport("mlagents_envs.side_channel.float_properties_channel")
    copy!(FloatPropertiesChannel, fp.FloatPropertiesChannel)
    ec = pyimport("mlagents_envs.side_channel.engine_configuration_channel")
    copy!(EngineConfig, ec.EngineConfig)
    copy!(EngineConfigurationChannel, ec.EngineConfigurationChannel)
    ep = pyimport("mlagents_envs.side_channel.environment_parameters_channel")
    copy!(EnvironmentParametersChannel, ep.EnvironmentParametersChannel)
    ss = pyimport("mlagents_envs.side_channel.stats_side_channel")
    copy!(StatsAggregationMethod, ss.StatsAggregationMethod)
    copy!(StatsSideChannel, ss.StatsSideChannel)

    benv = pyimport("mlagents_envs.base_env")
    copy!(ActionTuple, benv.ActionTuple)
    env = pyimport("mlagents_envs.environment")
    copy!(PyUnityEnvironment, env.UnityEnvironment)

end


end
