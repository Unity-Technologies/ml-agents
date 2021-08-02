struct DecisionSteps{TS,TR,TI,AM} <: Any where {TS,TR,TI,AM}
    obs::TS
    reward::TR
    agent_id::TI
    action_mask::AM
    group_id::PyVector{Int}
    group_reward::TR
    aid_to_index::PyDict{Int,Int,true}
    
    function DecisionSteps(ds::PyObject)
        n = length(ds."obs")
        if n > 1
            obs = PyVector{PyArray}(ds."obs")
        else
            obs = get(ds."obs", PyArray, 0)
        end
        rews = PyArray(ds."reward")
        aid = PyArray(ds."agent_id")
        am = nothing
        gid = PyVector{Int}(ds."group_id")
        grew = PyArray(ds."group_reward")
        a2i = PyDict{Int, Int}(ds."agent_id_to_index")
        new{typeof(obs), typeof(rews), typeof(aid), typeof(am)}(obs, rews, aid, am, gid, grew, a2i)
    end
end

struct DecisionStep{TS,TR} <: Any where {TS,TR}
    obs::TS
    reward::TR
    agent_id::Int
    am::Nothing
    group_id::Int
    group_reward::TR

    function DecisionStep(ds::DecisionSteps{PyVector}, idx)
        n = length(ds.obs)
        obs = [ds.obs[i][idx, :] for i in 1:n]
        reward = ds.reward[idx]
        aid = ds.agent_id[idx]
        am = nothing
        group_id = ds.group_id[idx]
        group_reward = ds.group_reward[idx]
        new{typeof(obs), typeof(reward)}(obs, reward, aid, am, group_id, group_reward)
    end

    function DecisionStep(ds::DecisionSteps{PyArray{T,2}}, idx) where {T}
        obs = ds.obs[idx, :]
        reward = ds.reward[idx]
        aid = ds.agent_id[idx]
        am = nothing
        group_id = ds.group_id[idx]
        group_reward = ds.group_reward[idx]
        new{typeof(obs), typeof(reward)}(obs, reward, aid, am, group_id, group_reward)
    end

    function DecisionStep(obs, reward, agent_id, am, group_id, group_reward)
        new{typeof(obs), typeof(reward)}(obs, reward, agent_id, am, group_id, group_reward)
    end
end

struct TerminalSteps{TS,TR,TB,TI} <: Any where {TS,TR,TB,TI}
    obs::TS
    reward::TR
    interrupted::TB
    agent_id::TI
    group_id::PyVector{Int}
    group_reward::TR
    aid_to_index::PyDict{Int,Int,true}
    
    function TerminalSteps(ds::PyObject)
        n = length(ds."obs")
        if n > 1
            obs = PyVector{PyArray}(ds."obs")
        else
            obs = get(ds."obs", PyArray, 0)
        end
        rews = PyArray(ds."reward")
        interrupted = PyArray(ds."interrupted")
        aid = PyArray(ds."agent_id")
        gid = PyVector{Int}(ds."group_id")
        grew = PyArray(ds."group_reward")
        a2i = PyDict{Int, Int}(ds."agent_id_to_index")
        new{typeof(obs), typeof(rews), typeof(interrupted), typeof(aid)}(obs, rews, interrupted, aid, gid, grew, a2i)
    end
end

struct TerminalStep{TS,TR} <: Any where {TS,TR}
    obs::TS
    reward::TR
    interrupted::Bool
    agent_id::Int
    group_id::Int
    group_reward::TR

    function TerminalStep(ds::TerminalSteps{PyVector}, idx)
        n = length(ds.obs)
        obs = [ds.obs[i][idx, :] for i in 1:n]
        reward = ds.reward[idx]
        interrupted = ds.interrupted[idx]
        aid = ds.agent_id[idx]
        group_id = ds.group_id[idx]
        group_reward = ds.group_reward[idx]
        new{typeof(obs), typeof(reward)}(obs, reward, interrupted, aid, group_id, group_reward)
    end

    function TerminalStep(ds::TerminalSteps{PyArray{T,2}}, idx) where {T}
        obs = ds.obs[idx, :]
        reward = ds.reward[idx]
        interrupted = ds.interrupted[idx]
        aid = ds.agent_id[idx]
        group_id = ds.group_id[idx]
        group_reward = ds.group_reward[idx]
        new{typeof(obs), typeof(reward)}(obs, reward, interrupted, aid, group_id, group_reward)
    end

    function TerminalStep(obs, reward, interrupted, agent_id, group_id, group_reward)
        new{typeof(obs), typeof(reward)}(obs, reward, interrupted, agent_id, group_id, group_reward)
    end
end


function make_all_decision_steps(ds)
    ds2 = DecisionSteps(ds)
    # return [DecisionStep(ds2, ds2.aid_to_index[k]+1) for k in keys(ds2.aid_to_index)]
    return [DecisionStep(ds2, i) for i in 1:length(ds2.reward)]
end

function make_all_terminal_steps(ds)
    ds2 = TerminalSteps(ds)
    # return [TerminalStep(ds2, ds2.aid_to_index[k]+1) for k in keys(ds2.aid_to_index)]
    return [TerminalStep(ds2, i) for i in 1:length(ds2.reward)]
end

abstract type AbstractUnityEnvironment <: Any end 

struct UnityEnvironment <: AbstractUnityEnvironment
    pyenv::PyObject
    behavior_names::Vector{String}
    side_channels::NamedTuple
    function UnityEnvironment(env::PyObject, side_channels=nothing)
        behavior_names = collect(String, env.behavior_specs)
        if side_channels == nothing
            side_channels = NamedTuple()
        end
        new(env, behavior_names, side_channels)
    end
end

struct SingleBehaviorUnityEnvironment <: AbstractUnityEnvironment
    pyenv::PyObject
    behavior_name::String
    side_channels::NamedTuple

    function SingleBehaviorUnityEnvironment(env, side_channels=nothing)
        behavior_names = collect(String, env.behavior_specs)[1]
        if side_channels == nothing
            side_channels = NamedTuple()
        end
        new(env, behavior_names, side_channels)
    end
end

function make_unityenvironment(;
        file_name=nothing,
        worker_id::Int= 0,
        base_port=nothing,
        seed::Int=0,
        no_graphics::Bool=false,
        timeout_wait::Int= 60,
        additional_args=nothing,
        side_channels=nothing,
        log_folder=nothing,
    )
    if side_channels == nothing
        engine_channel = EngineConfigurationChannel()
        env_params = EnvironmentParametersChannel()
        stats_channel = StatsSideChannel()
        side_channels = (engine=engine_channel, params=env_params, stats=stats_channel)
    end
    env = PyUnityEnvironment(
        file_name=file_name, worker_id=worker_id, base_port=base_port, 
        seed=seed, no_graphics=no_graphics, timeout_wait=timeout_wait,
        additional_args=additional_args, side_channels=side_channels, log_folder=log_folder
    )
    behavior_names = collect(String, env.behavior_specs)
    println(behavior_names)

    if length(behavior_names) == 1
        return SingleBehaviorUnityEnvironment(env, side_channels)
    else
        return UnityEnvironment(env, side_channels)
    end
end

function step!(env::AbstractUnityEnvironment)
    env.pyenv.step()
end

function reset!(env::AbstractUnityEnvironment)
    env.pyenv.reset()
end

function close!(env::AbstractUnityEnvironment)
    env.pyenv.close()
end

function get_steps(env::AbstractUnityEnvironment, behavior_name)
    ds, ts = env.pyenv.get_steps(behavior_name)
    return make_all_decision_steps(ds), make_all_terminal_steps(ts)
end

function get_action_tuple(env::AbstractUnityEnvironment, behavior_name, n::Int)
    aspec = get(env.pyenv."behavior_specs", PyObject, behavior_name)."action_spec"
    atup = aspec.empty_action(n)
    return atup
end

function set_actions!(env::AbstractUnityEnvironment, behavior_name, action)
        env.pyenv.set_actions(behavior_name, action)
end

function get_steps(env::SingleBehaviorUnityEnvironment)
    ds, ts = env.pyenv.get_steps(env.behavior_name)
    return make_all_decision_steps(ds), make_all_terminal_steps(ts)
end

function get_action_tuple(env::SingleBehaviorUnityEnvironment, n::Int)
    aspec = get(env.pyenv."behavior_specs", PyObject, env.behavior_name)."action_spec"
    atup = aspec.empty_action(n)
    return atup
end

function set_actions!(env::SingleBehaviorUnityEnvironment, action)
        env.pyenv.set_actions(env.behavior_name, action)
end



