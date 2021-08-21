function make_registry_env(name; time_scale=20.0, no_graphics=false, worker_id=0)
    engine_channel = EngineConfigurationChannel()
    env_params = EnvironmentParametersChannel()
    stats_channel = StatsSideChannel()
    channels = (engine=engine_channel, params=env_params, stats=stats_channel)
    env = get(default_registry, name).make(side_channels=[engine_channel,env_params,stats_channel], no_graphics=no_graphics, worker_id=worker_id)
    engine_channel.set_configuration_parameters(time_scale = time_scale)

    env.reset()
    if length(env.behavior_specs) == 1
        env = SingleBehaviorUnityEnvironment(env, channels)
    else
        env = UnityEnvironment(env, channels)
    end
    return env
end

function make_basic(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("Basic", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_3dball(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("3DBall", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_3dballhard(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("3DBallHard", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_gridworld(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("GridWorld", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_hallway(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("Hallway", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_hallway_visual(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("VisualHallway", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_crawler_dynamic(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("CrawlerDynamicTarget", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_crawler_static(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("CrawlerStaticTarget", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_bouncer(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("Bouncer", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_soccertwos(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("SoccerTwos", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_pushblock(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("PushBlock", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_pushblock_visual(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("VisualPushBlock", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_walljump(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("WallJump", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_tennis(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("Tennis", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_reacher(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("Reacher", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_pyramids(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("Pyramids", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_pyramids_visual(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("VisualPyramids", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_walker(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("Walker", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_foodcollector(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("FoodCollector", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_foodcollector_visual(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("VisualFoodCollector", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_strikervsgoalie(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("StrikerVsGoalie", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_worm_static(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("WormStaticTarget", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

function make_worm_dynamic(;time_scale=20.0, no_graphics=false, worker_id=0)
    return make_registry_env("WormDynamicTarget", time_scale=time_scale, no_graphics=no_graphics, worker_id=worker_id)
end

