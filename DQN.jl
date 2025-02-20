using Lux, ComponentArrays, Random, Optimisers, Zygote, ExperienceReplay

function fill_buffer(rng, env, obs_space, action_space, n)
    replay = Buffer(obs_space, action_space, n)
    while replay.len < n
        (observation, info), done = env.reset(), false
        while !done
            action = rand(rng, 1:action_space)
            observation_next, reward, terminated, truncated, info = env.step(action - 1)
            done = terminated || truncated
            store!(replay, observation, action, reward, observation_next, done)
            observation = observation_next
        end
    end
    return replay
end

function create_model(rng, obs_space, act_space)
    model = Chain(
        Dense(obs_space => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => act_space)
    )
    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps)
    opt = Optimisers.setup(Optimisers.Adam(0.001f0), ps)
    return model, ps, st, opt
end

function experience_replay!(replay, model, ps, st, opt, gamma, n)
    s, a, r, sn, t = get_batch!(replay, n)
    grads = Zygote.gradient(ps, st) do ps, st
        q, _ = model(s, ps, st)
        q = sum(q .* a, dims=1)

        qn, _ = model(sn, ps, st)
        qn = maximum(qn, dims=1)

        q_t = r .+ gamma .* qn .* t
        loss = sum((q .- q_t).^2) / n
    end
    opt, ps = Optimisers.update(opt, ps, grads[1])
end
