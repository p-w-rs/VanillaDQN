using PyCall
include("DQN.jl")

# Global Variables
rng = Random.default_rng()
Random.seed!(rng, 0)
gym = pyimport("gymnasium")
env = gym.make("CartPole-v1")
obs_space = env.observation_space.shape[1]
action_space = 2
epsilon_decay = 0.99995
gamma = 0.99
batch_size = 128

replay = fill_buffer(rng, env, obs_space, action_space, 10000)
model, ps, st, opt = create_model(rng, obs_space, action_space)
rewards = zeros(100000)
epsilon = 1.0
for i in 1:100000
    global epsilon, rewards
    (state, info), done = env.reset(), false
    total_reward = 0
    while !done
        action = if epsilon < rand(rng)
            q, st = model(state, ps, st)
            argmax(q)
        else
            rand(rng, 1:action_space)
        end
        (state_next, reward, terminated, truncated, info) = env.step(action - 1)
        done = terminated || truncated

        store!(replay, state, action, reward, state_next, done)
        state = state_next
        total_reward += reward
    end
    rewards[i] = total_reward
    epsilon = max(0.01, epsilon * epsilon_decay)
    experience_replay!(replay, model, ps, st, opt, gamma, batch_size)
    if i % 100 == 0
        println("Episode: $i, Epsilon: $epsilon, Total Reward: $total_reward")
    end
end
