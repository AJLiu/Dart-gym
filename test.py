from envs.cartpole import DartCartpoleEnv
import itertools

env = DartCartpoleEnv()

for _ in range(10):
    env.reset()
    done = False
    for i in itertools.count():
        a = env.action_space.sample()
        state, reward, done, _ = env.step(a)
        env.render()
        if done:
            print(i)
            break