#import gymnasium as gym
import gym
import numpy as np
import matplotlib.pyplot as plt

DIGIT = 4

#env = gym.make('CartPole-v1')
env = gym.make('CartPole-v1',render_mode='human')

#___________________________________
#___________________________________
q_table = np.random.uniform(low=-1, high=1, size=(DIGIT ** DIGIT, env.action_space.n))


#___________________________________
#___________________________________
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]



#___________________________________
#___________________________________
def digitize_state(observation):
    # 各値を4個の離散値に変換
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, DIGIT)),
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, DIGIT)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, DIGIT)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, DIGIT))]
    # 0~255に変換
    return sum([x * (DIGIT ** i) for i, x in enumerate(digitized)])

#___________________________________
#___________________________________
goal_average_steps = 195
max_number_of_steps = 200
num_consecutive_iterations = 100
num_episodes = 300
#num_episodes = 200
#um_episodes = 10
last_time_steps = np.zeros(num_consecutive_iterations)
step_list = []
frames = []



#___________________________________
#___________________________________
def get_action(state, action, observation, reward, episode):
    next_state = digitize_state(observation)
    next_action = np.argmax(q_table[next_state])


    # Qテーブルの更新
    alpha = 0.2
    gamma = 0.99
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * q_table[next_state, next_action])

    return next_action, next_state

#___________________________________
#___________________________________
for episode in range(num_episodes):
    # 環境の初期化
    observation = env.reset()

    state = digitize_state(observation)
    action = np.argmax(q_table[state])

    #print('observation, reward, done, info')
    episode_reward = 0
    for t in range(max_number_of_steps):
        # CartPoleの描画
        #env.render()

        # 行動の実行とフィードバックの取得
        observation, reward, done, info = env.step(action)

        #print(observation, reward, done, info)
       
        #print(action, state)

        if done:
            if t < 195:
                reward = -200
            else:
                reward = 1
        else:
            reward = 1
        
        episode_reward += reward

 # 行動の選択
        action, state = get_action(state, action, observation, reward, episode)

        if done or t>=199:
            print('%d Episode finished after %d time steps / mean %f' % (episode, t + 1,
                last_time_steps.mean()))
            step_list.append(t+1)

            #print('episode_reward', episode_reward)
            #print('episode_reward/step', episode_reward/(t + 1))

            last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))
            
            break

        
    if (last_time_steps.mean() >= goal_average_steps): # 直近の100エピソードが195以上であれば成功
        print('Episode %d train agent successfuly!' % episode)
        break


#___________________________________
#___________________________________
es = np.arange(0, len(step_list))
plt.plot(es, step_list)
plt.savefig("cartpole.png")
plt.figure()
