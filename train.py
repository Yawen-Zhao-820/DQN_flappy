import sys
sys.path.append('game/')

import argparse
import os
import numpy as np
import tensorflow as tf
from model import DQN
from replay_buffer import ReplayBuffer
from game import wrapped_flappy_bird as bird
import cv2

NUM_ACTIONS = 2
FINAL_EPSILON = 0.0001
BUFFER_SIZE = 50000
GAMMA = 0.99
DEVICE = 'GPU'
GAME_NAME = 'bird'
BATCH_SIZE = 32
C = 200
LOG_FREQUENCY = 1000
SAVE_FREQUENCY = 10000
####################### To Demo ###########################
# OBSERVE = 100000 # timesteps to observe before training
# EXPLORE = 2000000 # frames over which to anneal epsilon
# INITIAL_EPSILON = 0.0001 # starting value of epsilon

####################### To Train ###########################
OBSERVE = 10000 # timesteps to observe before training
EXPLORE = 2000000 # frames over which to anneal epsilon
INITIAL_EPSILON = 0.1


# def wirte_param(path):
#     with open('')

def eval_performance(env, model, num_episodes=10):
    """
    Evaluate the performance of current model by averge rewards of 
    a batch of episodes.
    """
    total_reward = 0.
    for _ in range(num_episodes):
        state = get_init_state(env)
        terminate = False
        episode_reward = 0.
        while not terminate:
            action = model.policy(state)
            next_img, reward, terminate, score = env.frame_step(action)
            next_img = resize_convert(next_img)
            next_img = np.expand_dims(next_img, axis=-1)
            next_state = np.append(next_img, state[:, :, :3], axis=2)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    avg_reward = total_reward / num_episodes
    return avg_reward

def resize_convert(img, size=(80, 80), code=cv2.COLOR_BGR2GRAY, type=cv2.THRESH_BINARY):
    """
    Helper function.

    Resize the image to size and convert colored image to gray scale.
    """
    img = cv2.cvtColor(cv2.resize(img, size), code)
    _, img = cv2.threshold(img, 1, 255, type)
    return img

def get_init_state(env):
    """
    Helper function.

    Get the initial state by doing nothing at game start.
    """
    init_action = np.zeros(NUM_ACTIONS)
    init_action[0] = 1 # action=[1, 0]: let bird do nothing
    img, _, _, _ = env.frame_step(init_action)
    img = resize_convert(img)
    state = np.stack((img, img, img, img), axis=2)
    return state

def collect_gameplay_exps(state, env, model, buffer, epsilon=INITIAL_EPSILON):
    """
    Collect and store the gameplay experiences into buffer until an episode
    is terminated.
    """
    # state = get_init_state(env)
    # terminate = False
    # while not terminate:
    # q_max = np.max(model(np.expand_dims(state, axis=0)))
    action, q_max = model.collect_policy(state, epsilon=epsilon)
    next_img, reward, terminate, score = env.frame_step(action)
    next_img = resize_convert(next_img)
    next_img = np.expand_dims(next_img, axis=-1)
    next_state = np.append(next_img, state[:, :, :3], axis=2)
    buffer.store_gameplay_exp(state, next_state, reward, action, terminate)
    return next_state, action, reward, q_max, terminate, score

def train_model(path):
    """
    Main function

    Train the model.
    """
    dirname = os.path.dirname(__file__)
    log_dir = os.path.join(dirname, 'logs', path)
    episode_file_name = os.path.join(log_dir, 'episode_data.txt')
    timestep_file_name = os.path.join(log_dir, 'timestep_data.txt')
    model = DQN(gamma=GAMMA)
    buffer = ReplayBuffer(max_len=BUFFER_SIZE)
    env = bird.GameState()

    ###
    state = get_init_state(env)
    epsilon = tf.Variable(INITIAL_EPSILON, trainable=False)
    t = tf.Variable(0, trainable=False)
    episode_count = tf.Variable(0, trainable=False)
    episode_reward = 0.
    episode_best_score = 0
    timestep_reward = 0.
    timestep_best_score = 0
    ###

    # ckpt_dir = './checkpoints/bird_5'
    ckpt_dir = os.path.join('./checkpoints', path)
    ckpt = tf.train.Checkpoint(
        t=t, 
        episode_count=episode_count,
        epsilon=epsilon,
        model=model
    )
    ckpt_name = GAME_NAME + '-dqn'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, 5,
                                              checkpoint_name=ckpt_name)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print(f'Restored from {ckpt_manager.latest_checkpoint}')
    else:
        print('Initializing from scratch.')
    

    # for episode in range(max_episode):
    while 'Duck' != 'Bird':
        # Scale down epsilon
        if t > OBSERVE and epsilon > FINAL_EPSILON:
            epsilon.assign_sub((INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE)
        # Play a step and collect gameplay exp
        state, action, reward, q_max, terminate, score = \
            collect_gameplay_exps(state, env, model, buffer, epsilon)
        t.assign_add(1)
        # Update rewards and scores
        timestep_reward += reward
        if timestep_best_score < score:
            timestep_best_score = score

        episode_reward += reward
        if episode_best_score < score:
            episode_best_score = score

        if terminate:
            episode_count.assign_add(1)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(episode_file_name, 'a') as f:
                f.write(f'{int(t)}' + ',' + \
                        f'{int(episode_count)}' + ',' + \
                        f'{episode_best_score}' + ',' + \
                        f'{episode_reward}' + '\n')
                f.close()
            episode_best_score = 0
            episode_reward = 0.
            

        # local_t += 1
        # Only train when t > OBSERVE
        # if int(ckpt.t) > OBSERVE and local_t > BUFFER_SIZE / 10:
        if t > OBSERVE and len(buffer.gameplay_exps) > 500:
            gameplay_exp_batch = buffer.get_gameplay_exp_batch(BATCH_SIZE)
            loss = model.train(gameplay_exp_batch)
            # Update target net
            if t % C == 0:
                model.update_target_net()
            # Wirte timestep log
            if t % LOG_FREQUENCY == 0:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                with open(timestep_file_name, 'a') as f:
                    f.write(f'{int(t)}' + ',' + \
                            f'{int(episode_count)}' + ',' + \
                            f'{timestep_best_score}' + ',' + \
                            f'{timestep_reward}' + '\n')
                    f.close()
                timestep_best_score = 0
                timestep_reward = 0.
            # Save the model
            if t % SAVE_FREQUENCY == 0:
                save_path = ckpt_manager.save(checkpoint_number=int(t))
                print(f'Saved checkpoint for step {int(t)}: {save_path}')
        # avg_reward = eval_performance(env, model)
        # print(f'Current performance is {avg_reward}.')
        
        # print info
        stage = ""
        if t <= OBSERVE:
            stage = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            stage = "explore"
        else:
            stage = "train"

        print("TIMESTEP", t.numpy(), "/ STATE", stage, \
            "/ EPSILON", epsilon.numpy(), "/ ACTION", np.argmax(action), "/ REWARD", reward, \
            "/ Q_MAX %e" % q_max,
            '/ Bird Name', path
            )

def main():
    parser = argparse.ArgumentParser(description='Ya - DQN')
    parser.add_argument(
        '-p', '--path', help='The path of check point', required=True)
    args = parser.parse_args()
    with tf.device(DEVICE):
        train_model(args.path)

if __name__ == '__main__':
    main()
    # train_model()
