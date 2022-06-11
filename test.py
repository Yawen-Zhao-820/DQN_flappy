from train import get_init_state, resize_convert
from game import wrapped_flappy_bird as bird
from model import DQN
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
sys.path.append('game/')


def play_under_policy(state, env, model):
    action = model.policy(state)
    print(action)
    next_img, reward, _, _ = env.frame_step(action)
    next_img = resize_convert(next_img)
    next_img = np.expand_dims(next_img, axis=-1)
    next_state = np.append(next_img, state[:, :, :3], axis=2)
    return next_state, action, reward


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
            action, q_max = model.policy(state)
            next_img, reward, terminate, score = env.frame_step(action)
            next_img = resize_convert(next_img)
            next_img = np.expand_dims(next_img, axis=-1)
            next_state = np.append(next_img, state[:, :, :3], axis=2)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    avg_reward = total_reward / num_episodes
    return avg_reward


def test(path):
    model = DQN(gamma=0.99)
    env = bird.GameState()

    ckpt_dir = os.path.join('./checkpoints', path)
    # ckpt_dir = './checkpoints/bird_4'
    ckpt = tf.train.Checkpoint(
        t=tf.Variable(0), model=model
    )
    ckpt_name = 'bird-dqn'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, 5,
                                              checkpoint_name=ckpt_name)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print(f'Restored from {ckpt_manager.latest_checkpoint}')
    else:
        print('Initializing from scratch.')

    init_state = get_init_state(env)

    state = init_state

    while 'yawen' != 'bird':
        # next_state, _, _ = play_under_policy(state, env, model)
        # state = next_state
        avg_reward = eval_performance(env, model, num_episodes=10)
        print(f'So far the average reward per 10 episodes is {avg_reward}.')


def main():
    parser = argparse.ArgumentParser(description='Ya - DQN')
    parser.add_argument(
        '-p', '--path', help='The path of check point', required=True)
    args = parser.parse_args()
    test(args.path)
    # test()


if __name__ == '__main__':
    main()
