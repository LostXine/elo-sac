import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy
import setproctitle

import utils
from logger import Logger
from video import VideoRecorder

from elo_sac import EloSacAgent


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--pre_transform_image_size', default=100, type=int) # image render size from DMC

    parser.add_argument('--image_size', default=84, type=int) # input image size of CNN
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='elo_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=25000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=12500, type=int)
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) 
    parser.add_argument('--critic_target_update_freq', default=2, type=int) 
    parser.add_argument('--cpc_update_freq', default=1, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    # aug
    parser.add_argument('--pre_image_size', default=100, type=int)
    parser.add_argument('--rl_pre_image_size', default=100, type=int)

    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d_%d.mp4' % (step, i))
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)
    return all_ep_rewards


def make_agent(obs_shape, action_shape, args, device, weights):
    if args.agent == 'elo_sac':
        return EloSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            cpc_update_freq=args.cpc_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            weights=weights,
        )
    else:
        assert 'agent is not supported: %s' % args.agent

def main(weights, tid=0, aug=None, rl_aug=None, seed=None, task_str=None):
    args = parse_args()

    if seed is not None:
        args.seed = seed

    if aug is not None:
        args.pre_image_size = int(aug)

    if rl_aug is not None:
        args.rl_pre_image_size = int(rl_aug)

    replay_buffer_image_size = max(args.pre_image_size, args.rl_pre_image_size)

    if args.seed == -1:
        args.seed = np.random.randint(1,1000000)

    print('Raw, Tgt, Oln Image size:', args.pre_transform_image_size, args.pre_image_size, args.rl_pre_image_size)

    utils.set_seed_everywhere(args.seed)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat
    )
    print('Env shape:', env.reset().shape)
    env.seed(args.seed)
    # ref: https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
    env.action_space.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    # make directory
    # ts = time.gmtime()
    # ts = time.strftime("%m_%d", ts)
    env_name = f'{args.domain_name}-{args.task_name}'

    process_title = 'ELo-SAC-Gen2'

    setproctitle.setproctitle(f"python {process_title}-{env_name}-{tid}-s{args.seed}")
    
    
    weights_str = '_'.join([f"{w:.2f}" for w in weights])

    # exp_name = f'{env_name}-{ts}-im-{args.image_size:d}-b{args.batch_size}-s{args.seed}'
    exp_name = f'{env_name}-{tid:04d}-w{weights_str}-a{args.pre_image_size:d}-arl{args.rl_pre_image_size:d}-b{args.batch_size}-s{args.seed}'

    args.work_dir = f'{args.work_dir}/{process_title}-{exp_name}'
    print("Work dir: ", args.work_dir)
    utils.make_dir(args.work_dir)

    if args.save_video:
        video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    if args.save_model:
        model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    if args.save_buffer:
        buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if task_str is not None:
        with open(os.path.join(args.work_dir, 'task.json'), 'w') as f:
            f.write(task_str)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack, replay_buffer_image_size, replay_buffer_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=min(args.replay_buffer_capacity, args.num_train_steps),
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        rl_pre_image_size=args.rl_pre_image_size,
        pre_image_size=args.pre_image_size,
    )

    # convert log-scale weights
    weights = [ 10 ** i if i > -4 else 0 for i in weights]
    
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device,
        weights=weights,
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically
        
        if step % args.eval_freq == 0 and step > 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
            if args.save_model:
                agent.save(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)
        
        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)
                L.dump(step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        
        # print('obs shape:', obs.shape)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1

    # evaluate and save results
    res = evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
    if args.save_model:
        agent.save(model_dir, step)
    if args.save_buffer:
        replay_buffer.save(buffer_dir)
    return res


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    num_of_losses = 6
    main([0] * num_of_losses)
