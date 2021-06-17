import json
import os
import pickle
import time
import warnings

import numpy as np
import tensorflow as tf
import ic3net_envs

import sarnet_td3.common.tf_util as U
from sarnet_td3.common.env_setup import create_env
from experiments.config_args import parse_args
from sarnet_td3.common.action_util_td3 import ActionOPTD3, GroupActionOPTD3
#from sarnet_td3.common.action_util_vpg import ActionOPVPG
from sarnet_td3.trainer.policy_trainer import load_model, load_group_model
from sarnet_td3.common.buffer_util_td3 import BufferOp
import sarnet_td3.common.np_utils as nutil
import sarnet_td3.common.bench_util as wutil
from sarnet_td3.common.gpu_multithread import get_gputhreads, close_gputhreads

warnings.simplefilter(action='ignore', category=FutureWarning)


def reset_group(obs_n_t, group_train_act_op):

    for current_obs in obs_n_t[0]:
        for i, action in enumerate(group_train_act_op):
            if i == 0:
                action.obs_n_t.append(np.squeeze(np.asarray([current_obs[0], current_obs[2], 0, 0, 0])))
            elif i == 1:
                action.obs_n_t.append(np.squeeze(np.asarray([current_obs[1], current_obs[3], 0, 0, 0])))
            elif i == 2:
                action.obs_n_t.append(np.squeeze(np.asarray([current_obs[4], current_obs[6], current_obs[8], current_obs[10], current_obs[12]])))
            elif i == 3:
                action.obs_n_t.append(np.squeeze(np.asarray([current_obs[5], current_obs[7], current_obs[9], current_obs[11], current_obs[13]])))

def assign_group_obs(obs_n_t, group_train_act_op):

    for i, action in enumerate(group_train_act_op):
        for current_obs in obs_n_t[0]:
            if i == 0:
                action.obs_n_t.append(np.squeeze(np.asarray([current_obs[0], current_obs[2], 0, 0, 0])))
            elif i == 1:
                action.obs_n_t.append(np.squeeze(np.asarray([current_obs[1], current_obs[3], 0, 0, 0])))
            elif i == 2:
                action.obs_n_t.append(np.squeeze(np.asarray([current_obs[4], current_obs[6], current_obs[8], current_obs[10], current_obs[12]])))
            elif i == 3:
                action.obs_n_t.append(np.squeeze(np.asarray([current_obs[5], current_obs[7], current_obs[9], current_obs[11], current_obs[13]])))

def get_env_act(group_train_act_op, train_act_op, args):
    obs_n_t1 = train_act_op.obs_n_t
    rew_n_t = train_act_op.rew_n_t
    done_n_t = train_act_op.done_n_t
    info_n_t = train_act_op.info_n_t

    assign_group_obs(obs_n_t1, group_train_act_op)


def train():
    # Setup random seeds and args parameters
    args = parse_args()
    if args.benchmark or args.display:
        args.random_seed = int(time.time())
        args.memory_dropout = 1.0
        args.read_dropout = 1.0
        args.write_dropout = 1.0
        args.output_dropout = 1.0
    np.random.seed(args.random_seed)
    tf.compat.v1.set_random_seed(args.random_seed)

    """" 
    --------------------------------------------------------------------------- 
    Set experiment directory structure and files to read/write data to
    ---------------------------------------------------------------------------
    """

    # exp_name, exp_itr, tboard_dir, data_file = nutil.create_dir(args)
    is_bench_dis = args.benchmark or args.display
    is_train = not is_bench_dis

    """" 
    --------------------------------------------------------------------------- 
    Create the number of environments, num_env == 1 for benchmark and display
    ---------------------------------------------------------------------------
    """
    cpu_proc_envs, num_env, num_agents, num_adversaries, obs_shape_n, action_space, group_shape_n, group_space_output  = create_env(args)
    args.num_gpu_threads = int(num_agents + 1)

    """" 
    --------------------------------------------------------------------------- 
    Load/Create Model
    ---------------------------------------------------------------------------
    """

    trainers, sess = load_model(num_agents, obs_shape_n, action_space, args, num_env, is_train)
    group_trainers = []

    for i in range(0, args.number_group):
        g_trainers = load_group_model(num_agents, i, group_shape_n, group_space_output, args, num_env, is_train)
        group_trainers.append(g_trainers)

    # Initialize a replay buffer
    # TODO need to add replayBuffer for group trainer
    buffer_op = BufferOp(args, num_agents)
    group_buffer_op = []
    for i in range(0, args.number_group):
        g_buffer_op = BufferOp(args, num_agents)
        group_buffer_op.append(g_buffer_op)

    # Get GPU Trainer Threads
    gpu_threads_train = get_gputhreads(trainers, args, buffer_op, num_env, num_agents, num_adversaries)

    group_gpu_threads_train = []
    for i in range(0, args.number_group):
        g_gpu_threads_train = get_gputhreads(group_trainers[i], args, group_buffer_op[i], num_env, num_agents, num_adversaries)
        group_gpu_threads_train.append(g_gpu_threads_train)

    # Initialize action/train calls
    train_act_op = ActionOPTD3(trainers, args, num_env, num_agents, cpu_proc_envs, gpu_threads_train, is_train)

    group_train_act_op = []
    for i in range(0, args.number_group):
        g_train_act_op = GroupActionOPTD3(group_trainers[i], args, num_env, num_agents, cpu_proc_envs, group_gpu_threads_train[i], is_train, i)
        group_train_act_op.append(g_train_act_op)

    U.initialize()

    # Load previous results, if necessary
#    if args.load_dir == "":
#        args.load_dir = os.path.join('./exp_data/' + train_act_op.exp_name + '/' + train_act_op.exp_itr + args.save_dir + args.policy_file)
#    if args.display or args.restore or args.benchmark:
#        print('Loading previous state...')
#        U.load_state(args.load_dir)

    """" 
    --------------------------------------------------------------------------- 
    Initialize environment and reward data structures
    ---------------------------------------------------------------------------
    """
    # Initialize training parameters
    saver = tf.compat.v1.train.Saver()

    print('Starting iterations...')
    main_run_time = time.time()
    # print([x.name for x in tf.global_variables()])

    # CPU: Reset all environments and initialize all hidden states
    train_act_op.reset_states()
    for i in range(0, args.number_group):
        group_train_act_op[i].reset_states(i)

    reset_group(train_act_op.obs_n_t, group_train_act_op)


    # TODO reset the status need to seperate the obs_n and assign to each of g_train_act_op based on the logic of queue_recv_actor
    start_time = time.time()
    while True:
        """ 
        Perform following steps: 
        1. Queue and receive action from GPU session
        2. Queue critic hidden states to GPU session
        3. Queue and receive environment steps
        4. Receive critic hidden state from GPU
        5. Queue (and move on) buffer additions
        6. Prepare updated hidden states for next step (non multi thread)
        
        """
        # GPU: Queue and wait for all actions
        # Stores actions in self.action_n_t

        train_act_op.queue_recv_actor()

        for i in range(0, args.number_group):
            group_train_act_op[i].queue_recv_actor()

        # GPU: Queue for all critic states
        if args.policy_grad == "maddpg":
            train_act_op.queue_critic()
            for i in range(0, args.number_group):
                group_train_act_op[i].queue_critic()
        # Stores new observation, reward, done and benchmark
        train_act_op.get_env_act()
        if args.display:
            train_act_op.display_env()
        # Get all critic states and store in self.q1/2_h_n_t1 for next step for ddpg updates
        if args.policy_grad == "maddpg":
            train_act_op.recv_critic()
        # Queue values to be saved into the buffer, also computes done status before feeding data
        update_status = train_act_op.save_buffer()
        # Queue rewards to be saved
        train_act_op.save_rew_info()
        # Prepare inputs for next step
        train_act_op.update_states()

        # Update the actors and critics by sampling from the buffer, also write to tensor board
        if not (args.benchmark or args.display):
            if update_status:
                train_act_op.get_loss()

        if train_act_op.terminal:
            # eps_completed = train_act_op.train_step * num_env / args.max_episode_len
            if not (args.benchmark or args.display):
                train_act_op.save_model_rew_disk(saver, time.time() - start_time)
            else:
                done_bench = train_act_op.save_benchmark()
                if done_bench:
                    print("Finished Benchmarking")
                    cpu_proc_envs.cancel()
                    close_gputhreads(gpu_threads_train)
                    tf.compat.v1.InteractiveSession.close(sess)
                    time.sleep(60)
                    break
            train_act_op.reset_rew_info()
            start_time = time.time()

        # saves final episode reward for plotting training curve later
        if args.policy_grad == "maddpg":
            if not args.benchmark:
                if train_act_op.train_step * num_env > args.num_total_frames:
                    eps_completed = train_act_op.train_step * num_env / args.max_episode_len
                    wutil.write_runtime(train_act_op.data_file, eps_completed, main_run_time)
                    cpu_proc_envs.cancel()
                    close_gputhreads(gpu_threads_train)
                    tf.compat.v1.InteractiveSession.close(sess)
                    time.sleep(60)
                    break

        elif args.policy_grad == "reinforce":
            if not args.benchmark:
                if train_act_op.train_step * num_env / args.max_episode_len > args.num_episodes:
                    eps_completed = train_act_op.train_step * num_env / args.max_episode_len
                    wutil.write_runtime(train_act_op.data_file, eps_completed, main_run_time)
                    cpu_proc_envs.cancel()
                    close_gputhreads(gpu_threads_train)
                    tf.compat.v1.InteractiveSession.close(sess)
                    time.sleep(60)

if __name__ == '__main__':
    train()