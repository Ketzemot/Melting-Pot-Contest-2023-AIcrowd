from meltingpot import substrate
from ray.rllib.policy import policy
from baselines.train import make_envs

SUPPORTED_SCENARIOS = [
    'allelopathic_harvest__open_0',
    'allelopathic_harvest__open_1',
    'allelopathic_harvest__open_2',
    'clean_up_2',
    'clean_up_3',
    'clean_up_4',
    'clean_up_5',
    'clean_up_6',
    'clean_up_7',
    'clean_up_8',
    'prisoners_dilemma_in_the_matrix__arena_0',
    'prisoners_dilemma_in_the_matrix__arena_1',
    'prisoners_dilemma_in_the_matrix__arena_2',
    'prisoners_dilemma_in_the_matrix__arena_3',
    'prisoners_dilemma_in_the_matrix__arena_4',
    'prisoners_dilemma_in_the_matrix__arena_5',
    'territory__rooms_0',
    'territory__rooms_1',
    'territory__rooms_2',
    'territory__rooms_3',
]

IGNORE_KEYS = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP']


def get_experiment_config(args, default_config):
    
    if args.exp == 'pd_arena':
        substrate_name = "prisoners_dilemma_in_the_matrix__arena"
    elif args.exp == 'al_harvest':
        substrate_name = "allelopathic_harvest__open"
    elif args.exp == 'clean_up':
        substrate_name = "clean_up"
    elif args.exp == 'territory_rooms':
        substrate_name = "territory__rooms"
    else:
        raise Exception("Please set --exp to be one of ['pd_arena', 'al_harvest', 'clean_up', \
                        'territory_rooms']. Other substrates are not supported.")

    # Fetch player roles
    player_roles = substrate.get_config(substrate_name).default_player_roles

    if args.downsample:
        scale_factor = 8
    else:
        scale_factor = 1

    params_dict = {

        # resources
        "num_rollout_workers": args.num_workers,
        "num_gpus": args.num_gpus,

        # Env
        "env_name": "meltingpot",
        "env_config": {"substrate": substrate_name, "roles": player_roles, "scaled": scale_factor},

        # training
        "seed": args.seed,
        
        # rollout_fragment_length: LSTM state is not backpropagated across rollout fragments; 
        # Value function is used to bootstrap rewards past the end of the sequence;
        # Affects system performance (too small => high overhead, too large => high delay).
        "rollout_fragment_length": 25, # 5 10 Note: 
        # "train_batch_size": 16, # 400 # best for train time: 256: 6.1s/train_iter
        # "sgd_minibatch_size": 8, # 32 # best for train time: 256: 6.1s/train_iter
        # "train_batch_size": 500, # 400 # best for train time: 256: 6.1s/train_iter # for actual training!
        # "sgd_minibatch_size": 500, # 32 # best for train time: 256: 6.1s/train_iter # for actual training!
        "train_batch_size": 1000, # 400 # best for train time: 256: 6.1s/train_iter # for actual training!
        "sgd_minibatch_size": 128, # 32 # best for train time: 256: 6.1s/train_iter # for actual training!
        # "train_batch_size": 2000, # 400 # best for train time: 256: 6.1s/train_iter # for actual training!
        # "sgd_minibatch_size": 2000, # 32 # best for train time: 256: 6.1s/train_iter # for actual training!

        "disable_observation_precprocessing": True,
        "use_new_rl_modules": False,
        "use_new_learner_api": False,
        "framework": args.framework,
        
        # impala params
        "num_envs_per_worker": 1,
        "lr": 1e-4, # 1e-4
        # "lr_schedule": [ # will overrule lr value if lr param is also provided
        #     [0, 0.0005],
        #     [200000000, 0.000000000001], # the fist value is the num_env_steps the second entry the lr
        #     # [0, 0.0005],
        #     # [2000000, 0.000000000001],  # too aggrive!!
        # ],
        "entropy_coeff": 0.02, #  0.0  a higher entropy_coeff make the policy more explorative and less greedy wirt to the policy loss.
        # ppo params 
        "clip_param": 1, # 0.3, # clips policy updates between 1 - clip_param to 1 + clip_param for the action probabilty changes of a sampled trajectory
        "kl_coeff": 0.01, # 0.2, # how strong of policy updates are permitted. High value of kl_coeff will only change the polcy slightly since large penealty for policy change
        "kl_target": 0.5, # 0.01, # determines how strong the new candidate / target policy can diverge from the current one. A high value allows for more extrem policy cahnges.

        # agent model fcn params
        "fcnet_hiddens": (32, 32), # (4, 4)
        "post_fcnet_hiddens": (64,), # (16,)
        "cnn_activation": "relu",
        "fcnet_activation": "relu",
        "post_fcnet_activation": "relu",
        
        # orig:
        # "fcnet_hidden": (4, 4),
        # "post_fcnet_hidden": (16,),
        # "cnn_activation": "relu",
        # "fcnet_activation": "relu",
        # "post_fcnet_activation": "relu",
        # "use_lstm": True,
        # "lstm_use_prev_action": True,
        # "lstm_use_prev_reward": False,
        # "lstm_cell_size": 2,
        # "shared_policy": False,

        # agent model lstm params
        "max_seq_len": 25, # added by me to lstm
        "use_lstm": False, # True
        "lstm_use_prev_action": True, # True
        "lstm_use_prev_reward": False, # False
        "lstm_cell_size": 2, # 2
        "shared_policy": False, # False

        # agent model attention params
        # "use_attention": True,
        # "max_seq_len": 25,
        # "attention_num_transformer_units": 2,
        # "attention_dim": 32,
        # "attention_memory_inference": 25,
        # "attention_memory_training": 25,
        # "attention_num_heads": 1,
        # "attention_head_dim": 32,
        # "attention_position_wise_mlp_dim": 128, # 32

        # experiment trials
        "exp_name": args.exp,
        "stopping": {
                    # "timesteps_total": 1000000,
                    "training_iteration": 10000, # 1
                    #"episode_reward_mean": 100,
        },
        "num_checkpoints": 2,
        "checkpoint_interval": 10,
        "checkpoint_at_end": True,
        "results_dir": args.results_dir,
        "logging": args.logging,

    }

    # Preferrable to update the parameters in above dict before changing anything below
    
    run_configs = default_config
    experiment_configs = {}
    tune_configs = None

    # Resources 
    run_configs.num_rollout_workers = params_dict['num_rollout_workers']
    run_configs.num_gpus = params_dict['num_gpus']


    # Training
    run_configs.train_batch_size = params_dict['train_batch_size']
    run_configs.sgd_minibatch_size = params_dict['sgd_minibatch_size']
    run_configs.preprocessor_pref = None
    run_configs._disable_preprocessor_api = params_dict['disable_observation_precprocessing']
    run_configs.rl_module(_enable_rl_module_api=params_dict['use_new_rl_modules'])
    run_configs.training(_enable_learner_api=params_dict['use_new_learner_api'])
    run_configs = run_configs.framework(params_dict['framework'])
    run_configs.log_level = params_dict['logging']
    run_configs.seed = params_dict['seed']

    # Environment
    run_configs.env = params_dict['env_name']
    run_configs.env_config = params_dict['env_config']

    # Central critic params
    # run_configs.rollouts(batch_mode="complete_episodes", num_rollout_workers=run_configs.num_rollout_workers)
    run_configs.rollouts(batch_mode="complete_episodes", num_rollout_workers=0)
    run_configs.model["custom_model_config"] = {"player_roles": player_roles} # needed to find dimension of central critic model concatenations during init.

    # .rollouts(batch_mode="complete_episodes", num_rollout_workers=0) # orig

    # Setup multi-agent policies. The below code will initialize independent
    # policies for each agent.
    base_env = make_envs.env_creator(run_configs.env_config)
    policies = {}
    player_to_agent = {}
    for i in range(len(player_roles)):
        rgb_shape = base_env.observation_space[f"player_{i}"]["RGB"].shape
        sprite_x = rgb_shape[0]
        sprite_y = rgb_shape[1]

        policies[f"agent_{i}"] = policy.PolicySpec(
            observation_space=base_env.observation_space[f"player_{i}"],
            action_space=base_env.action_space[f"player_{i}"],
            config={
                "model": {
                    "conv_filters": [[16, [8, 8], 1],
                                    [128, [sprite_x, sprite_y], 1]],
                },
            })
        player_to_agent[f"player_{i}"] = f"agent_{i}"

    run_configs.multi_agent(policies=policies, policy_mapping_fn=(lambda agent_id, *args, **kwargs: 
                                                                  player_to_agent[agent_id]))
    
    # fully connected net params
    run_configs.model["fcnet_hiddens"] = params_dict['fcnet_hiddens']
    run_configs.model["post_fcnet_hiddens"] = params_dict['post_fcnet_hiddens']
    run_configs.model["conv_activation"] = params_dict['cnn_activation'] 
    run_configs.model["fcnet_activation"] = params_dict['fcnet_activation']
    run_configs.model["post_fcnet_activation"] = params_dict['post_fcnet_activation']
    
    # lstm params
    run_configs.model["use_lstm"] = params_dict['use_lstm']
    run_configs.model["lstm_use_prev_action"] = params_dict['lstm_use_prev_action']
    run_configs.model["lstm_use_prev_reward"] = params_dict['lstm_use_prev_reward']
    run_configs.model["lstm_cell_size"] = params_dict['lstm_cell_size']
    
    # transformer based params
    run_configs.model["custom_model"] = "cc_model" # transformer, melting_pot_nn, cc_model
    # run_configs.model["use_attention"] = params_dict['use_attention'] # will use AttentionWrapper from attention_net.py -> not needed for custom transformer
    # run_configs.model["max_seq_len"] = params_dict['max_seq_len']
    # run_configs.model["attention_num_transformer_units"] = params_dict['attention_num_transformer_units']
    # run_configs.model["attention_dim"] = params_dict['attention_dim']
    # run_configs.model["attention_memory_inference"] = params_dict['attention_memory_inference']
    # run_configs.model["attention_memory_training"] = params_dict['attention_memory_training']
    # run_configs.model["attention_num_heads"] = params_dict['attention_num_heads']
    # run_configs.model["attention_head_dim"] = params_dict['attention_head_dim']
    # run_configs.model["attention_position_wise_mlp_dim"] = params_dict['attention_position_wise_mlp_dim']
    
    # impala algo settings:
    run_configs["lr"] = params_dict['lr']
    # run_configs["lr_schedule"] = params_dict['lr_schedule']
    run_configs["entropy_coeff"] = params_dict['entropy_coeff']
    run_configs["clip_param"] = params_dict['clip_param']
    run_configs["kl_coeff"] = params_dict['kl_coeff']
    run_configs["kl_target"] = params_dict['kl_target']

    # Experiment Trials
    experiment_configs['name'] = params_dict['exp_name']
    experiment_configs['stop'] = params_dict['stopping']
    experiment_configs['keep'] = params_dict['num_checkpoints']
    experiment_configs['freq'] = params_dict['checkpoint_interval']
    experiment_configs['end'] = params_dict['checkpoint_at_end']
    if args.framework == 'tf':
        experiment_configs['dir'] = f"{params_dict['results_dir']}/tf"
    else:
        experiment_configs['dir'] = f"{params_dict['results_dir']}/torch"
 
    return run_configs, experiment_configs, tune_configs