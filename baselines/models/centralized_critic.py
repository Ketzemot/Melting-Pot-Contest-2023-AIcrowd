import argparse
from typing import Tuple, Type
import numpy as np
from gymnasium.spaces import Discrete
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import (
    PPOTF1Policy,
    PPOTF2Policy,
)
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper

from baselines.models.torch_cc_model import TorchCCModel # import own model

from ray.rllib.examples.models.centralized_critic_models import (
    CentralizedCriticModel,
    TorchCentralizedCriticModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

from baselines.train.utils import PLAYER_STR_FORMAT

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

# OPPONENT_OBS = "opponent_obs"
# OPPONENT_ACTION = "opponent_action"

OTHER_AGENTS_OBS = "other_agents_obs"
OTHER_AGENTS_ACTIONS = "other_agents_actions"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=100, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=7.99, help="Reward at which we stop training."
)





class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""
    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or (
        not pytorch and policy.loss_initialized()
    ):
        assert other_agent_batches is not None
        if isinstance(other_agent_batches, dict):
            # other_agent_sample_batches = [policy_tuple[2] for player_id, policy_tuple in other_agent_batches.items()]
            # other_agent_sample_batches = {player_id: policy_tuple[2] for player_id, policy_tuple in other_agent_batches.items()}
            other_agent_sample_batches = {policy.config["policy_mapping_fn"](player_id): policy_tuple[2] for player_id, policy_tuple in other_agent_batches.items()}
        
            # also record the opponent obs and actions in the trajectory
            # will throw error in: tree.map_structure(lambda v: v[permutation], self_as_dict) in sample_batch.py
            # sample_batch[OTHER_AGENTS_OBS] = [other_agent_sample_batch[SampleBatch.CUR_OBS] for other_agent_sample_batch in other_agent_sample_batches]
            # sample_batch[OTHER_AGENTS_ACTIONS] = [other_agent_sample_batch[SampleBatch.ACTIONS] for other_agent_sample_batch in other_agent_sample_batches]
            
            sample_batch[OTHER_AGENTS_OBS] = {agent_id: other_agent_sample_batch[SampleBatch.CUR_OBS] for agent_id, other_agent_sample_batch in other_agent_sample_batches.items()}
            sample_batch[OTHER_AGENTS_ACTIONS] = {agent_id: other_agent_sample_batch[SampleBatch.ACTIONS] for agent_id, other_agent_sample_batch in other_agent_sample_batches.items()}

            # testing:
            # self_as_dict = {k: v for k, v in self.items()}
            # permutation = np.random.permutation(len(sample_batch))
            # self_as_dict = {SampleBatch.CUR_OBS: sample_batch[SampleBatch.CUR_OBS], OTHER_AGENTS_ACTIONS: sample_batch[OTHER_AGENTS_OBS]}
            # self_as_dict = sample_batch.copy()
            # import tree
            # shuffle_out = tree.map_structure(lambda v: v[permutation], self_as_dict)
            # with dict:
            # sample_batch[OTHER_AGENTS_OBS] = {player_id: other_agent_sample_batch[SampleBatch.CUR_OBS] for player_id, other_agent_sample_batch in other_agent_sample_batches.items()}
            # sample_batch[OTHER_AGENTS_ACTIONS] = {player_id: other_agent_sample_batch[SampleBatch.ACTIONS] for player_id, other_agent_sample_batch in other_agent_sample_batches.items()}
            # orig:
            # sample_batch[OTHER_AGENTS_ACTIONS] = opponent_batch[SampleBatch.ACTIONS]
            # sample_batch[OTHER_AGENTS_ACTIONS] = opponent_batch[SampleBatch.ACTIONS]
        else:
            if policy.config["enable_connectors"]:
                [(_, _, opponent_batch)] = list(other_agent_batches.values())
            else:
                [(_, opponent_batch)] = list(other_agent_batches.values())

            # also record the opponent obs and actions in the trajectory
            sample_batch[OTHER_AGENTS_OBS] = opponent_batch[SampleBatch.CUR_OBS] 
            sample_batch[OTHER_AGENTS_ACTIONS] = opponent_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF
        if policy.framework == "torch":
            # central_value_function(self, obs, other_agents_obs, other_agents_actions, ego_agent_indizes):
            sample_batch[SampleBatch.VF_PREDS] = (
                policy.compute_central_vf(
                    convert_to_torch_tensor(
                        sample_batch[SampleBatch.CUR_OBS], policy.device
                    ),
                    convert_to_torch_tensor(sample_batch[OTHER_AGENTS_OBS], policy.device),
                    convert_to_torch_tensor(
                        sample_batch[OTHER_AGENTS_ACTIONS], policy.device
                    ),
                    1 if "agent" in policy.config["__policy_id"] else 0, # NOTE this has to be changed for bots
                )
                .cpu()
                .detach()
                .numpy()
            )
        else:
            sample_batch[SampleBatch.VF_PREDS] = convert_to_numpy(
                policy.compute_central_vf(
                    sample_batch[SampleBatch.CUR_OBS],
                    sample_batch[OTHER_AGENTS_OBS],
                    sample_batch[OTHER_AGENTS_ACTIONS],
                )
            )
    else:
        # Policy hasn't been initialized yet, use zeros.
        # orig:
        # sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        # sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        # sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

        # sample_batch[OPPONENT_OBS] = {k: np.zeros_like(sample_batch[SampleBatch.CUR_OBS][k] ) for k,v in sample_batch[SampleBatch.CUR_OBS].items()} if isinstance(sample_batch[SampleBatch.CUR_OBS], dict) else np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        
        # maybe I don't need the number of other agents during policy intialisation. I can simply look a the len of opponents obs and iterate over them in the cummon critic?
        # sample_batch[OPPONENT_OBS] = [{k: np.zeros_like(sample_batch[SampleBatch.CUR_OBS][k] ) for k,v in sample_batch[SampleBatch.CUR_OBS].items()} if isinstance(sample_batch[SampleBatch.CUR_OBS], dict) else np.zeros_like(sample_batch[SampleBatch.CUR_OBS])]

        num_other_agents = len(policy.config["model"]["custom_model_config"]["player_roles"]) - 1

        _ordered_player_ids = [PLAYER_STR_FORMAT.format(index=index) for index in range(num_other_agents+1)]

        # num_other_agents = 1 # for debug!
        if isinstance(sample_batch[SampleBatch.CUR_OBS], dict):
            # NOTE: this cannot be a list later but it has to identify the agent type so no more *num_other_agents
            # orig: this is a list we need a dict
            # sample_batch[OTHER_AGENTS_OBS] = [{k: np.zeros_like(sample_batch[SampleBatch.CUR_OBS][k] ) for k,v in sample_batch[SampleBatch.CUR_OBS].items()} ]*num_other_agents
            sample_batch_other_agents_obs_dict, sample_batch_other_agents_act_dict = {}, {}
            for player_id_candidate in _ordered_player_ids:
                if policy.config["policy_mapping_fn"](player_id_candidate) != policy.config["__policy_id"]: 
                    sample_batch_other_agents_obs_dict.update({policy.config["policy_mapping_fn"](player_id_candidate): {k: np.zeros_like(sample_batch[SampleBatch.CUR_OBS][k] ) for k,v in sample_batch[SampleBatch.CUR_OBS].items()} })
                    sample_batch_other_agents_act_dict.update({policy.config["policy_mapping_fn"](player_id_candidate): np.zeros_like(sample_batch[SampleBatch.ACTIONS]) } )
            
            sample_batch[OTHER_AGENTS_OBS] = sample_batch_other_agents_obs_dict
            sample_batch[OTHER_AGENTS_ACTIONS] = sample_batch_other_agents_act_dict
            
            # this logic is not needed
            # sample_batch[OPPONENT_OBS] = [{"obs_flat" if k == "obs" else k : np.zeros_like(sample_batch[SampleBatch.CUR_OBS][k] ) for k,v in sample_batch[SampleBatch.CUR_OBS].items()}]
            # NOTE: this cannot be a list later but it has to identify the agent type so no more *num_other_agents
            # orig: this is a list we need a dict
            # sample_batch[OTHER_AGENTS_ACTIONS] = [np.zeros_like(sample_batch[SampleBatch.ACTIONS])]*num_other_agents
        
        # result = [{"obs_flat" if k == "obs" else k: np.zeros_like(v) for k, v in sample_batch[SampleBatch.CUR_OBS].items()}]
        else:
            np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
            # sample_batch[OPPONENT_OBS] = [{k: np.zeros_like(sample_batch[SampleBatch.CUR_OBS][k] ) for k,v in sample_batch[SampleBatch.CUR_OBS].items()} if isinstance(sample_batch[SampleBatch.CUR_OBS], dict) else np.zeros_like(sample_batch[SampleBatch.CUR_OBS])]
            # sample_batch[OPPONENT_OBS] = [k for opp_ops.items() in  sample_batch[OPPONENT_OBS] ]
            sample_batch[OTHER_AGENTS_ACTIONS] = [np.zeros_like(sample_batch[SampleBatch.ACTIONS])]*num_other_agents
        
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32) # central critic value function output prediction

        # if self.model_config.get("_disable_preprocessor_api"):
        #     restored["obs_flat"] = input_dict["obs"]

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[OTHER_AGENTS_OBS],
        train_batch[OTHER_AGENTS_ACTIONS],
        train_batch[SampleBatch.AGENT_INDEX],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss


def central_vf_stats(policy, train_batch):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out
        )
    }


def get_ccppo_policy(base):
    class CCPPOTFPolicy(CentralizedValueMixin, base):
        def __init__(self, observation_space, action_space, config):
            base.__init__(self, observation_space, action_space, config)
            CentralizedValueMixin.__init__(self)

        @override(base)
        def loss(self, model, dist_class, train_batch):
            # Use super() to get to the base PPO policy.
            # This special loss function utilizes a shared
            # value function defined on self, and the loss function
            # defined on PPO policies.
            return loss_with_central_critic(
                self, super(), model, dist_class, train_batch
            )

        @override(base)
        def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
        ):
            return centralized_critic_postprocessing(
                self, sample_batch, other_agent_batches, episode
            )

        @override(base)
        def stats_fn(self, train_batch: SampleBatch):
            stats = super().stats_fn(train_batch)
            stats.update(central_vf_stats(self, train_batch))
            return stats

    return CCPPOTFPolicy


CCPPOStaticGraphTFPolicy = get_ccppo_policy(PPOTF1Policy)
CCPPOEagerTFPolicy = get_ccppo_policy(PPOTF2Policy)


class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return centralized_critic_postprocessing(
            self, sample_batch, other_agent_batches, episode
        )

    # added by me to pass model_kwargs: player_roles = player_roles, agent_id=1,
    # @override(PPOTorchPolicy)
    # def make_model_and_action_dist(self) -> Tuple[ModelV2, type[TorchDistributionWrapper]]:
        
    #     # player_roles = self.config["model"]["custom_model_config"].pop("player_roles")
    #     player_roles = self.config["model"]["custom_model_config"]["player_roles"]
    #     dist_class, logit_dim = ModelCatalog.get_action_dist(
    #             self.action_space, self.config["model"], framework=self.framework
    #         )
        
    #     model = ModelCatalog.get_model_v2(
    #         obs_space=self.observation_space,
    #         action_space=self.action_space,
    #         num_outputs=logit_dim,
    #         model_config=self.config["model"],
    #         framework=self.framework,
    #         player_roles = player_roles,
    #         agent_id=1,
    #     )

    #     # model = ModelCatalog.get_model_v2(
    #     #         obs_space=self.observation_space,
    #     #         action_space=self.action_space,
    #     #         num_outputs=logit_dim,
    #     #         model_config=self.config["model"],
    #     #         framework=self.framework,
    #     #     )
        

    #     return model, dist_class




class CentralizedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
        if config["framework"] == "torch":
            return CCPPOTorchPolicy
        elif config["framework"] == "tf":
            return CCPPOStaticGraphTFPolicy
        else:
            return CCPPOEagerTFPolicy
        


if __name__ == "__main__":
    ray.init(local_mode=True) # orig! 
    # ray.init(local_mode=False)
    # args = parser.parse_args()

    ModelCatalog.register_custom_model(
        # "cc_model", TorchCCModel if "torch" == "torch" else CentralizedCriticModel,
        "cc_model", TorchCentralizedCriticModel,  # TorchCCModel TorchCentralizedCriticModel,  continue here use: TorchCentralizedCriticModel
    )

    config = (
        PPOConfig()
        .environment(TwoStepGame)
        # .framework(args.framework)
        .framework("torch")
        .rollouts(batch_mode="complete_episodes", num_rollout_workers=0)
        # TODO (Kourosh): Lift this example to the new RLModule stack, and enable it.
        .training(model={"custom_model": "cc_model"}, _enable_learner_api=False)
        .multi_agent(
            policies={
                "pol1": (
                    None,
                    Discrete(6),
                    TwoStepGame.action_space,
                    # `framework` would also be ok here.
                    PPOConfig.overrides(framework_str="torch"),
                ),
                "pol2": (
                    None,
                    Discrete(6),
                    TwoStepGame.action_space,
                    # `framework` would also be ok here.
                    PPOConfig.overrides(framework_str="torch"),
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "pol1"
            if agent_id == 0
            else "pol2",
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .resources(num_gpus=1, num_cpus_for_local_worker=3, num_cpus_per_worker=2)
        .rl_module(_enable_rl_module_api=False)
    )

    stop = {
        "training_iteration": 100,
        "timesteps_total": 100000,
        "episode_reward_mean": 7.99,
    }

    tuner = tune.Tuner(
        CentralizedCritic,
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1),
    )
    results = tuner.fit()

    # also test later!! 
    if True:
        # check_learning_achieved(results, args.stop_reward)
        check_learning_achieved(results, 7.99)

