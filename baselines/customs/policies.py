from typing import Tuple

import dm_env
from scipy import special
from meltingpot.utils.policies import policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy import sample_batch

_IGNORE_KEYS = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP'] # orig
# _IGNORE_KEYS = []

class EvalPolicy(policy.Policy):
  """ Loads the policies from  Policy checkpoints and removes unrequired observations
  that policies cannot expect to have access to during evaluation.
  """
  def __init__(self,
               chkpt_dir: str,
               policy_id: str = sample_batch.DEFAULT_POLICY_ID) -> None:
    
    policy_path = f'{chkpt_dir}/{policy_id}'
    self._policy = Policy.from_checkpoint(policy_path)
    self._prev_action = 0
  
  def initial_state(self) -> policy.State:
    """See base class."""

    self._prev_action = 0
    state = self._policy.get_initial_state()
    self.prev_state = state
    return state

  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""

    observations = {
        key: value
        for key, value in timestep.observation.items()
        if key not in _IGNORE_KEYS
    }

    # We want the logic to be stateless so don't use prev_state from input
    # self.action_tuple = ray.get(self.inference_trainer.compute_action.remote(observation=self.network_inf_data_ext_features, full_fetch=True))# works!)
    # self.sample_probabilities = special.softmax(self.action_tuple[2]['action_dist_inputs'])
    
    # action_tuple = self._policy.compute_actions(        
    #     [observations],
    #     self.prev_state,
    #     prev_action=self._prev_action,
    #     prev_reward=timestep.reward) 
        # obs_batch: Union[List[TensorStructType], TensorStructType],
        # state_batches: Optional[List[TensorType]] = None,
        # prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        # prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
    
    # .inference_trainer.compute_action.remote(observation=self.network_inf_data_ext_features, full_fetch=True)


    ### can I introduce custom extra obs? to state

    # orig:
    # action, state, _ = self._policy.compute_single_action(
    #     observations,
    #     self.prev_state,
    #     prev_action=self._prev_action,
    #     prev_reward=timestep.reward)

    action_sampled, state, action_dist = self._policy.compute_single_action(
        observations,
        self.prev_state,
        prev_action=self._prev_action,
        prev_reward=timestep.reward)
    
    sample_probabilities = special.softmax(action_dist['action_dist_inputs'])
    
    action_best = sample_probabilities.argmax()
    action = action_sampled
        # obs: Optional[TensorStructType] = None,
        # state: Optional[List[TensorType]] = None,
        # *,
        # prev_action: Optional[TensorStructType] = None,
        # prev_reward: Optional[TensorStructType] = None,
  
    self._prev_action = action
    self.prev_state = state
    return action, state

  def close(self) -> None:

    """See base class."""