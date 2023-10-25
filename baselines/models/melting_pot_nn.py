from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Any, Union

# TODO (sven): add IMPALA-style option.
# from ray.rllib.examples.models.impala_vision_nets import TorchImpalaVisionNet
from ray.rllib.models.torch.misc import (
    normc_initializer as torch_normc_initializer,
    SlimFC,
)
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_filter_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch, TensorType
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once

from meltingpot.utils.substrates.wrappers.collective_reward_wrapper import _FOCAL_AGENT, _COLLECTIVE_REWARD_OBS

torch, nn = try_import_torch()


class MeltingPotNN(TorchModelV2, nn.Module):
    """TorchModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).

    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.

    The data flow is as follows:

    `obs` (e.g. Tuple[img0, img1, discrete0]) -> `CNN0 + CNN1 + ONE-HOT`
    `CNN0 + CNN1 + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> (optional) FC-stack -> `out2`
    `out2` -> action (logits) and value heads.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **model_kwargs):
        if log_once("complex_input_net_deprecation_torch"):
            deprecation_warning(
                old="ray.rllib.models.torch.complex_input_net.ComplexInputNetwork",
            )
        self.original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )

        self.processed_obs_space = (
            self.original_space
            if model_config.get("_disable_preprocessor_api")
            else obs_space
        )

        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, self.original_space, action_space, num_outputs, model_config, name
        )

        self.flattened_input_space = flatten_space(self.original_space)

        # Atari type CNNs or IMPALA type CNNs (with residual layers)?
        # self.cnn_type = self.model_config["custom_model_config"].get(
        #     "conv_type", "atari")
        
        # self.ego_agend_id = model_kwargs["agent_id"]
        # Build the CNN(s) given obs_space's image components.
        self.cnns = nn.ModuleDict()
        self.one_hot = nn.ModuleDict()
        self.flatten_dims = {}
        self.flatten = nn.ModuleDict()
        concat_size = 0
        for i, component in enumerate(self.flattened_input_space):
            i = str(i)
            # Image space.
            if len(component.shape) == 3 and isinstance(component, Box):
                config = {
                    "conv_filters": model_config["conv_filters"]
                    if "conv_filters" in model_config
                    else get_filter_config(component.shape),
                    "conv_activation": model_config.get("conv_activation"),
                    "post_fcnet_hiddens": [],
                }
                # if self.cnn_type == "atari":
                self.cnns[i] = ModelCatalog.get_model_v2(
                    component,
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    framework="torch",
                    name="cnn_{}".format(i),
                )
                # TODO (sven): add IMPALA-style option.
                # else:
                #    cnn = TorchImpalaVisionNet(
                #        component,
                #        action_space,
                #        num_outputs=None,
                #        model_config=config,
                #        name="cnn_{}".format(i))

                concat_size += self.cnns[i].num_outputs
                self.add_module("cnn_{}".format(i), self.cnns[i])
            # Discrete|MultiDiscrete inputs -> One-hot encode.
            elif isinstance(component, (Discrete, MultiDiscrete)):
                if isinstance(component, Discrete):
                    size = component.n
                else:
                    size = np.sum(component.nvec)
                config = {
                    "fcnet_hiddens": model_config["fcnet_hiddens"],
                    "fcnet_activation": model_config.get("fcnet_activation"),
                    "post_fcnet_hiddens": [],
                }
                self.one_hot[i] = ModelCatalog.get_model_v2(
                    Box(-1.0, 1.0, (size,), np.float32),
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    framework="torch",
                    name="one_hot_{}".format(i),
                )
                concat_size += self.one_hot[i].num_outputs
                self.add_module("one_hot_{}".format(i), self.one_hot[i])
            # Everything else (1D Box).
            else:
                size = int(np.product(component.shape))
                config = {
                    "fcnet_hiddens": model_config["fcnet_hiddens"],
                    "fcnet_activation": model_config.get("fcnet_activation"),
                    "post_fcnet_hiddens": [],
                }
                self.flatten[i] = ModelCatalog.get_model_v2(
                    Box(-1000.0, 1000.0, (size,), np.float32),
                    # Box(-1.0, 1.0, (size,), np.float32), # orig
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    framework="torch",
                    name="flatten_{}".format(i),
                )
                self.flatten_dims[i] = size
                concat_size += self.flatten[i].num_outputs
                self.add_module("flatten_{}".format(i), self.flatten[i])

        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", []),
            "fcnet_activation": model_config.get("post_fcnet_activation", "relu"),
        }
        self.post_fc_stack = ModelCatalog.get_model_v2(
            Box(float("-inf"), float("inf"), shape=(concat_size,), dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="torch",
            name="post_fc_stack",
        )

        # Actions and value heads.
        self.logits_layer = None
        self.value_layer = None
        self._value_out = None

        if num_outputs:
            # Action-distribution head.
            self.logits_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs,
                out_size=num_outputs,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01),
            )
            # Create the value branch model.
            self.value_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs,
                out_size=1,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01),
            )
        else:
            self.num_outputs = concat_size if not self.post_fc_stack else self.post_fc_stack.num_outputs
            # self.num_outputs = concat_size

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS] # orig

        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.processed_obs_space, tensorlib="torch"
            )
        # add focal agent = True in case of eval call where _FOCAL_AGENT does not exist in input
        if input_dict[SampleBatch.OBS] .get(_FOCAL_AGENT) is None:
            orig_obs.update({_FOCAL_AGENT: torch.ones_like(input_dict[SampleBatch.OBS][_COLLECTIVE_REWARD_OBS])})

        # Push observations through the different components
        # (CNNs, one-hot + FC, etc..).
        outs = []
        for i, component in enumerate(tree.flatten(orig_obs)):
            i = str(i)
            if i in self.cnns:
                cnn_out, _ = self.cnns[i](SampleBatch({SampleBatch.OBS: component}))
                outs.append(cnn_out)
            elif i in self.one_hot:
                if component.dtype in [
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                ]:
                    one_hot_in = {
                        SampleBatch.OBS: one_hot(
                            component, self.flattened_input_space[int(i)]
                        )
                    }
                else:
                    one_hot_in = {SampleBatch.OBS: component}
                one_hot_out, _ = self.one_hot[i](SampleBatch(one_hot_in))
                outs.append(one_hot_out)
            else:
                nn_out, _ = self.flatten[i](
                    SampleBatch(
                        {
                            SampleBatch.OBS: torch.reshape(
                                component, [-1, self.flatten_dims[i]]
                            )
                        }
                    )
                )
                outs.append(nn_out)

        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out, _ = self.post_fc_stack(SampleBatch({SampleBatch.OBS: out}))

        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        # Logits- and value branches.
        logits, values = self.logits_layer(out), self.value_layer(out)
        self._value_out = torch.reshape(values, [-1])
        # self._value_out is needed as forward output to get value out for feeding through opp. obs in central critic
        return (logits, self._value_out), []  
        # return logits, [], self._value_out
        # return logits, [] # orig

    @override(ModelV2)
    def value_function(self):
        return self._value_out


    @override(ModelV2)
    def custom_loss(
            self, policy_loss: TensorType, loss_inputs: Dict[str, TensorType]
        ) -> Union[List[TensorType], TensorType]:
            """Override to customize the loss function used to optimize this model.

            This can be used to incorporate self-supervised losses (by defining
            a loss over existing input and output tensors of this model), and
            supervised losses (by defining losses over a variable-sharing copy of
            this model's layers).

            You can find an runnable example in examples/custom_loss.py.

            Args:
                policy_loss: List of or single policy loss(es) from the policy.
                loss_inputs: map of input placeholders for rollout data.

            Returns:
                List of or scalar tensor for the customized loss(es) for this
                model.
            """

            # TODO maybe calc loss exponentaily weighted over time
            # TODO maybe I should edit the reward mapping from dmlab to ray reward (baselines/wrappers/meltingpot_wrapper.py line 47) to add COLLECTIVE_REWARDto reward
            
            # common_reward = torch.mean(loss_inputs["obs"]["COLLECTIVE_REWARD"])
            # policy_loss[0] += common_reward
            return policy_loss
            # return policy_loss
    


    # @override
    # def metrics(self) -> Dict[str, TensorType]:
    #     """Override to return custom metrics from your model.

    #     The stats will be reported as part of the learner stats, i.e.,
    #     info.learner.[policy_id, e.g. "default_policy"].model.key1=metric1

    #     Returns:
    #         The custom metrics for this model.
    #     """
    #     return {}