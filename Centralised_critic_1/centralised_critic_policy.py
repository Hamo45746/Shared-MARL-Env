import numpy as np
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from centralised_critic_postprocessing import centralized_critic_postprocessing
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

class CCPPOTorchPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.compute_central_vf = self.model.central_value_function

    # Override the loss function to use the centralized value function
    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        # Save the original value function
        vf_saved = model.value_function

        # Replace with the centralized value function
        model.value_function = lambda: self.model.central_value_function(
            train_batch[SampleBatch.CUR_OBS],
            train_batch["global_obs"],
            train_batch["global_actions"]
        )

        loss = super().loss(model, dist_class, train_batch)

        # Restore the original value function
        model.value_function = vf_saved

        return loss

    # Override the post-processing function to collect global info
    @override(PPOTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        return centralized_critic_postprocessing(self, sample_batch, other_agent_batches, episode)