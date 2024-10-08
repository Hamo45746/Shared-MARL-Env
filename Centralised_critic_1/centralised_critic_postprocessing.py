import torch
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np

def centralized_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    # Ensure we have other agent batches
    assert other_agent_batches is not None
    
    # Collect observations and actions from other agents
    global_obs = []
    global_actions = []

    for agent_id, (other_policy, other_batch) in other_agent_batches.items():
        global_obs.append(other_batch[SampleBatch.CUR_OBS])
        global_actions.append(other_batch[SampleBatch.ACTIONS])

    # Stack the observations and actions into numpy arrays
    sample_batch["global_obs"] = np.stack(global_obs, axis=1)
    sample_batch["global_actions"] = np.stack(global_actions, axis=1)

    # Replace the default VF prediction with the centralized value function prediction
    if hasattr(policy, "compute_central_vf"):
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            sample_batch[SampleBatch.CUR_OBS],
            sample_batch["global_obs"],
            sample_batch["global_actions"]
        ).detach().cpu().numpy()

    return compute_advantages(
        sample_batch,
        last_r=0.0,  # Set this according to the environment termination conditions
        gamma=policy.config["gamma"],
        lambda_=policy.config["lambda"],
        use_gae=policy.config["use_gae"]
    )
