"""Inference-time changes without weight compression: world-model residency, conditioning
tokens, flow-matching solver steps, and predicted/executed action horizons.

Every handler takes (policy, change, calibration) and returns a list of event records.
"""

from evaluation.compression.topology import roots


def remove_world_model(policy, change, calibration):
    """Drop the V-JEPA 2 encoder and predictor; inference never calls them."""
    policy.model.video_encoder = None
    policy.model.video_predictor = None
    policy.model.video_processor = None
    return [{"method": "world_remove", "outputs_expected_identical": True}]


def conditioning_tokens(policy, change, calibration):
    if change.get("scope") == "latent":
        policy.model.replace_prompt = ""
    else:
        token = policy.config.embodied_action_token
        policy.model.embodied_replace_prompt = token * change["count"]
    return [change]


def solver_steps(policy, change, calibration):
    roots(policy)["action"].num_inference_timesteps = change["count"]
    return [change]


def action_horizon(policy, change, calibration):
    """Predict and execute fewer actions per chunk."""
    count = change["count"]
    if not 1 <= count <= policy.config.chunk_size:
        raise ValueError("Action horizon must fit the trained chunk")
    roots(policy)["action"].action_horizon = count
    policy.config.n_action_steps = count
    policy.reset()
    return [
        {
            **change,
            "trained_chunk_size": policy.config.chunk_size,
            "executed_before_replanning": count,
        }
    ]


def replanning(policy, change, calibration):
    """Execute fewer of the predicted actions before predicting again."""
    count = change["count"]
    horizon = roots(policy)["action"].action_horizon
    if not 1 <= count <= horizon:
        raise ValueError("Replanning horizon must fit the predicted chunk")
    policy.config.n_action_steps = count
    policy.reset()
    return [{**change, "predicted_chunk_size": horizon}]
