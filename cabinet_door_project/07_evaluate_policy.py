"""
Step 7: Evaluate a Minimal State-Based Diffusion Policy
======================================================

This script evaluates the checkpoint produced by 06_train_policy.py by
iteratively denoising an action chunk conditioned on low-dimensional state.

Design goals:
- use the same state schema and normalization saved at training time
- reconstruct the diffusion model from checkpoint config
- sample an action sequence by iterative denoising
- execute only a short action horizon before replanning
- keep the code compact and readable for a class project
"""

import argparse
import math
import os
import sys
from collections import deque
from typing import Dict, List, Optional, Sequence

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env


DEFAULT_STATE_KEYS = [
    "robot0_gripper_qpos",
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
]

DEFAULT_CHECKPOINT_PATH = "/tmp/cabinet_policy_checkpoints/best_policy.pt"
REQUIRED_CHECKPOINT_KEYS = [
    "model_config",
    "model_state_dict",
    "state_mean",
    "state_std",
    "action_mean",
    "action_std",
]


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def resolve_checkpoint_path(checkpoint_path: Optional[str]) -> str:
    path = os.path.expanduser(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Expected a checkpoint from 06_train_policy.py."
        )
    return path


def validate_checkpoint(checkpoint: Dict, checkpoint_path: str) -> None:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type from {checkpoint_path}: "
            f"{type(checkpoint).__name__}"
        )

    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(
            f"Incompatible checkpoint file: {checkpoint_path}\n"
            f"Missing required keys: {missing}\n"
            f"Available keys: {sorted(checkpoint.keys())}"
        )

    model_cfg = checkpoint["model_config"]
    for key in ["state_dim", "action_dim", "pred_horizon", "action_horizon", "hidden_dim"]:
        if key not in model_cfg:
            raise ValueError(
                f"Checkpoint model_config is missing '{key}': {sorted(model_cfg.keys())}"
            )


class SinusoidalTimeEmbedding:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, timesteps, torch_module):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch_module.exp(
            torch_module.arange(half_dim, device=timesteps.device) * -emb_scale
        )
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch_module.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch_module.nn.functional.pad(emb, (0, 1))
        return emb


def build_diffusion_mlp(
    obs_horizon: int,
    state_dim: int,
    action_dim: int,
    pred_horizon: int,
    hidden_dim: int,
):
    import torch
    import torch.nn as nn

    cond_dim = obs_horizon * state_dim
    flat_action_dim = pred_horizon * action_dim
    time_embed_dim = hidden_dim

    class DiffusionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
            self.time_proj = nn.Sequential(
                nn.Linear(time_embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.net = nn.Sequential(
                nn.Linear(flat_action_dim + cond_dim + hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, flat_action_dim),
            )

        def forward(self, noisy_actions, timesteps, obs_seq):
            batch_size = noisy_actions.shape[0]
            noisy_flat = noisy_actions.reshape(batch_size, -1)
            obs_flat = obs_seq.reshape(batch_size, -1)
            t_emb = self.time_embed(timesteps, torch)
            t_emb = self.time_proj(t_emb)
            hidden = torch.cat([noisy_flat, obs_flat, t_emb], dim=-1)
            out = self.net(hidden)
            return out.reshape_as(noisy_actions)

    return DiffusionMLP()


def build_model_from_config(model_cfg: Dict):
    return build_diffusion_mlp(
        obs_horizon=int(model_cfg.get("obs_horizon", 1)),
        state_dim=int(model_cfg["state_dim"]),
        action_dim=int(model_cfg["action_dim"]),
        pred_horizon=int(model_cfg["pred_horizon"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
    )


def load_policy(checkpoint_path: str, device):
    import torch

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint(checkpoint, checkpoint_path)

    model = build_model_from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cfg = checkpoint["model_config"]
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch:          {checkpoint['epoch']}")
    print(
        f"  Metric:         {checkpoint.get('metric_name', 'loss')}="
        f"{checkpoint['loss']:.6f}"
    )
    print(f"  State dim:      {cfg['state_dim']}")
    print(f"  Action dim:     {cfg['action_dim']}")
    print(f"  Obs horizon:    {cfg.get('obs_horizon', 1)}")
    print(f"  Pred horizon:   {cfg['pred_horizon']}")
    print(f"  Action horizon: {cfg['action_horizon']}")
    print(f"  Diffusion iters:{cfg['num_diffusion_steps']}")

    return checkpoint, model


def flatten_value(val) -> np.ndarray:
    if isinstance(val, np.ndarray):
        return val.astype(np.float32).reshape(-1)
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=np.float32).reshape(-1)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return np.asarray([float(val)], dtype=np.float32)
    return np.asarray([], dtype=np.float32)


def extract_state_from_keys(
    obs: Dict,
    state_keys: Sequence[str],
    expected_dim: int,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    missing = [key for key in state_keys if key not in obs]
    if missing:
        raise KeyError(
            f"Missing observation key(s) in live env: {missing}. "
            f"Available keys sample: {sorted(obs.keys())[:40]}"
        )

    for key in state_keys:
        val = flatten_value(obs[key])
        if val.size == 0:
            raise ValueError(f"Observation key '{key}' is empty or unsupported")
        parts.append(val)

    state = np.concatenate(parts).astype(np.float32)
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"Extracted state dim {state.shape[0]} does not match checkpoint dim {expected_dim}"
        )
    return state


def extract_flat_state(obs: Dict, expected_dim: int) -> np.ndarray:
    parts: List[np.ndarray] = []
    for key in sorted(obs.keys()):
        if key.endswith("_image"):
            continue
        val = flatten_value(obs[key])
        if val.size > 0:
            parts.append(val)

    if not parts:
        raise ValueError("Could not extract any low-dimensional state from environment observation")

    state = np.concatenate(parts).astype(np.float32)
    if state.shape[0] < expected_dim:
        raise ValueError(
            f"Flat observation dim {state.shape[0]} is smaller than checkpoint dim {expected_dim}"
        )
    if state.shape[0] > expected_dim:
        state = state[:expected_dim]
    return state


def extract_live_state(obs: Dict, checkpoint: Dict) -> np.ndarray:
    model_cfg = checkpoint["model_config"]
    expected_dim = int(model_cfg["state_dim"])
    state_format = checkpoint.get("state_format", "structured_state_keys")
    state_keys = checkpoint.get("state_keys", DEFAULT_STATE_KEYS)
    packed_state_column = checkpoint.get("packed_state_column")

    if state_format == "structured_state_keys":
        return extract_state_from_keys(obs, state_keys, expected_dim)

    if packed_state_column and packed_state_column in obs:
        packed = flatten_value(obs[packed_state_column])
        if packed.shape[0] != expected_dim:
            raise ValueError(
                f"Packed observation dim {packed.shape[0]} does not match checkpoint dim "
                f"{expected_dim}"
            )
        return packed.astype(np.float32)

    fallback_keys_present = all(key in obs for key in state_keys)
    if fallback_keys_present:
        state = extract_state_from_keys(obs, state_keys, expected_dim)
        if not checkpoint.get("_warned_packed_fallback_to_keys", False):
            print(
                "WARNING: Checkpoint was trained with packed observation state, but live env does "
                "not expose that packed field. Falling back to ordered state keys."
            )
            checkpoint["_warned_packed_fallback_to_keys"] = True
        return state

    if not checkpoint.get("_warned_packed_fallback_to_flat", False):
        print(
            "WARNING: Checkpoint was trained with packed observation state, but live env does not "
            "expose the packed field. Falling back to flattened observation order."
        )
        checkpoint["_warned_packed_fallback_to_flat"] = True
    return extract_flat_state(obs, expected_dim)


def normalize_state_sequence(obs_seq: np.ndarray, checkpoint: Dict) -> np.ndarray:
    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
    return (obs_seq - state_mean[None, :]) / state_std[None, :]


def unnormalize_action_sequence(action_seq: np.ndarray, checkpoint: Dict) -> np.ndarray:
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)
    return action_seq * action_std[None, :] + action_mean[None, :]


def adapt_action_to_env(action: np.ndarray, env_action_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < env_action_dim:
        action = np.pad(action, (0, env_action_dim - action.shape[0]))
    elif action.shape[0] > env_action_dim:
        action = action[:env_action_dim]
    return np.clip(action, -1.0, 1.0)


def sample_action_chunk(model, obs_seq: np.ndarray, checkpoint: Dict, device) -> np.ndarray:
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    cfg = checkpoint["model_config"]
    pred_horizon = int(cfg["pred_horizon"])
    action_dim = int(cfg["action_dim"])
    num_diffusion_steps = int(cfg["num_diffusion_steps"])

    norm_obs = normalize_state_sequence(obs_seq, checkpoint)
    obs_tensor = torch.from_numpy(norm_obs).unsqueeze(0).to(device=device, dtype=torch.float32)

    scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_steps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(num_diffusion_steps)

    with torch.no_grad():
        sample = torch.randn((1, pred_horizon, action_dim), device=device)

        for timestep in scheduler.timesteps:
            timestep_tensor = torch.full(
                (1,),
                int(timestep),
                device=device,
                dtype=torch.long,
            )
            noise_pred = model(sample, timestep_tensor, obs_tensor)
            sample = scheduler.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=sample,
            ).prev_sample

    action_chunk = sample[0].cpu().numpy()
    return unnormalize_action_sequence(action_chunk, checkpoint).astype(np.float32)


def summarize_array(name: str, values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return (
        f"{name}: min={values.min(): .3f} max={values.max(): .3f} "
        f"mean={values.mean(): .3f} std={values.std(): .3f}"
    )


def run_evaluation(
    model,
    checkpoint: Dict,
    num_rollouts: int,
    max_steps: int,
    split: str,
    video_path: Optional[str],
    seed: int,
    debug_stats: bool = False,
):
    import imageio

    cfg = checkpoint["model_config"]
    obs_horizon = int(cfg.get("obs_horizon", 1))
    action_horizon = int(cfg["action_horizon"])

    device = next(model.parameters()).device
    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )

    writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        writer = imageio.get_writer(video_path, fps=20)

    results = {"successes": [], "episode_lengths": [], "rewards": []}

    for ep in range(num_rollouts):
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")

        first_state = extract_live_state(obs, checkpoint)
        obs_queue = deque([first_state.copy() for _ in range(obs_horizon)], maxlen=obs_horizon)
        action_queue = deque()

        ep_reward = 0.0
        success = False

        for step in range(max_steps):
            if not action_queue:
                obs_seq = np.stack(list(obs_queue), axis=0).astype(np.float32)
                action_chunk = sample_action_chunk(
                    model=model,
                    obs_seq=obs_seq,
                    checkpoint=checkpoint,
                    device=device,
                )

                if debug_stats and ep == 0 and step == 0:
                    print(summarize_array("obs_seq", obs_seq))
                    print(summarize_array("sampled_action_chunk", action_chunk))

                for action in action_chunk[:action_horizon]:
                    action_queue.append(action)

            action = action_queue.popleft()
            env_action = adapt_action_to_env(action, env.action_dim)

            if debug_stats and ep == 0 and step < 3:
                print(summarize_array(f"env_action_step_{step}", env_action))

            obs, reward, done, info = env.step(env_action)
            ep_reward += reward

            live_state = extract_live_state(obs, checkpoint)
            obs_queue.append(live_state)

            if writer is not None:
                frame = env.sim.render(
                    height=512,
                    width=768,
                    camera_name="robot0_agentview_center",
                )[::-1]
                writer.append_data(frame)

            if env._check_success():
                success = True
                break
            if done:
                break

        results["successes"].append(success)
        results["episode_lengths"].append(step + 1)
        results["rewards"].append(ep_reward)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"Episode {ep + 1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={step + 1:4d}, reward={ep_reward:.1f}) "
            f'layout={env.layout_id}, style={env.style_id}, task="{lang}"'
        )

    if writer is not None:
        writer.close()
    env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a minimal diffusion OpenCabinet policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to policy checkpoint (.pt file)",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split to evaluate on",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save evaluation video (optional)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--debug_stats",
        action="store_true",
        help="Print state and sampled action statistics for debugging",
    )
    args = parser.parse_args()

    try:
        import torch
        import diffusers  # noqa: F401
    except ImportError:
        print("ERROR: Missing dependencies. Install with: pip install torch diffusers")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Diffusion Policy Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {resolve_checkpoint_path(args.checkpoint)}")

    checkpoint, model = load_policy(args.checkpoint, device)

    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")
    results = run_evaluation(
        model=model,
        checkpoint=checkpoint,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
        debug_stats=args.debug_stats,
    )

    print_section("Results")
    successes = sum(results["successes"])
    success_rate = 100.0 * successes / args.num_rollouts
    avg_len = float(np.mean(results["episode_lengths"]))
    avg_reward = float(np.mean(results["rewards"]))

    print(f"Split:         {args.split}")
    print(f"Episodes:      {args.num_rollouts}")
    print(f"Successes:     {successes}/{args.num_rollouts}")
    print(f"Success rate:  {success_rate:.1f}%")
    print(f"Avg length:    {avg_len:.1f}")
    print(f"Avg reward:    {avg_reward:.3f}")
    if args.video_path:
        print(f"Video saved:   {args.video_path}")


if __name__ == "__main__":
    main()
