"""
Minimal Step 7: Evaluate a state-based diffusion policy for OpenCabinet
======================================================================

Loads the checkpoint from 06_train_diffusion_policy.py and evaluates it
by iteratively denoising an action sequence conditioned on the current state.
"""

import argparse
import os
import sys
from collections import deque
import math

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

DEFAULT_CHECKPOINT_PATH = "/tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt"
REQUIRED_CHECKPOINT_KEYS = [
    "model_config",
    "model_state_dict",
    "state_mean",
    "action_mean",
]


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def resolve_checkpoint_path(checkpoint_path: str | None) -> str:
    path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Expected a diffusion checkpoint from 06_train_diffusion_policy.py, "
            f"for example: {DEFAULT_CHECKPOINT_PATH}"
        )
    return path


def validate_diffusion_checkpoint(checkpoint, checkpoint_path: str):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type from {checkpoint_path}: {type(checkpoint).__name__}. "
            "Expected a dict saved by 06_train_diffusion_policy.py."
        )

    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        available = sorted(checkpoint.keys())
        raise ValueError(
            f"Incompatible checkpoint file: {checkpoint_path}\n"
            f"Missing required keys: {missing}\n"
            f"Available keys: {available}\n"
            "Use /tmp/cabinet_policy_checkpoints/best_diffusion_policy.pt "
            "or final_diffusion_policy.pt from 06_train_diffusion_policy.py."
        )


class SinusoidalTimeEmbedding:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, timesteps, torch_module):
        import torch

        device = timesteps.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


def build_diffusion_mlp(state_dim: int, action_dim: int, pred_horizon: int, hidden_dim: int):
    import torch.nn as nn

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
                nn.Linear(flat_action_dim + state_dim + hidden_dim, hidden_dim),
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

        def forward(self, noisy_actions, timesteps, cond_state):
            B = noisy_actions.shape[0]
            x = noisy_actions.reshape(B, -1)
            t_emb = self.time_embed(timesteps, None)
            t_emb = self.time_proj(t_emb)
            h = np  # placeholder to keep linter quiet

    return DiffusionMLP()  # placeholder


def build_model_from_config(cfg):
    import torch
    import torch.nn as nn

    state_dim = int(cfg["state_dim"])
    action_dim = int(cfg["action_dim"])
    pred_horizon = int(cfg["pred_horizon"])
    hidden_dim = int(cfg["hidden_dim"])

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
                nn.Linear(flat_action_dim + state_dim + hidden_dim, hidden_dim),
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

        def forward(self, noisy_actions, timesteps, cond_state):
            B = noisy_actions.shape[0]
            x = noisy_actions.reshape(B, -1)
            t_emb = self.time_embed(timesteps, torch)
            t_emb = self.time_proj(t_emb)
            h = torch.cat([x, cond_state, t_emb], dim=-1)
            out = self.net(h)
            return out.reshape_as(noisy_actions)

    return DiffusionMLP()


def load_policy(checkpoint_path, device):
    import torch

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_diffusion_checkpoint(checkpoint, checkpoint_path)
    model_cfg = checkpoint["model_config"]

    model = build_model_from_config(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch:          {checkpoint['epoch']}")
    print(f"  Metric:         {checkpoint.get('metric_name', 'loss')}={checkpoint['loss']:.6f}")
    print(f"  State dim:      {model_cfg['state_dim']}")
    print(f"  Action dim:     {model_cfg['action_dim']}")
    print(f"  Pred horizon:   {model_cfg['pred_horizon']}")
    print(f"  Action horizon: {model_cfg['action_horizon']}")
    print(f"  Diffusion iters:{model_cfg['num_diffusion_steps']}")

    return checkpoint, model


def extract_state(obs, state_keys, expected_dim):
    parts = []
    for key in state_keys:
        if key not in obs:
            raise KeyError(f"Missing observation key in live env: {key}")
        parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    state = np.concatenate(parts).astype(np.float32)
    if len(state) != expected_dim:
        raise ValueError(
            f"Extracted state dim {len(state)} does not match checkpoint dim {expected_dim}"
        )
    return state


def extract_flat_state(obs, expected_dim):
    parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray) and not key.endswith("_image"):
            parts.append(val.reshape(-1))

    if not parts:
        return np.zeros(expected_dim, dtype=np.float32)

    state = np.concatenate(parts).astype(np.float32)
    if len(state) < expected_dim:
        state = np.pad(state, (0, expected_dim - len(state)))
    elif len(state) > expected_dim:
        state = state[:expected_dim]
    return state


def adapt_action_to_env(action, env_action_dim):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if len(action) < env_action_dim:
        action = np.pad(action, (0, env_action_dim - len(action)))
    elif len(action) > env_action_dim:
        action = action[:env_action_dim]
    return np.clip(action, -1.0, 1.0)


def sample_action_chunk(
    model,
    state_vec,
    checkpoint,
    device,
):
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    cfg = checkpoint["model_config"]
    pred_horizon = int(cfg["pred_horizon"])
    action_dim = int(cfg["action_dim"])
    num_diffusion_steps = int(cfg["num_diffusion_steps"])

    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)

    norm_state = (state_vec - state_mean) / state_std
    state_tensor = torch.from_numpy(norm_state).unsqueeze(0).to(device, dtype=torch.float32)

    scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_steps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(num_diffusion_steps)

    with torch.no_grad():
        sample = torch.randn((1, pred_horizon, action_dim), device=device)

        for t in scheduler.timesteps:
            timestep = torch.full((1,), int(t), device=device, dtype=torch.long)
            noise_pred = model(sample, timestep, state_tensor)
            sample = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=sample,
            ).prev_sample

    naction = sample[0].cpu().numpy()
    action_chunk = naction * action_std[None, :] + action_mean[None, :]
    return action_chunk.astype(np.float32)


def summarize_array(name, values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return (
        f"{name}: min={values.min(): .3f} max={values.max(): .3f} "
        f"mean={values.mean(): .3f} std={values.std(): .3f}"
    )


def run_evaluation(
    model,
    checkpoint,
    num_rollouts,
    max_steps,
    split,
    video_path,
    seed,
    debug_stats=False,
):
    import imageio

    cfg = checkpoint["model_config"]
    state_keys = checkpoint.get("state_keys", DEFAULT_STATE_KEYS)
    state_format = checkpoint.get("state_format", "structured_state_keys")
    state_dim = int(cfg["state_dim"])
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

        action_queue = deque()
        ep_reward = 0.0
        success = False

        for step in range(max_steps):
            if not action_queue:
                if state_format == "packed_observation_state":
                    state_vec = extract_flat_state(obs, state_dim)
                else:
                    state_vec = extract_state(obs, state_keys, state_dim)
                action_chunk = sample_action_chunk(
                    model=model,
                    state_vec=state_vec,
                    checkpoint=checkpoint,
                    device=device,
                )
                if debug_stats and ep == 0 and step == 0:
                    print(summarize_array("state", state_vec))
                    print(summarize_array("sampled_action_chunk", action_chunk))
                for a in action_chunk[:action_horizon]:
                    action_queue.append(a)

            action = action_queue.popleft()
            env_action = adapt_action_to_env(action, env.action_dim)
            if debug_stats and ep == 0 and step < 3:
                print(summarize_array(f"env_action_step_{step}", env_action))
            obs, reward, done, info = env.step(env_action)
            ep_reward += reward

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a minimal diffusion OpenCabinet policy")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--split", type=str, default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_stats", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import diffusers  # noqa: F401
    except ImportError:
        print("ERROR: Missing dependencies. Install with: pip install torch diffusers")
        sys.exit(1)

    print_section("Minimal Diffusion Policy Evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Checkpoint: {resolve_checkpoint_path(args.checkpoint)}")
    checkpoint, model = load_policy(args.checkpoint, device)

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
