"""
Improved Step 7: Evaluate a trained OpenCabinet policy
======================================================

Loads the improved checkpoint from 06_train_policy_improved.py and evaluates it
with the exact same state schema, normalization, and optional action chunking.
"""

import argparse
import os
import sys
from collections import deque

# Force osmesa (CPU offscreen renderer) on Linux/WSL2.
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


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


class PolicyNetConfig:
    def __init__(self, cfg_dict):
        self.state_dim = int(cfg_dict["state_dim"])
        self.action_dim = int(cfg_dict["action_dim"])
        self.chunk_size = int(cfg_dict.get("chunk_size", 1))
        self.hidden_dim = int(cfg_dict.get("hidden_dim", 256))

    @property
    def output_dim(self):
        return self.action_dim * self.chunk_size


def build_model(net_cfg, nn):
    return nn.Sequential(
        nn.Linear(net_cfg.state_dim, net_cfg.hidden_dim),
        nn.LayerNorm(net_cfg.hidden_dim),
        nn.GELU(),
        nn.Linear(net_cfg.hidden_dim, net_cfg.hidden_dim),
        nn.LayerNorm(net_cfg.hidden_dim),
        nn.GELU(),
        nn.Linear(net_cfg.hidden_dim, net_cfg.hidden_dim),
        nn.LayerNorm(net_cfg.hidden_dim),
        nn.GELU(),
        nn.Linear(net_cfg.hidden_dim, net_cfg.output_dim),
        nn.Tanh(),
    )


def load_policy(checkpoint_path, device):
    import torch
    import torch.nn as nn

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = PolicyNetConfig(checkpoint["model_config"])
    model = build_model(model_cfg, nn).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    state_keys = checkpoint.get("state_keys", DEFAULT_STATE_KEYS)
    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)

    print(f"Loaded policy from: {checkpoint_path}")
    print(f"  Epoch:        {checkpoint['epoch']}")
    print(f"  Metric:       {checkpoint.get('metric_name', 'loss')}={checkpoint['loss']:.6f}")
    print(f"  State dim:    {model_cfg.state_dim}")
    print(f"  Action dim:   {model_cfg.action_dim}")
    print(f"  Chunk size:   {model_cfg.chunk_size}")
    print(f"  State keys:   {state_keys}")

    return model, model_cfg, state_keys, state_mean, state_std, action_mean, action_std


def extract_state(obs, state_keys, state_dim):
    parts = []
    for key in state_keys:
        if key not in obs:
            raise KeyError(f"Observation key '{key}' missing from live environment obs.")
        parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    state = np.concatenate(parts).astype(np.float32)
    if len(state) != state_dim:
        raise ValueError(
            f"Extracted state dim {len(state)} does not match checkpoint dim {state_dim}."
        )
    return state


def policy_step(
    model,
    obs,
    action_queue,
    state_keys,
    state_mean,
    state_std,
    action_mean,
    action_std,
    model_cfg,
    device,
):
    import torch

    if action_queue:
        return action_queue.popleft()

    raw_state = extract_state(obs, state_keys, model_cfg.state_dim)
    norm_state = (raw_state - state_mean) / state_std

    with torch.no_grad():
        state_tensor = torch.from_numpy(norm_state).unsqueeze(0).to(device)
        pred = model(state_tensor).cpu().numpy().reshape(model_cfg.chunk_size, model_cfg.action_dim)

    denorm_actions = pred * action_std[None, :] + action_mean[None, :]
    for a in denorm_actions:
        action_queue.append(a.astype(np.float32))
    return action_queue.popleft()


def adapt_action_to_env(action, env_action_dim):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if len(action) < env_action_dim:
        action = np.pad(action, (0, env_action_dim - len(action)))
    elif len(action) > env_action_dim:
        action = action[:env_action_dim]
    return np.clip(action, -1.0, 1.0)


def run_evaluation(
    model,
    model_cfg,
    state_keys,
    state_mean,
    state_std,
    action_mean,
    action_std,
    num_rollouts,
    max_steps,
    split,
    video_path,
    seed,
):
    import imageio

    device = next(model.parameters()).device
    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )

    video_writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {"successes": [], "episode_lengths": [], "rewards": []}

    for ep in range(num_rollouts):
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        ep_reward = 0.0
        success = False
        action_queue = deque()

        for step in range(max_steps):
            action = policy_step(
                model=model,
                obs=obs,
                action_queue=action_queue,
                state_keys=state_keys,
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
                model_cfg=model_cfg,
                device=device,
            )
            env_action = adapt_action_to_env(action, env.action_dim)
            obs, reward, done, info = env.step(env_action)
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(frame)

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
            f"  Episode {ep + 1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={step + 1:4d}, reward={ep_reward:.1f}) "
            f'layout={env.layout_id}, style={env.style_id}, task="{lang}"'
        )

    if video_writer:
        video_writer.close()
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate an improved OpenCabinet policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
    )
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Improved Policy Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, model_cfg, state_keys, state_mean, state_std, action_mean, action_std = load_policy(
        args.checkpoint,
        device,
    )

    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")
    results = run_evaluation(
        model=model,
        model_cfg=model_cfg,
        state_keys=state_keys,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
    )

    print_section("Evaluation Results")
    num_success = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100.0
    avg_length = float(np.mean(results["episode_lengths"]))
    avg_reward = float(np.mean(results["rewards"]))

    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")
    if args.video_path:
        print(f"  Video saved to: {args.video_path}")


if __name__ == "__main__":
    main()