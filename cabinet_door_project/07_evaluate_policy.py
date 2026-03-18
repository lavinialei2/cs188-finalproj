"""
Step 7: Evaluate a Trained Policy
===================================
Runs a trained policy in the OpenCabinet environment and reports
success rate across multiple episodes and kitchen scenes.

Usage:
    # Evaluate the simplified diffusion policy from Step 6
    python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Evaluate with more episodes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --num_rollouts 50

    # Evaluate on target (held-out) kitchen scenes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --split target

    # Save evaluation videos
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --video_path /tmp/eval_videos.mp4

For evaluating official Diffusion Policy / pi-0 / GR00T checkpoints,
use the evaluation scripts from those repos instead (see 06_train_policy.py).
"""

import argparse
import os
import sys
import time

# Force osmesa (CPU offscreen renderer) on Linux/WSL2 -- EGL requires
# /dev/dri device access that is unavailable in WSL environments.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_policy(checkpoint_path, device):
    """Load a trained policy checkpoint."""
    import torch
    import torch.nn as nn

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]

    model_type = checkpoint.get("model_type", "mlp")

    if model_type == "diffusion_unet1d":
        from diffusion_unet1d import DiffusionScheduler, UNet1D

        model = UNet1D(
            action_dim=action_dim,
            cond_dim=state_dim,
            base_channels=checkpoint.get("unet_channels", 64),
            channel_mults=tuple(checkpoint.get("unet_channel_mults", (1, 2, 4))),
            time_embed_dim=checkpoint.get("time_embed_dim", 128),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        scheduler = DiffusionScheduler(
            num_steps=checkpoint.get("diffusion_steps", 50),
            beta_start=checkpoint.get("beta_start", 1e-4),
            beta_end=checkpoint.get("beta_end", 0.02),
            device=device,
        )

        policy = {
            "type": "diffusion",
            "model": model,
            "scheduler": scheduler,
            "state_mean": checkpoint.get("state_mean", None),
            "state_std": checkpoint.get("state_std", None),
            "action_mean": checkpoint.get("action_mean", None),
            "action_std": checkpoint.get("action_std", None),
            "normalize_state": checkpoint.get("normalize_state", True),
            "normalize_action": checkpoint.get("normalize_action", True),
            "state_keys": checkpoint.get("state_keys", None),
        }
    else:

        class SimplePolicy(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim=256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                    nn.Tanh(),
                )

            def forward(self, state):
                return self.net(state)

        model = SimplePolicy(state_dim, action_dim).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        policy = {
            "type": "mlp",
            "model": model,
            "state_keys": checkpoint.get("state_keys", None),
        }

    print(f"Loaded policy from: {checkpoint_path}")
    print(f"  Trained for {checkpoint['epoch']} epochs, loss={checkpoint['loss']:.6f}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    print(f"  Model type: {model_type}")

    return policy, state_dim, action_dim


def _strip_obs_prefix(key):
    return key[len("observation.") :] if key.startswith("observation.") else key


def _joint_name_to_id(model, joint_name):
    if hasattr(model, "joint_name2id"):
        return model.joint_name2id(joint_name)
    for jidx in range(model.njnt):
        if model.joint(jidx).name == joint_name:
            return jidx
    return None


def compute_handle_features(env):
    """
    Compute handle features on-the-fly to match augmented dataset columns.
    Returns dict with keys:
      handle_pos, handle_to_eef_pos, door_openness, handle_xaxis, hinge_direction
    """
    fxtr = getattr(env, "fxtr", None)
    if fxtr is None:
        return {}

    model = env.sim.model
    data = env.sim.data
    fxtr_name = getattr(fxtr, "name", "")

    handle_bodies = []
    for i in range(model.nbody):
        name = model.body(i).name
        if fxtr_name in name and "handle" in name:
            handle_bodies.append(name)

    if not handle_bodies:
        return {}

    eef_pos = data.body("gripper0_right_eef").xpos.copy()
    dists = [np.linalg.norm(data.body(hb).xpos - eef_pos) for hb in handle_bodies]
    target_handle = handle_bodies[int(np.argmin(dists))]

    handle_pos = data.body(target_handle).xpos.copy()
    handle_to_eef = handle_pos - eef_pos
    handle_xaxis = data.body(target_handle).xmat.copy()[:3]

    joint_names = getattr(fxtr, "door_joint_names", [])
    if joint_names:
        joint_state = fxtr.get_joint_state(env, joint_names)
        door_openness = float(np.mean([joint_state[j] for j in joint_names]))
        jidx = _joint_name_to_id(model, joint_names[0])
        if jidx is not None:
            jmin, jmax = model.jnt_range[jidx]
            hinge_direction = 1.0 if abs(jmin) < abs(jmax) else -1.0
        else:
            hinge_direction = 0.0
    else:
        door_openness = 0.0
        hinge_direction = 0.0

    return {
        "handle_pos": handle_pos.astype(np.float32),
        "handle_to_eef_pos": handle_to_eef.astype(np.float32),
        "door_openness": np.array([door_openness], dtype=np.float32),
        "handle_xaxis": handle_xaxis.astype(np.float32),
        "hinge_direction": np.array([hinge_direction], dtype=np.float32),
    }


def compute_lerobot_state(obs):
    """
    Reconstruct the 16-dim LeRobot observation.state vector.
    Falls back to common PandaOmron keys if robocasa utils aren't available.
    """
    default_state_keys = [
        ("robot0_base_pos", 3),
        ("robot0_base_quat", 4),
        ("robot0_base_to_eef_pos", 3),
        ("robot0_base_to_eef_quat", 4),
        ("robot0_gripper_qpos", 2),
    ]
    try:
        from robocasa.utils.lerobot_utils import LEROBOT_STATE_TO_HDF5_STATE

        state_keys = [(k, None) for k in LEROBOT_STATE_TO_HDF5_STATE.values()]
    except Exception:
        state_keys = default_state_keys

    parts = []
    for key, expected_dim in state_keys:
        if key in obs and isinstance(obs[key], np.ndarray):
            parts.append(obs[key].flatten())
        else:
            if expected_dim is None:
                # Fall back to default sizes if available
                expected_dim = dict(default_state_keys).get(key, 1)
            parts.append(np.zeros(expected_dim, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def extract_state(obs, state_dim, state_keys=None, env=None):
    """Extract a fixed-size state vector from observations."""
    state_parts = []

    if state_keys:
        computed = None
        fallback_state = None
        for key in state_keys:
            obs_key = _strip_obs_prefix(key)
            if obs_key in obs and isinstance(obs[obs_key], np.ndarray):
                state_parts.append(obs[obs_key].flatten())
                continue
            if obs_key == "state":
                if fallback_state is None:
                    fallback_state = compute_lerobot_state(obs)
                state_parts.append(fallback_state)
                continue
            if env is not None:
                if computed is None:
                    computed = compute_handle_features(env)
                if obs_key in computed:
                    state_parts.append(computed[obs_key].flatten())
        if not state_parts:
            return np.zeros(state_dim, dtype=np.float32)
    else:
        # Gather available state observations in a consistent order
        state_keys = sorted(
            k
            for k in obs.keys()
            if not k.endswith("_image") and isinstance(obs[k], np.ndarray)
        )
        for key in state_keys:
            state_parts.append(obs[key].flatten())

    state = np.concatenate(state_parts).astype(np.float32)

    # Pad or truncate to match expected state_dim
    if len(state) < state_dim:
        state = np.pad(state, (0, state_dim - len(state)))
    elif len(state) > state_dim:
        state = state[:state_dim]

    return state


def _normalize(x, mean, std, enabled):
    if mean is None or std is None or not enabled:
        return x
    return (x - mean) / std


def _unnormalize(x, mean, std, enabled):
    if mean is None or std is None or not enabled:
        return x
    return x * std + mean


def sample_diffusion_action(policy, state, device, deterministic=True):
    """Sample an action from a diffusion policy."""
    import torch

    state_mean = policy["state_mean"]
    state_std = policy["state_std"]
    action_mean = policy["action_mean"]
    action_std = policy["action_std"]

    state_norm = _normalize(
        state, state_mean, state_std, policy.get("normalize_state", True)
    )
    state_tensor = torch.from_numpy(state_norm).unsqueeze(0).to(device)

    model = policy["model"]
    scheduler = policy["scheduler"]
    action_dim = model.action_dim

    x = torch.randn(1, action_dim, device=device)
    for t in reversed(range(scheduler.num_steps)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        x = scheduler.p_sample(
            model, x, t_tensor, state_tensor, noise=not deterministic
        )

    action = x.squeeze(0).cpu().numpy()
    action = _unnormalize(
        action, action_mean, action_std, policy.get("normalize_action", True)
    )
    return action


def check_one_door_open_success(env, threshold=0.90):
    fxtr = getattr(env, "fxtr", None)
    if fxtr is None or not hasattr(fxtr, "door_joint_names"):
        return env._check_success()
    joint_names = fxtr.door_joint_names
    if not joint_names:
        return env._check_success()
    joint_state = fxtr.get_joint_state(env, joint_names)
    for j_name in joint_names:
        if joint_state.get(j_name, 0.0) >= threshold:
            return True
    return False


def run_evaluation(
    policy,
    state_dim,
    action_dim,
    num_rollouts,
    max_steps,
    split,
    video_path,
    seed,
    deterministic,
    action_reorder,
):
    """Run evaluation rollouts and collect statistics."""
    import torch
    import imageio

    if policy["type"] == "diffusion":
        device = next(policy["model"].parameters()).device
    else:
        device = next(policy["model"].parameters()).device

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )

    lerobot_dir = None
    if action_reorder:
        try:
            from pathlib import Path
            from robocasa.utils.dataset_registry_utils import get_ds_path
            from robocasa.utils.lerobot_utils import reorder_lerobot_action

            ds_path = get_ds_path("OpenCabinet", source="human")
            if ds_path:
                if os.path.exists(os.path.join(ds_path, "lerobot")):
                    lerobot_dir = Path(os.path.join(ds_path, "lerobot"))
                elif os.path.exists(os.path.join(ds_path, "meta")):
                    lerobot_dir = Path(ds_path)
        except Exception:
            lerobot_dir = None

    video_writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {
        "successes": [],
        "episode_lengths": [],
        "rewards": [],
    }

    for ep in range(num_rollouts):
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")

        ep_reward = 0.0
        success = False

        for step in range(max_steps):
            # Extract state and predict action
            state = extract_state(
                obs, state_dim, state_keys=policy.get("state_keys"), env=env
            )
            with torch.no_grad():
                if policy["type"] == "diffusion":
                    action = sample_diffusion_action(
                        policy, state, device, deterministic=deterministic
                    )
                else:
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    action = policy["model"](state_tensor).cpu().numpy().squeeze(0)

            # Pad action to match environment action dim if needed
            env_action_dim = env.action_dim
            if len(action) < env_action_dim:
                action = np.pad(action, (0, env_action_dim - len(action)))
            elif len(action) > env_action_dim:
                action = action[:env_action_dim]

            # Reorder from LeRobot action layout to HDF5 / env layout if possible
            if lerobot_dir is not None:
                action = reorder_lerobot_action(action[None, :], lerobot_dir)[0]

            # Clip to action bounds
            low, high = env.action_spec
            action = np.clip(action, low, high)

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(frame)

            if check_one_door_open_success(env):
                success = True
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
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenCabinet policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (.pt file)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
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
        "--stochastic",
        action="store_true",
        help="Use stochastic diffusion sampling (default is deterministic)",
    )
    parser.add_argument(
        "--no_action_reorder",
        action="store_true",
        help="Disable LeRobot->HDF5 action reordering",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Policy Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the trained policy
    policy, state_dim, action_dim = load_policy(args.checkpoint, device)

    # Run evaluation
    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")

    results = run_evaluation(
        policy=policy,
        state_dim=state_dim,
        action_dim=action_dim,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
        deterministic=not args.stochastic,
        action_reorder=not args.no_action_reorder,
    )

    # Print summary
    print_section("Evaluation Results")

    num_success = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100
    avg_length = np.mean(results["episode_lengths"])
    avg_reward = np.mean(results["rewards"])

    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")
    print("  Success rule:   at least one cabinet door open")

    if args.video_path:
        print(f"\n  Video saved to: {args.video_path}")

    # Context for expected performance
    print_section("Performance Context")
    print(
        "Expected success rates from the RoboCasa benchmark:\n"
        "\n"
        "  Method            | Pretrain | Target\n"
        "  ------------------|----------|-------\n"
        "  Random actions    |    ~0%   |   ~0%\n"
        "  Diffusion Policy  |  ~30-60% | ~20-50%\n"
        "  pi-0              |  ~40-70% | ~30-60%\n"
        "  GR00T N1.5        |  ~35-65% | ~25-55%\n"
        "\n"
        "Note: The simplified 1D U-Net diffusion policy from Step 6 is\n"
        "not expected to achieve meaningful success rates. Use the\n"
        "official Diffusion Policy repo for real results."
    )


if __name__ == "__main__":
    main()
