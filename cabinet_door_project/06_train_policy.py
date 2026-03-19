"""
Step 6: Train a Diffusion Policy
==================================
This script provides a self-contained training loop for a small
1D U-Net diffusion policy on the OpenCabinet task, suitable for
understanding the training pipeline.

For production-quality training, use the official Diffusion Policy repo:
    git clone https://github.com/robocasa-benchmark/diffusion_policy
    cd diffusion_policy && pip install -e .
    python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/OpenCabinet

This simplified version trains a small 1D Conv U-Net diffusion policy
to demonstrate the data loading -> training -> checkpoint pipeline.

Usage:
    python 06_train_policy.py [--epochs 50] [--batch_size 32] [--lr 1e-4]
    python 06_train_policy.py --action_horizon 8 --ema_decay 0.999
    python 06_train_policy.py --use_diffusion_policy   # Use official repo
"""

import argparse
import os
import sys
import time
import yaml

import numpy as np


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path():
    """Get the path to the OpenCabinet dataset."""
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


def train_diffusion_policy(config):
    """
    Train a simple diffusion policy with a 1D Conv U-Net backbone.

    This is a simplified training loop to illustrate the pipeline.
    For real results, use the official Diffusion Policy codebase.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    from diffusion_unet1d import DiffusionScheduler, UNet1D

    print_section("1D U-Net Diffusion Policy")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    # ----------------------------------------------------------------
    # 1. Build a simple dataset from the LeRobot format
    # ----------------------------------------------------------------
    print("\nLoading dataset...")

    class CabinetDemoDataset(Dataset):
        """
        Loads state-action pairs from the LeRobot-format dataset.

        For simplicity, this uses only the low-dimensional state observations
        (gripper qpos, base pose, eef pose) rather than images.
        Full visuomotor training with images requires the Diffusion Policy repo.
        """

        def __init__(
            self,
            dataset_path,
            max_episodes=None,
            normalize_state=True,
            normalize_action=True,
            action_horizon=1,
        ):
            import pyarrow.parquet as pq

            self.states = []
            self.actions = []
            self.normalize_state = normalize_state
            self.normalize_action = normalize_action
            self.state_cols = None
            self.action_cols = None
            self.action_horizon = max(1, int(action_horizon))

            # Prefer augmented data if present
            aug_dir = os.path.join(dataset_path, "augmented")
            use_augmented = False
            if os.path.exists(aug_dir):
                aug_parquets = sorted(
                    f for f in os.listdir(aug_dir) if f.endswith(".parquet")
                )
                if aug_parquets:
                    chunk_dir = aug_dir
                    parquet_files = aug_parquets
                    use_augmented = True

            if not use_augmented:
                # The dataset path from get_ds_path may point to the lerobot dir directly
                # or to the parent. Try both layouts.
                data_dir = os.path.join(dataset_path, "data")
                if not os.path.exists(data_dir):
                    data_dir = os.path.join(dataset_path, "lerobot", "data")
                if not os.path.exists(data_dir):
                    raise FileNotFoundError(
                        f"Data directory not found under: {dataset_path}\n"
                        "Make sure you downloaded the dataset with 04_download_dataset.py"
                    )

                # Load parquet files
                chunk_dir = os.path.join(data_dir, "chunk-000")
                if not os.path.exists(chunk_dir):
                    raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

                parquet_files = sorted(
                    f for f in os.listdir(chunk_dir) if f.endswith(".parquet")
                )

            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {chunk_dir}")

            if use_augmented:
                print("Using augmented dataset with handle features.")

            episodes_loaded = 0
            for pf in parquet_files:
                table = pq.read_table(os.path.join(chunk_dir, pf))
                df = table.to_pandas()

                # Extract state and action columns
                obs_cols = [
                    c
                    for c in df.columns
                    if c.startswith("observation.") and not c.endswith("_image")
                ]
                state_cols = [c for c in obs_cols if c.startswith("observation.state")]
                extra_state_cols = [
                    c for c in obs_cols if not c.startswith("observation.state")
                ]
                action_cols = [
                    c for c in df.columns
                    if c == "action" or c.startswith("action.")
                ]

                if not state_cols or not action_cols:
                    # Try alternative column naming
                    state_cols = [
                        c
                        for c in df.columns
                        if "gripper" in c or "base" in c or "eef" in c
                    ]
                    action_cols = [c for c in df.columns if "action" in c]

                if state_cols and action_cols:
                    if state_cols:
                        state_cols = sorted(state_cols) + sorted(extra_state_cols)
                    else:
                        state_cols = sorted(obs_cols)
                    action_cols = sorted(action_cols)
                    if self.state_cols is None:
                        self.state_cols = state_cols
                    if self.action_cols is None:
                        self.action_cols = action_cols
                    states_seq = []
                    actions_seq = []
                    for _, row in df.iterrows():
                        # Values may be numpy arrays (object columns) or scalars
                        state_parts = []
                        for c in state_cols:
                            val = row[c]
                            if isinstance(val, np.ndarray):
                                state_parts.extend(val.flatten().tolist())
                            elif isinstance(val, (int, float, np.floating)):
                                state_parts.append(float(val))
                        action_parts = []
                        for c in action_cols:
                            val = row[c]
                            if isinstance(val, np.ndarray):
                                action_parts.extend(val.flatten().tolist())
                            elif isinstance(val, (int, float, np.floating)):
                                action_parts.append(float(val))

                        if state_parts and action_parts:
                            states_seq.append(np.array(state_parts, dtype=np.float32))
                            actions_seq.append(np.array(action_parts, dtype=np.float32))

                    if states_seq and actions_seq:
                        states_seq = np.stack(states_seq, axis=0)
                        actions_seq = np.stack(actions_seq, axis=0)
                        max_start = len(actions_seq) - self.action_horizon + 1
                        if max_start <= 0:
                            continue
                        for i in range(max_start):
                            self.states.append(states_seq[i])
                            action_chunk = actions_seq[
                                i : i + self.action_horizon
                            ].reshape(-1)
                            self.actions.append(action_chunk)

                episodes_loaded += 1
                if max_episodes and episodes_loaded >= max_episodes:
                    break

            if len(self.states) == 0:
                print("WARNING: Could not extract state-action pairs from parquet files.")
                print("The dataset may use a different format.")
                print("Generating synthetic demo data for illustration...")
                self._generate_synthetic_data()

            self.states = np.array(self.states, dtype=np.float32)
            self.actions = np.array(self.actions, dtype=np.float32)

            self.state_mean = self.states.mean(axis=0)
            self.state_std = self.states.std(axis=0)
            self.action_mean = self.actions.mean(axis=0)
            self.action_std = self.actions.std(axis=0)

            self.state_std = np.maximum(self.state_std, 1e-6)
            self.action_std = np.maximum(self.action_std, 1e-6)

            print(f"Loaded {len(self.states)} state-action pairs")
            print(f"State dim:  {self.states.shape[-1]}")
            print(f"Action dim: {self.actions.shape[-1]}")

        def _generate_synthetic_data(self):
            """Generate synthetic data for demonstration purposes."""
            rng = np.random.default_rng(42)
            for _ in range(1000):
                state = rng.standard_normal(16).astype(np.float32)
                action = rng.standard_normal(12).astype(np.float32) * 0.1
                self.states.append(state)
                self.actions.append(action)

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            state = self.states[idx]
            action = self.actions[idx]

            if self.normalize_state:
                state = (state - self.state_mean) / self.state_std
            if self.normalize_action:
                action = (action - self.action_mean) / self.action_std

            return torch.from_numpy(state), torch.from_numpy(action)

    dataset = CabinetDemoDataset(
        dataset_path,
        max_episodes=config.get("max_episodes", 50),
        normalize_state=config.get("normalize_state", True),
        normalize_action=config.get("normalize_action", True),
        action_horizon=config.get("action_horizon", 1),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # ----------------------------------------------------------------
    # 2. Define diffusion model + scheduler
    # ----------------------------------------------------------------
    state_dim = dataset.states.shape[-1]
    action_dim = dataset.actions.shape[-1]
    action_horizon = config.get("action_horizon", 1)
    action_dim_per_step = action_dim // max(1, action_horizon)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    def _parse_channel_mults(value):
        if isinstance(value, str):
            return tuple(int(x.strip()) for x in value.split(",") if x.strip())
        if isinstance(value, (list, tuple)):
            return tuple(int(x) for x in value)
        return (1, 2, 4)

    channel_mults = _parse_channel_mults(config.get("unet_channel_mults", (1, 2, 4)))

    model = UNet1D(
        action_dim=action_dim,
        cond_dim=state_dim,
        base_channels=config.get("unet_channels", 64),
        channel_mults=channel_mults,
        time_embed_dim=config.get("time_embed_dim", 128),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    ema_decay = config.get("ema_decay", 0.999)
    ema_model = UNet1D(
        action_dim=action_dim,
        cond_dim=state_dim,
        base_channels=config.get("unet_channels", 64),
        channel_mults=channel_mults,
        time_embed_dim=config.get("time_embed_dim", 128),
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False
    scheduler = DiffusionScheduler(
        num_steps=config.get("diffusion_steps", 50),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 0.02),
        device=device,
    )

    # ----------------------------------------------------------------
    # 3. Training loop
    # ----------------------------------------------------------------
    print_section("Training")
    start_time = time.time()
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"Start time: {start_ts}")
    print(f"Epochs:     {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"LR:         {config['learning_rate']}")
    print(f"Steps:      {scheduler.num_steps}")
    print(f"Horizon:    {action_horizon}")
    print(f"EMA decay:  {ema_decay}")

    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    avg_loss = float("inf")
    ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for states_batch, actions_batch in dataloader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            t = torch.randint(
                0, scheduler.num_steps, (states_batch.shape[0],), device=device
            )
            noise = torch.randn_like(actions_batch)
            noisy_actions = scheduler.q_sample(actions_batch, t, noise)

            pred_noise = model(noisy_actions, t, states_batch)
            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(
                f"  Epoch {epoch + 1:4d}/{config['epochs']}  "
                f"Loss: {avg_loss:.6f}  Time: {now_ts}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
            torch.save(
                {
                    "model_type": "diffusion_unet1d",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "diffusion_steps": scheduler.num_steps,
                    "beta_start": config.get("beta_start", 1e-4),
                    "beta_end": config.get("beta_end", 0.02),
                    "unet_channels": config.get("unet_channels", 64),
                    "unet_channel_mults": channel_mults,
                    "time_embed_dim": config.get("time_embed_dim", 128),
                    "normalize_state": config.get("normalize_state", True),
                    "normalize_action": config.get("normalize_action", True),
                    "state_mean": dataset.state_mean,
                    "state_std": dataset.state_std,
                    "action_mean": dataset.action_mean,
                    "action_std": dataset.action_std,
                    "state_keys": dataset.state_cols,
                    "action_horizon": action_horizon,
                    "action_dim_per_step": action_dim_per_step,
                    "ema_decay": ema_decay,
                    "ema_state_dict": ema_model.state_dict(),
                },
                ckpt_path,
            )

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final_policy.pt")
    torch.save(
        {
            "model_type": "diffusion_unet1d",
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "diffusion_steps": scheduler.num_steps,
            "beta_start": config.get("beta_start", 1e-4),
            "beta_end": config.get("beta_end", 0.02),
            "unet_channels": config.get("unet_channels", 64),
            "unet_channel_mults": channel_mults,
            "time_embed_dim": config.get("time_embed_dim", 128),
            "normalize_state": config.get("normalize_state", True),
            "normalize_action": config.get("normalize_action", True),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "action_mean": dataset.action_mean,
            "action_std": dataset.action_std,
            "state_keys": dataset.state_cols,
            "action_horizon": action_horizon,
            "action_dim_per_step": action_dim_per_step,
            "ema_decay": ema_decay,
            "ema_state_dict": ema_model.state_dict(),
        },
        final_path,
    )

    end_time = time.time()
    end_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    elapsed_min = (end_time - start_time) / 60.0
    print(f"\nTraining complete!")
    print(f"End time:         {end_ts}")
    print(f"Elapsed (min):    {elapsed_min:.1f}")
    print(f"Best loss:        {best_loss:.6f}")
    print(f"Best checkpoint:  {ckpt_path}")
    print(f"Final checkpoint: {final_path}")

    print_section("Next Steps")
    print(
        "This simple 1D U-Net diffusion policy is for educational purposes only.\n"
        "For a policy that can actually solve the task, use the\n"
        "official Diffusion Policy codebase:\n"
        "\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
        "\n"
        "Alternatively, try pi-0 or GR00T N1.5:\n"
        "  https://github.com/robocasa-benchmark/openpi\n"
        "  https://github.com/robocasa-benchmark/Isaac-GR00T"
    )


def print_diffusion_policy_instructions():
    """Print instructions for using the official Diffusion Policy repo."""
    print_section("Official Diffusion Policy Training")
    print(
        "For production-quality policy training, use the official repos:\n"
        "\n"
        "Option A: Diffusion Policy (recommended for single-task)\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "\n"
        "  # Train\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
        "\n"
        "  # Evaluate\n"
        "  python eval_robocasa.py \\\n"
        "    --checkpoint <path-to-checkpoint> \\\n"
        "    --task_set atomic \\\n"
        "    --split target\n"
        "\n"
        "Option B: pi-0 via OpenPi (for foundation model fine-tuning)\n"
        "  git clone https://github.com/robocasa-benchmark/openpi\n"
        "  cd openpi && pip install -e . && pip install -e packages/openpi-client/\n"
        "\n"
        "  XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \\\n"
        "    robocasa_OpenCabinet --exp-name=cabinet_door\n"
        "\n"
        "Option C: GR00T N1.5 (NVIDIA foundation model)\n"
        "  git clone https://github.com/robocasa-benchmark/Isaac-GR00T\n"
        "  cd groot && pip install -e .\n"
        "\n"
        "  python scripts/gr00t_finetune.py \\\n"
        "    --output-dir experiments/cabinet_door \\\n"
        "    --dataset_soup robocasa_OpenCabinet \\\n"
        "    --max_steps 50000\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Train a policy for OpenCabinet")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=50,
        help="Max episodes to load from dataset (<=0 for all)",
    )
    parser.add_argument(
        "--diffusion_steps", type=int, default=50, help="Diffusion timesteps"
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=8,
        help="Number of future actions to predict per step",
    )
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Beta start")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta end")
    parser.add_argument(
        "--unet_channels", type=int, default=64, help="Base channel size for U-Net"
    )
    parser.add_argument(
        "--unet_channel_mults",
        type=str,
        default="1,2,4",
        help="Comma-separated channel multipliers (e.g. 1,2,4)",
    )
    parser.add_argument(
        "--time_embed_dim", type=int, default=128, help="Time embedding dimension"
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay for model weights",
    )
    parser.add_argument(
        "--no_normalize_state",
        action="store_true",
        help="Disable state normalization",
    )
    parser.add_argument(
        "--no_normalize_action",
        action="store_true",
        help="Disable action normalization",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/cabinet_policy_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )
    parser.add_argument(
        "--use_diffusion_policy",
        action="store_true",
        help="Print instructions for using the official Diffusion Policy repo",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Training")
    print("=" * 60)

    if args.use_diffusion_policy:
        print_diffusion_policy_instructions()
        return

    # Build config from args or YAML file
    if args.config:
        config = load_config(args.config)
    else:
        unet_channel_mults = tuple(
            int(x.strip()) for x in args.unet_channel_mults.split(",") if x.strip()
        )
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "checkpoint_dir": args.checkpoint_dir,
            "diffusion_steps": args.diffusion_steps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "unet_channels": args.unet_channels,
            "unet_channel_mults": unet_channel_mults,
            "time_embed_dim": args.time_embed_dim,
            "normalize_state": not args.no_normalize_state,
            "normalize_action": not args.no_normalize_action,
            "max_episodes": None if args.max_episodes <= 0 else args.max_episodes,
            "action_horizon": args.action_horizon,
            "ema_decay": args.ema_decay,
        }

    train_diffusion_policy(config)


if __name__ == "__main__":
    main()
