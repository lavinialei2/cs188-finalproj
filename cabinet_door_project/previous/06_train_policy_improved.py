"""
Improved Step 6: Train an OpenCabinet policy
============================================

This version keeps the starter-project spirit but fixes several issues in the
basic baseline:
- uses an explicit, shared low-dimensional state specification
- normalizes states and actions
- splits episodes into train / validation sets
- saves the full preprocessing metadata into the checkpoint
- optionally predicts action chunks (K-step open-loop execution)

The goal is to provide a stronger, debuggable baseline entirely within the
starter-code framework.
"""

import argparse
import os
import sys
import yaml
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Force osmesa on Linux / WSL2 for consistency with the eval script.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


STATE_KEYS = [
    "robot0_gripper_qpos",
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
]


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path() -> str:
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


def resolve_data_dir(dataset_path: str) -> str:
    candidates = [
        os.path.join(dataset_path, "data"),
        os.path.join(dataset_path, "lerobot", "data"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find a LeRobot data directory under: {dataset_path}"
    )


def flatten_value(val) -> List[float]:
    if isinstance(val, np.ndarray):
        return val.astype(np.float32).reshape(-1).tolist()
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=np.float32).reshape(-1).tolist()
    if isinstance(val, (int, float, np.floating, np.integer)):
        return [float(val)]
    return []


def find_state_columns(df_columns: List[str]) -> Dict[str, str]:
    mapping = {}
    for key in STATE_KEYS:
        exact_candidates = [
            c for c in df_columns if c == key or c.endswith(f".{key}") or c.endswith(key)
        ]
        if exact_candidates:
            mapping[key] = sorted(exact_candidates)[0]
            continue

        fuzzy_candidates = [c for c in df_columns if key in c]
        if fuzzy_candidates:
            mapping[key] = sorted(fuzzy_candidates)[0]

    missing = [k for k in STATE_KEYS if k not in mapping]
    if missing:
        raise ValueError(
            "Could not find all required state columns. "
            f"Missing: {missing}\nAvailable columns example: {df_columns[:25]}"
        )
    return mapping


def find_action_columns(df_columns: List[str]) -> List[str]:
    if "action" in df_columns:
        return ["action"]

    action_cols = [c for c in df_columns if c.startswith("action.") or c.startswith("action")]
    action_cols = sorted(set(action_cols))
    if not action_cols:
        raise ValueError("Could not find action columns in parquet file.")
    return action_cols


@dataclass
class EpisodeData:
    states: np.ndarray
    actions: np.ndarray
    source_file: str


class CabinetSequenceBuilder:
    def __init__(self, dataset_path: str, max_episodes: int = None):
        import pyarrow.parquet as pq

        data_dir = resolve_data_dir(dataset_path)
        chunk_dir = os.path.join(data_dir, "chunk-000")
        if not os.path.exists(chunk_dir):
            raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

        parquet_files = sorted(
            os.path.join(chunk_dir, f)
            for f in os.listdir(chunk_dir)
            if f.endswith(".parquet")
        )
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {chunk_dir}")

        if max_episodes is not None:
            parquet_files = parquet_files[:max_episodes]

        self.episodes: List[EpisodeData] = []
        self.state_columns = None
        self.action_columns = None
        self.total_steps = 0

        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            if len(df) == 0:
                continue

            if self.state_columns is None:
                self.state_columns = find_state_columns(list(df.columns))
            if self.action_columns is None:
                self.action_columns = find_action_columns(list(df.columns))

            states = []
            actions = []
            for _, row in df.iterrows():
                state_parts = []
                for key in STATE_KEYS:
                    state_parts.extend(flatten_value(row[self.state_columns[key]]))

                action_parts = []
                for col in self.action_columns:
                    action_parts.extend(flatten_value(row[col]))

                if state_parts and action_parts:
                    states.append(np.asarray(state_parts, dtype=np.float32))
                    actions.append(np.asarray(action_parts, dtype=np.float32))

            if states and actions:
                states_arr = np.stack(states)
                actions_arr = np.stack(actions)
                self.episodes.append(EpisodeData(states_arr, actions_arr, pf))
                self.total_steps += len(states_arr)

        if not self.episodes:
            raise RuntimeError("No usable state-action episodes were loaded from the dataset.")

        self.state_dim = int(self.episodes[0].states.shape[-1])
        self.action_dim = int(self.episodes[0].actions.shape[-1])


class ChunkedEpisodeDataset:
    def __init__(
        self,
        episodes: List[EpisodeData],
        chunk_size: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        self.chunk_size = int(chunk_size)
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std = action_std.astype(np.float32)

        self.inputs = []
        self.targets = []
        for ep in episodes:
            T = len(ep.states)
            if T < self.chunk_size:
                continue
            for t in range(T - self.chunk_size + 1):
                state = ep.states[t]
                action_chunk = ep.actions[t : t + self.chunk_size].reshape(-1)
                self.inputs.append(state)
                self.targets.append(action_chunk)

        if not self.inputs:
            raise RuntimeError(
                "No chunked training samples were created. Try reducing --chunk_size "
                "or loading more episodes."
            )

        self.inputs = np.stack(self.inputs).astype(np.float32)
        self.targets = np.stack(self.targets).astype(np.float32)

        chunk_action_mean = np.tile(self.action_mean, self.chunk_size)
        chunk_action_std = np.tile(self.action_std, self.chunk_size)

        self.inputs = (self.inputs - self.state_mean) / self.state_std
        self.targets = (self.targets - chunk_action_mean) / chunk_action_std

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        import torch

        return (
            torch.from_numpy(self.inputs[idx]),
            torch.from_numpy(self.targets[idx]),
        )


class PolicyNetConfig:
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, hidden_dim: int):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.hidden_dim = int(hidden_dim)

    @property
    def output_dim(self) -> int:
        return self.action_dim * self.chunk_size

    def to_dict(self) -> Dict:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "hidden_dim": self.hidden_dim,
        }


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


def compute_normalization(episodes: List[EpisodeData]) -> Tuple[np.ndarray, ...]:
    all_states = np.concatenate([ep.states for ep in episodes], axis=0)
    all_actions = np.concatenate([ep.actions for ep in episodes], axis=0)

    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0)
    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0)

    state_std = np.where(state_std < 1e-6, 1.0, state_std)
    action_std = np.where(action_std < 1e-6, 1.0, action_std)
    return (
        state_mean.astype(np.float32),
        state_std.astype(np.float32),
        action_mean.astype(np.float32),
        action_std.astype(np.float32),
    )


def split_episodes(episodes: List[EpisodeData], val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    n_val = max(1, int(round(len(indices) * val_ratio))) if len(indices) > 1 else 0
    val_idx = set(indices[:n_val].tolist())
    train_eps = [ep for i, ep in enumerate(episodes) if i not in val_idx]
    val_eps = [ep for i, ep in enumerate(episodes) if i in val_idx]

    if not train_eps:
        train_eps = val_eps
        val_eps = []
    return train_eps, val_eps


def evaluate_loader(model, loader, device, nn):
    model.eval()
    total_loss = 0.0
    total_items = 0
    with __import__("torch").no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb, reduction="sum")
            total_loss += float(loss.item())
            total_items += int(xb.shape[0])
    return total_loss / max(total_items, 1)


def train_improved_policy(config: Dict):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    print_section("Improved Behavior Cloning Policy")
    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    builder = CabinetSequenceBuilder(
        dataset_path=dataset_path,
        max_episodes=config.get("max_episodes"),
    )

    print(f"Loaded {len(builder.episodes)} episodes / {builder.total_steps} steps")
    print(f"State dim:  {builder.state_dim}")
    print(f"Action dim: {builder.action_dim}")
    print(f"State keys: {STATE_KEYS}")
    print(f"Action cols: {builder.action_columns}")

    train_eps, val_eps = split_episodes(
        builder.episodes,
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )
    print(f"Train episodes: {len(train_eps)}")
    print(f"Val episodes:   {len(val_eps)}")

    state_mean, state_std, action_mean, action_std = compute_normalization(train_eps)

    train_ds = ChunkedEpisodeDataset(
        train_eps,
        chunk_size=config["chunk_size"],
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    val_ds = (
        ChunkedEpisodeDataset(
            val_eps,
            chunk_size=config["chunk_size"],
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
        )
        if val_eps
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, drop_last=False)
        if val_ds is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net_cfg = PolicyNetConfig(
        state_dim=builder.state_dim,
        action_dim=builder.action_dim,
        chunk_size=config["chunk_size"],
        hidden_dim=config["hidden_dim"],
    )
    model = build_model(net_cfg, nn).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config["epochs"], 1),
        eta_min=config["learning_rate"] * 0.1,
    )

    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print_section("Training")
    print(f"Epochs:       {config['epochs']}")
    print(f"Batch size:   {config['batch_size']}")
    print(f"LR:           {config['learning_rate']}")
    print(f"Chunk size:   {config['chunk_size']}")
    print(f"Hidden dim:   {config['hidden_dim']}")
    print(f"Weight decay: {config['weight_decay']}")

    best_val = math.inf
    best_train = math.inf
    best_path = os.path.join(checkpoint_dir, "best_policy.pt")
    final_path = os.path.join(checkpoint_dir, "final_policy.pt")

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = int(xb.shape[0])
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_loss = evaluate_loader(model, val_loader, device, nn) if val_loader else train_loss

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:4d}/{config['epochs']}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        metric = val_loss if val_loader else train_loss
        if metric < (best_val if val_loader else best_train):
            if val_loader:
                best_val = metric
            else:
                best_train = metric
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metric_name": "val_loss" if val_loader else "train_loss",
                    "loss": metric,
                    "model_config": net_cfg.to_dict(),
                    "state_keys": STATE_KEYS,
                    "action_columns": builder.action_columns,
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "action_mean": action_mean,
                    "action_std": action_std,
                },
                best_path,
            )

    torch.save(
        {
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_name": "final_train_loss",
            "loss": train_loss,
            "model_config": net_cfg.to_dict(),
            "state_keys": STATE_KEYS,
            "action_columns": builder.action_columns,
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
        },
        final_path,
    )

    print("\nTraining complete!")
    if val_loader:
        print(f"Best validation loss: {best_val:.6f}")
    else:
        print(f"Best train loss:      {best_train:.6f}")
    print(f"Best checkpoint:      {best_path}")
    print(f"Final checkpoint:     {final_path}")


def print_diffusion_policy_instructions():
    print_section("Official Diffusion Policy Training")
    print(
        "For production-quality policy training, use the official repos:\n"
        "\n"
        "Option A: Diffusion Policy (recommended for single-task)\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Train an improved OpenCabinet policy")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="/tmp/cabinet_policy_checkpoints")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--use_diffusion_policy", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=150)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Improved Policy Training")
    print("=" * 60)

    if args.use_diffusion_policy:
        print_diffusion_policy_instructions()
        return

    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "checkpoint_dir": args.checkpoint_dir,
            "max_episodes": args.max_episodes,
            "val_ratio": args.val_ratio,
            "chunk_size": args.chunk_size,
            "hidden_dim": args.hidden_dim,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        }

    train_improved_policy(config)


if __name__ == "__main__":
    main()