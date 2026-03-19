"""
Minimal Step 6: Train a state-based diffusion policy for OpenCabinet
===================================================================

This is a compact diffusion-policy adaptation for the CS 188 starter project.

Key ideas:
- condition on current low-dimensional robot state
- predict a short horizon of future actions
- corrupt clean action chunks with Gaussian noise
- train the network to predict the added noise
- at inference, iteratively denoise from Gaussian noise into an action chunk

This is intentionally much smaller and simpler than the official diffusion
policy repo, but it captures the core logic.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import yaml

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

PACKED_STATE_COLUMNS = [
    "observation.state",
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
    raise FileNotFoundError(f"Could not find LeRobot data directory under {dataset_path}")


def flatten_value(val) -> List[float]:
    if isinstance(val, np.ndarray):
        return val.astype(np.float32).reshape(-1).tolist()
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=np.float32).reshape(-1).tolist()
    if isinstance(val, (int, float, np.integer, np.floating)):
        return [float(val)]
    return []


def find_state_columns(df_columns: List[str]) -> Dict[str, str]:
    mapping = {}
    for key in STATE_KEYS:
        exact = [c for c in df_columns if c == key or c.endswith(f".{key}") or c.endswith(key)]
        if exact:
            mapping[key] = sorted(exact)[0]
            continue
        fuzzy = [c for c in df_columns if key in c]
        if fuzzy:
            mapping[key] = sorted(fuzzy)[0]

    missing = [k for k in STATE_KEYS if k not in mapping]
    if missing:
        raise ValueError(
            f"Missing required state columns: {missing}\n"
            f"Available columns sample: {df_columns[:30]}"
        )
    return mapping


def find_packed_state_column(df_columns: List[str]) -> str | None:
    for col in PACKED_STATE_COLUMNS:
        if col in df_columns:
            return col
    for col in sorted(df_columns):
        if col.startswith("observation.state"):
            return col
    return None


def find_action_columns(df_columns: List[str]) -> List[str]:
    if "action" in df_columns:
        return ["action"]

    action_cols = [c for c in df_columns if c.startswith("action.")]
    if not action_cols:
        action_cols = [c for c in df_columns if c.startswith("action")]
    action_cols = sorted(set(action_cols))

    if not action_cols:
        raise ValueError("Could not find action columns in parquet file.")
    return action_cols


@dataclass
class EpisodeData:
    states: np.ndarray
    actions: np.ndarray
    source_file: str


class CabinetEpisodeLoader:
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
        self.packed_state_column = None
        self.state_format = None
        self.action_columns = None
        self.total_steps = 0

        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            if len(df) == 0:
                continue

            if self.state_columns is None and self.packed_state_column is None:
                packed_state_column = find_packed_state_column(list(df.columns))
                if packed_state_column is not None:
                    self.packed_state_column = packed_state_column
                    self.state_format = "packed_observation_state"
                else:
                    self.state_columns = find_state_columns(list(df.columns))
                    self.state_format = "structured_state_keys"
            if self.action_columns is None:
                self.action_columns = find_action_columns(list(df.columns))

            states = []
            actions = []
            for _, row in df.iterrows():
                if self.packed_state_column is not None:
                    state_parts = flatten_value(row[self.packed_state_column])
                else:
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
            raise RuntimeError("No usable episodes were loaded from the dataset.")

        self.state_dim = int(self.episodes[0].states.shape[-1])
        self.action_dim = int(self.episodes[0].actions.shape[-1])


def split_episodes(episodes: List[EpisodeData], val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)

    n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) > 1 else 0
    val_idx = set(idx[:n_val].tolist())
    train_eps = [ep for i, ep in enumerate(episodes) if i not in val_idx]
    val_eps = [ep for i, ep in enumerate(episodes) if i in val_idx]

    if not train_eps:
        train_eps = val_eps
        val_eps = []
    return train_eps, val_eps


def compute_normalization(episodes: List[EpisodeData]):
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


class DiffusionSequenceDataset:
    """
    Each sample:
      input  = normalized current state at time t
      target = normalized future action chunk [a_t, ..., a_{t+pred_horizon-1}]
    """

    def __init__(
        self,
        episodes: List[EpisodeData],
        pred_horizon: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        self.pred_horizon = int(pred_horizon)
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std = action_std.astype(np.float32)

        inputs = []
        targets = []

        for ep in episodes:
            T = len(ep.states)
            if T < self.pred_horizon:
                continue

            for t in range(T - self.pred_horizon + 1):
                state = ep.states[t]
                action_chunk = ep.actions[t : t + self.pred_horizon]
                inputs.append(state)
                targets.append(action_chunk)

        if not inputs:
            raise RuntimeError(
                "No training samples created. Try lowering --pred_horizon or loading more episodes."
            )

        self.inputs = np.stack(inputs).astype(np.float32)
        self.targets = np.stack(targets).astype(np.float32)

        self.inputs = (self.inputs - self.state_mean) / self.state_std
        self.targets = (self.targets - self.action_mean[None, :]) / self.action_std[None, :]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        import torch

        return {
            "state": torch.from_numpy(self.inputs[idx]),               # (state_dim,)
            "action": torch.from_numpy(self.targets[idx]),             # (pred_horizon, action_dim)
        }


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
    """
    Minimal diffusion backbone:
      input  = [flattened noisy action chunk, normalized state, time embedding]
      output = predicted noise with same shape as flattened action chunk
    """
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
            import torch

            # noisy_actions: (B, pred_horizon, action_dim)
            B = noisy_actions.shape[0]
            x = noisy_actions.reshape(B, -1)
            t_emb = self.time_embed(timesteps, torch)
            t_emb = self.time_proj(t_emb)
            h = torch.cat([x, cond_state, t_emb], dim=-1)
            out = self.net(h)
            return out.reshape_as(noisy_actions)

    return DiffusionMLP()


def evaluate_val_loss(model, loader, noise_scheduler, device):
    import torch
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            B = state.shape[0]

            noise = torch.randn_like(action)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device,
            ).long()

            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)
            noise_pred = model(noisy_action, timesteps, state)
            loss = F.mse_loss(noise_pred, noise, reduction="sum")

            total_loss += float(loss.item())
            total_items += int(B)

    return total_loss / max(total_items, 1)


def train_diffusion_policy(config: Dict):
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except ImportError as e:
        print("ERROR: Missing dependencies.")
        print("Install with: pip install torch diffusers")
        raise e

    print_section("Minimal Diffusion Policy Training")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    builder = CabinetEpisodeLoader(
        dataset_path=dataset_path,
        max_episodes=config["max_episodes"],
    )

    print(f"Loaded episodes: {len(builder.episodes)}")
    print(f"Total steps:      {builder.total_steps}")
    print(f"State dim:        {builder.state_dim}")
    print(f"Action dim:       {builder.action_dim}")
    print(f"State format:     {builder.state_format}")
    print(f"State keys:       {STATE_KEYS if builder.state_columns is not None else [builder.packed_state_column]}")
    print(f"Action columns:   {builder.action_columns}")

    train_eps, val_eps = split_episodes(
        builder.episodes,
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )
    print(f"Train episodes:   {len(train_eps)}")
    print(f"Val episodes:     {len(val_eps)}")

    state_mean, state_std, action_mean, action_std = compute_normalization(train_eps)

    train_ds = DiffusionSequenceDataset(
        train_eps,
        pred_horizon=config["pred_horizon"],
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    val_ds = (
        DiffusionSequenceDataset(
            val_eps,
            pred_horizon=config["pred_horizon"],
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
    print(f"Device:           {device}")

    model = build_diffusion_mlp(
        state_dim=builder.state_dim,
        action_dim=builder.action_dim,
        pred_horizon=config["pred_horizon"],
        hidden_dim=config["hidden_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_steps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_diffusion_policy.pt")
    final_path = os.path.join(checkpoint_dir, "final_diffusion_policy.pt")

    best_metric = float("inf")

    print_section("Hyperparameters")
    print(f"Epochs:              {config['epochs']}")
    print(f"Batch size:          {config['batch_size']}")
    print(f"LR:                  {config['learning_rate']}")
    print(f"Pred horizon:        {config['pred_horizon']}")
    print(f"Action horizon:      {config['action_horizon']}")
    print(f"Diffusion steps:     {config['num_diffusion_steps']}")
    print(f"Hidden dim:          {config['hidden_dim']}")
    print(f"Weight decay:        {config['weight_decay']}")

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            state = batch["state"].to(device)      # (B, state_dim)
            action = batch["action"].to(device)    # (B, pred_horizon, action_dim)
            B = state.shape[0]

            noise = torch.randn_like(action)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device,
            ).long()

            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)
            noise_pred = model(noisy_action, timesteps, state)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * B
            seen += B

        train_loss = running_loss / max(seen, 1)
        val_loss = (
            evaluate_val_loss(model, val_loader, noise_scheduler, device)
            if val_loader is not None
            else train_loss
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:4d}/{config['epochs']}  "
                f"train={train_loss:.6f}  val={val_loss:.6f}"
            )

        metric = val_loss if val_loader is not None else train_loss
        if metric < best_metric:
            best_metric = metric
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": metric,
                    "metric_name": "val_loss" if val_loader is not None else "train_loss",
                    "state_keys": STATE_KEYS,
                    "state_format": builder.state_format,
                    "packed_state_column": builder.packed_state_column,
                    "action_columns": builder.action_columns,
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "model_config": {
                        "state_dim": builder.state_dim,
                        "action_dim": builder.action_dim,
                        "pred_horizon": config["pred_horizon"],
                        "action_horizon": config["action_horizon"],
                        "hidden_dim": config["hidden_dim"],
                        "num_diffusion_steps": config["num_diffusion_steps"],
                    },
                },
                best_path,
            )

    torch.save(
        {
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
            "metric_name": "final_train_loss",
            "state_keys": STATE_KEYS,
            "state_format": builder.state_format,
            "packed_state_column": builder.packed_state_column,
            "action_columns": builder.action_columns,
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "model_config": {
                "state_dim": builder.state_dim,
                "action_dim": builder.action_dim,
                "pred_horizon": config["pred_horizon"],
                "action_horizon": config["action_horizon"],
                "hidden_dim": config["hidden_dim"],
                "num_diffusion_steps": config["num_diffusion_steps"],
            },
        },
        final_path,
    )

    print("\nTraining complete.")
    print(f"Best checkpoint:  {best_path}")
    print(f"Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a minimal diffusion policy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="/tmp/cabinet_policy_checkpoints")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=107)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--pred_horizon", type=int, default=4)
    parser.add_argument("--action_horizon", type=int, default=2)
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
            "pred_horizon": args.pred_horizon,
            "action_horizon": args.action_horizon,
            "num_diffusion_steps": args.num_diffusion_steps,
            "hidden_dim": args.hidden_dim,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        }

    train_diffusion_policy(config)


if __name__ == "__main__":
    main()
