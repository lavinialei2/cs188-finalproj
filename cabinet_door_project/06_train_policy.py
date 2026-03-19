"""
Step 6: Train a Minimal State-Based Diffusion Policy
===================================================

This script replaces the starter MLP behavior cloning baseline with a small,
self-contained diffusion-style policy for the RoboCasa OpenCabinet task.

Design goals:
- use low-dimensional robot state only (no images)
- load the existing LeRobot / RoboCasa parquet dataset
- predict a short action sequence instead of a single action
- train by adding Gaussian noise to clean action chunks and predicting the noise
- save checkpoints with enough metadata for iterative denoising at evaluation time

The training logic is inspired by the state-based diffusion notebook, but kept
compact and readable for a class project.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


@dataclass
class EpisodeData:
    states: np.ndarray
    actions: np.ndarray
    source_file: str


@dataclass
class NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def default_config() -> Dict:
    return {
        "epochs": 60,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "checkpoint_dir": "/tmp/cabinet_policy_checkpoints",
        "max_episodes": 107,
        "val_ratio": 0.1,
        "obs_horizon": 1,
        "pred_horizon": 4,
        "action_horizon": 2,
        "num_diffusion_steps": 50,
        "hidden_dim": 512,
        "weight_decay": 1e-6,
        "lr_warmup_steps": 100,
        "num_workers": 0,
        "seed": 0,
    }


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_config(config: Dict) -> None:
    if int(config["obs_horizon"]) < 1:
        raise ValueError("--obs_horizon must be at least 1")
    if int(config["pred_horizon"]) < 1:
        raise ValueError("--pred_horizon must be at least 1")
    if int(config["action_horizon"]) < 1:
        raise ValueError("--action_horizon must be at least 1")
    if int(config["action_horizon"]) > int(config["pred_horizon"]):
        raise ValueError("--action_horizon cannot exceed --pred_horizon")
    if int(config["num_diffusion_steps"]) < 1:
        raise ValueError("--num_diffusion_steps must be at least 1")
    if float(config["val_ratio"]) < 0.0 or float(config["val_ratio"]) >= 1.0:
        raise ValueError("--val_ratio must be in [0.0, 1.0)")


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
        f"Could not find LeRobot data directory under {dataset_path}. "
        "Run 04_download_dataset.py first."
    )


def flatten_value(val) -> List[float]:
    if isinstance(val, np.ndarray):
        return val.astype(np.float32).reshape(-1).tolist()
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=np.float32).reshape(-1).tolist()
    if isinstance(val, (int, float, np.integer, np.floating)):
        return [float(val)]
    return []


def find_state_columns(df_columns: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for key in STATE_KEYS:
        exact = [c for c in df_columns if c == key or c.endswith(f".{key}") or c.endswith(key)]
        if exact:
            mapping[key] = sorted(exact)[0]
            continue
        fuzzy = [c for c in df_columns if key in c]
        if fuzzy:
            mapping[key] = sorted(fuzzy)[0]

    missing = [key for key in STATE_KEYS if key not in mapping]
    if missing:
        raise ValueError(
            "Missing required structured state columns: "
            f"{missing}\nAvailable columns sample: {list(df_columns)[:40]}"
        )
    return mapping


def find_packed_state_column(df_columns: Sequence[str]) -> Optional[str]:
    for col in PACKED_STATE_COLUMNS:
        if col in df_columns:
            return col
    for col in sorted(df_columns):
        if col.startswith("observation.state"):
            return col
    return None


def find_action_columns(df_columns: Sequence[str]) -> List[str]:
    if "action" in df_columns:
        return ["action"]

    action_cols = [c for c in df_columns if c.startswith("action.")]
    if not action_cols:
        action_cols = [c for c in df_columns if c.startswith("action")]
    action_cols = sorted(set(action_cols))

    if not action_cols:
        raise ValueError(
            "Could not find action columns in parquet file. "
            f"Available columns sample: {list(df_columns)[:40]}"
        )
    return action_cols


class CabinetEpisodeLoader:
    def __init__(self, dataset_path: str, max_episodes: Optional[int] = None):
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
        self.total_steps = 0
        self.state_columns: Optional[Dict[str, str]] = None
        self.packed_state_column: Optional[str] = None
        self.state_format: Optional[str] = None
        self.action_columns: Optional[List[str]] = None

        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            if len(df) == 0:
                continue

            if self.state_format is None:
                structured_columns = None
                try:
                    structured_columns = find_state_columns(list(df.columns))
                except ValueError:
                    structured_columns = None

                if structured_columns is not None:
                    self.state_columns = structured_columns
                    self.state_format = "structured_state_keys"
                else:
                    packed_col = find_packed_state_column(list(df.columns))
                    if packed_col is None:
                        raise ValueError(
                            "Could not find either the required structured state keys or a "
                            "packed observation.state column.\n"
                            f"Available columns sample: {list(df.columns)[:40]}"
                        )
                    self.packed_state_column = packed_col
                    self.state_format = "packed_observation_state"

            if self.action_columns is None:
                self.action_columns = find_action_columns(list(df.columns))

            states: List[np.ndarray] = []
            actions: List[np.ndarray] = []
            for _, row in df.iterrows():
                if self.state_format == "packed_observation_state":
                    state_parts = flatten_value(row[self.packed_state_column])
                else:
                    state_parts = []
                    assert self.state_columns is not None
                    for key in STATE_KEYS:
                        state_parts.extend(flatten_value(row[self.state_columns[key]]))

                action_parts: List[float] = []
                assert self.action_columns is not None
                for col in self.action_columns:
                    action_parts.extend(flatten_value(row[col]))

                if not state_parts or not action_parts:
                    continue

                states.append(np.asarray(state_parts, dtype=np.float32))
                actions.append(np.asarray(action_parts, dtype=np.float32))

            if not states or not actions:
                continue

            states_arr = np.stack(states)
            actions_arr = np.stack(actions)
            self.episodes.append(
                EpisodeData(states=states_arr, actions=actions_arr, source_file=parquet_file)
            )
            self.total_steps += len(states_arr)

        if not self.episodes:
            raise RuntimeError("No usable episodes were loaded from the dataset.")

        self.state_dim = int(self.episodes[0].states.shape[-1])
        self.action_dim = int(self.episodes[0].actions.shape[-1])

        for episode in self.episodes:
            if episode.states.shape[-1] != self.state_dim:
                raise ValueError(f"Inconsistent state dimension in {episode.source_file}")
            if episode.actions.shape[-1] != self.action_dim:
                raise ValueError(f"Inconsistent action dimension in {episode.source_file}")


def split_episodes(
    episodes: Sequence[EpisodeData],
    val_ratio: float,
    seed: int,
) -> Tuple[List[EpisodeData], List[EpisodeData]]:
    if len(episodes) <= 1 or val_ratio <= 0.0:
        return list(episodes), []

    rng = np.random.default_rng(seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)

    n_val = max(1, int(round(len(idx) * val_ratio)))
    val_idx = set(idx[:n_val].tolist())

    train_eps = [ep for i, ep in enumerate(episodes) if i not in val_idx]
    val_eps = [ep for i, ep in enumerate(episodes) if i in val_idx]

    if not train_eps:
        train_eps = val_eps
        val_eps = []
    return train_eps, val_eps


def compute_normalization(episodes: Sequence[EpisodeData]) -> NormalizationStats:
    all_states = np.concatenate([ep.states for ep in episodes], axis=0)
    all_actions = np.concatenate([ep.actions for ep in episodes], axis=0)

    state_mean = all_states.mean(axis=0).astype(np.float32)
    state_std = all_states.std(axis=0).astype(np.float32)
    action_mean = all_actions.mean(axis=0).astype(np.float32)
    action_std = all_actions.std(axis=0).astype(np.float32)

    state_std = np.where(state_std < 1e-6, 1.0, state_std).astype(np.float32)
    action_std = np.where(action_std < 1e-6, 1.0, action_std).astype(np.float32)

    return NormalizationStats(
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )


def create_sample_indices(
    episodes: Sequence[EpisodeData],
    pred_horizon: int,
    obs_horizon: int,
    action_horizon: int,
) -> List[Tuple[int, int]]:
    indices: List[Tuple[int, int]] = []
    min_len = max(pred_horizon, obs_horizon)

    for ep_idx, episode in enumerate(episodes):
        T = int(episode.states.shape[0])
        if T < min_len:
            continue
        max_start = T - pred_horizon
        for start in range(max_start + 1):
            state_idx = start + obs_horizon - 1
            if state_idx >= T:
                break
            indices.append((ep_idx, start))

    if not indices:
        raise RuntimeError(
            "No training samples created. Try lowering --pred_horizon, lowering --obs_horizon, "
            "or loading more episodes."
        )
    return indices


class CabinetDiffusionDataset:
    """Returns normalized state windows and normalized future action chunks."""

    def __init__(
        self,
        episodes: Sequence[EpisodeData],
        norm: NormalizationStats,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
    ):
        self.episodes = list(episodes)
        self.norm = norm
        self.pred_horizon = int(pred_horizon)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.indices = create_sample_indices(
            self.episodes,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        import torch

        ep_idx, start = self.indices[idx]
        episode = self.episodes[ep_idx]

        state_start = start
        state_end = start + self.obs_horizon
        action_end = start + self.pred_horizon

        state_seq = episode.states[state_start:state_end]
        action_seq = episode.actions[start:action_end]

        if state_seq.shape[0] != self.obs_horizon:
            raise ValueError(
                f"State window length mismatch for episode {episode.source_file}: "
                f"expected {self.obs_horizon}, got {state_seq.shape[0]}"
            )
        if action_seq.shape[0] != self.pred_horizon:
            raise ValueError(
                f"Action chunk length mismatch for episode {episode.source_file}: "
                f"expected {self.pred_horizon}, got {action_seq.shape[0]}"
            )

        norm_state = (state_seq - self.norm.state_mean) / self.norm.state_std
        norm_action = (action_seq - self.norm.action_mean) / self.norm.action_std

        return {
            "obs": torch.from_numpy(norm_state.astype(np.float32)),
            "action": torch.from_numpy(norm_action.astype(np.float32)),
        }


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


def evaluate_val_loss(model, loader, noise_scheduler, device) -> float:
    import torch
    import torch.nn.functional as F

    model.eval()
    running_loss = 0.0
    seen = 0

    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            batch_size = obs.shape[0]

            noise = torch.randn_like(action)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()
            noisy_actions = noise_scheduler.add_noise(action, noise, timesteps)
            noise_pred = model(noisy_actions, timesteps, obs)
            loss = F.mse_loss(noise_pred, noise)

            running_loss += float(loss.item()) * batch_size
            seen += batch_size

    model.train()
    return running_loss / max(seen, 1)


def train_diffusion_policy(config: Dict) -> None:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.optimization import get_scheduler
    except ImportError:
        print("ERROR: Missing dependencies. Install with: pip install torch diffusers pyarrow")
        sys.exit(1)

    print_section("Minimal Diffusion Policy Training")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    builder = CabinetEpisodeLoader(
        dataset_path=dataset_path,
        max_episodes=config.get("max_episodes"),
    )

    train_eps, val_eps = split_episodes(
        builder.episodes,
        val_ratio=float(config["val_ratio"]),
        seed=int(config["seed"]),
    )
    norm = compute_normalization(train_eps)

    train_dataset = CabinetDiffusionDataset(
        train_eps,
        norm,
        pred_horizon=int(config["pred_horizon"]),
        obs_horizon=int(config["obs_horizon"]),
        action_horizon=int(config["action_horizon"]),
    )
    val_dataset = (
        CabinetDiffusionDataset(
            val_eps,
            norm,
            pred_horizon=int(config["pred_horizon"]),
            obs_horizon=int(config["obs_horizon"]),
            action_horizon=int(config["action_horizon"]),
        )
        if val_eps
        else None
    )

    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "shuffle": True,
        "drop_last": False,
        "num_workers": int(config.get("num_workers", 0)),
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=int(config.get("num_workers", 0)),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_diffusion_mlp(
        obs_horizon=int(config["obs_horizon"]),
        state_dim=builder.state_dim,
        action_dim=builder.action_dim,
        pred_horizon=int(config["pred_horizon"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(config["num_diffusion_steps"]),
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    total_steps = len(train_loader) * int(config["epochs"])
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(config["lr_warmup_steps"]),
        num_training_steps=max(total_steps, 1),
    )

    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_policy.pt")
    final_path = os.path.join(checkpoint_dir, "final_policy.pt")

    print(f"Loaded episodes: {len(builder.episodes)}")
    print(f"Total steps:      {builder.total_steps}")
    print(f"State dim:        {builder.state_dim}")
    print(f"Action dim:       {builder.action_dim}")
    print(f"State format:     {builder.state_format}")
    if builder.state_format == "packed_observation_state":
        print(f"State keys:       [{builder.packed_state_column}]")
    else:
        print(f"State keys:       {STATE_KEYS}")
    print(f"Action columns:   {builder.action_columns}")
    print(f"Train episodes:   {len(train_eps)}")
    print(f"Val episodes:     {len(val_eps)}")
    print(f"Train samples:    {len(train_dataset)}")
    print(f"Val samples:      {len(val_dataset) if val_dataset is not None else 0}")
    print(f"Device:           {device}")

    print_section("Hyperparameters")
    print(f"Epochs:              {config['epochs']}")
    print(f"Batch size:          {config['batch_size']}")
    print(f"LR:                  {config['learning_rate']}")
    print(f"Obs horizon:         {config['obs_horizon']}")
    print(f"Pred horizon:        {config['pred_horizon']}")
    print(f"Action horizon:      {config['action_horizon']}")
    print(f"Diffusion steps:     {config['num_diffusion_steps']}")
    print(f"Hidden dim:          {config['hidden_dim']}")
    print(f"Weight decay:        {config['weight_decay']}")
    print(f"LR warmup steps:     {config['lr_warmup_steps']}")

    best_metric = float("inf")
    train_loss = float("inf")

    for epoch in range(int(config["epochs"])):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            batch_size = obs.shape[0]

            noise = torch.randn_like(action)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            noisy_actions = noise_scheduler.add_noise(action, noise, timesteps)
            noise_pred = model(noisy_actions, timesteps, obs)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            running_loss += float(loss.item()) * batch_size
            seen += batch_size

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
                    "loss": metric,
                    "metric_name": "val_loss" if val_loader is not None else "train_loss",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "state_keys": STATE_KEYS,
                    "state_format": builder.state_format,
                    "packed_state_column": builder.packed_state_column,
                    "action_columns": builder.action_columns,
                    "normalization": {
                        "type": "standard",
                        "state_mean": norm.state_mean,
                        "state_std": norm.state_std,
                        "action_mean": norm.action_mean,
                        "action_std": norm.action_std,
                    },
                    "state_mean": norm.state_mean,
                    "state_std": norm.state_std,
                    "action_mean": norm.action_mean,
                    "action_std": norm.action_std,
                    "model_config": {
                        "architecture": "diffusion_mlp",
                        "state_dim": builder.state_dim,
                        "action_dim": builder.action_dim,
                        "obs_horizon": int(config["obs_horizon"]),
                        "pred_horizon": int(config["pred_horizon"]),
                        "action_horizon": int(config["action_horizon"]),
                        "hidden_dim": int(config["hidden_dim"]),
                        "num_diffusion_steps": int(config["num_diffusion_steps"]),
                    },
                },
                best_path,
            )

    torch.save(
        {
            "epoch": int(config["epochs"]),
            "loss": train_loss,
            "metric_name": "final_train_loss",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_keys": STATE_KEYS,
            "state_format": builder.state_format,
            "packed_state_column": builder.packed_state_column,
            "action_columns": builder.action_columns,
            "normalization": {
                "type": "standard",
                "state_mean": norm.state_mean,
                "state_std": norm.state_std,
                "action_mean": norm.action_mean,
                "action_std": norm.action_std,
            },
            "state_mean": norm.state_mean,
            "state_std": norm.state_std,
            "action_mean": norm.action_mean,
            "action_std": norm.action_std,
            "model_config": {
                "architecture": "diffusion_mlp",
                "state_dim": builder.state_dim,
                "action_dim": builder.action_dim,
                "obs_horizon": int(config["obs_horizon"]),
                "pred_horizon": int(config["pred_horizon"]),
                "action_horizon": int(config["action_horizon"]),
                "hidden_dim": int(config["hidden_dim"]),
                "num_diffusion_steps": int(config["num_diffusion_steps"]),
            },
        },
        final_path,
    )

    print("\nTraining complete.")
    print(f"Best checkpoint:  {best_path}")
    print(f"Final checkpoint: {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal diffusion policy for OpenCabinet")
    defaults = default_config()
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=defaults["learning_rate"], help="Learning rate")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=defaults["checkpoint_dir"],
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )
    parser.add_argument("--max_episodes", type=int, default=defaults["max_episodes"], help="Cap on episodes to load")
    parser.add_argument("--val_ratio", type=float, default=defaults["val_ratio"], help="Episode validation split")
    parser.add_argument("--obs_horizon", type=int, default=defaults["obs_horizon"], help="Number of state steps to condition on")
    parser.add_argument("--pred_horizon", type=int, default=defaults["pred_horizon"], help="Number of future actions predicted")
    parser.add_argument("--action_horizon", type=int, default=defaults["action_horizon"], help="Number of actions to execute before replanning")
    parser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=defaults["num_diffusion_steps"],
        help="Number of diffusion denoising steps",
    )
    parser.add_argument("--hidden_dim", type=int, default=defaults["hidden_dim"], help="Hidden size for the noise model")
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"], help="AdamW weight decay")
    parser.add_argument("--lr_warmup_steps", type=int, default=defaults["lr_warmup_steps"], help="Cosine LR warmup steps")
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"], help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Training")
    print("=" * 60)

    if args.config:
        config = default_config()
        config.update(load_config(args.config))
    else:
        config = default_config()
        config.update(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "checkpoint_dir": args.checkpoint_dir,
                "max_episodes": args.max_episodes,
                "val_ratio": args.val_ratio,
                "obs_horizon": args.obs_horizon,
                "pred_horizon": args.pred_horizon,
                "action_horizon": args.action_horizon,
                "num_diffusion_steps": args.num_diffusion_steps,
                "hidden_dim": args.hidden_dim,
                "weight_decay": args.weight_decay,
                "lr_warmup_steps": args.lr_warmup_steps,
                "num_workers": args.num_workers,
                "seed": args.seed,
            }
        )

    validate_config(config)

    np.random.seed(int(config.get("seed", 0)))

    try:
        import torch

        torch.manual_seed(int(config.get("seed", 0)))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.get("seed", 0)))
    except ImportError:
        pass

    train_diffusion_policy(config)


if __name__ == "__main__":
    main()
