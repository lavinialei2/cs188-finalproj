# Cabinet Door Opening Robot - CS 188 Starter Project

## Overview

This repo documents a completed pipeline for training a robot policy to open
kitchen cabinet doors in **RoboCasa365**, a large-scale simulation benchmark
for everyday manipulation tasks. The work covers environment inspection,
demonstration playback, dataset augmentation, and training/evaluating a
diffusion-based policy that controls the PandaOmron robot autonomously.

### What this includes

1. How the manipulation environment is structured (MuJoCo + robosuite + RoboCasa)
2. The `OpenCabinet` task setup: sensors, actions, and success criteria
3. Demonstration datasets (human + MimicGen) and how they are used
4. Training a state-based diffusion policy with a 1D U‑Net backbone
5. Evaluating trained policies in simulation

### The robot

We use the **PandaOmron** mobile manipulator -- a Franka Panda 7-DOF arm
mounted on an Omron wheeled base with a torso lift joint. This is the default
and best-supported robot in RoboCasa.

---

## Installation

Run the install script (works on **macOS** and **WSL/Linux**):

```bash
./install.sh
```

This will:
- Create a Python virtual environment (`.venv`)
- Clone and install robosuite and robocasa
- Install all Python dependencies (PyTorch, numpy, matplotlib, etc.)
- Download RoboCasa kitchen assets (~10 GB)

After installation, activate the environment:

```bash
source .venv/bin/activate
```

Then verify everything works:

```bash
cd cabinet_door_project
python 00_verify_installation.py
```

> **macOS note:** Scripts that open a rendering window (03, 05) require
> `mjpython` instead of `python`. The install script will remind you of this.

---

## Project Structure

```
cabinet_door_project/
  00_verify_installation.py      # Check that everything is installed correctly
  01_explore_environment.py      # Create the OpenCabinet env, inspect observations/actions
  02_random_rollouts.py          # Run random actions, save video, understand the task
  03_teleop_collect_demos.py     # Teleoperate the robot to collect your own demonstrations
  04_download_dataset.py         # Download the pre-collected OpenCabinet dataset
  05_playback_demonstrations.py  # Play back demonstrations to see expert behavior
  06_train_policy.py             # Train the starter-code diffusion policy
  07_evaluate_policy.py          # Evaluate your trained policy in simulation
  08_visualize_policy_rollout.py # Visualize a rollout of your policy in RoboCasa
  configs/
    diffusion_policy.yaml        # Training hyperparameters
  notebook.ipynb                 # Interactive Jupyter notebook companion
install.sh                       # Installation script (macOS + WSL/Linux)
README.md                        # This file
```

---

## Step-by-Step Guide

### Step 0: Verify Installation

```bash
python 00_verify_installation.py
```

This checks that MuJoCo, robosuite, RoboCasa, and all dependencies are
correctly installed and that the `OpenCabinet` environment can be created.

### Step 1: Explore the Environment

```bash
python 01_explore_environment.py
```

This script creates the `OpenCabinet` environment and prints detailed
information about:
- **Observation space**: what the robot sees (camera images, joint positions,
  gripper state, base pose)
- **Action space**: what the robot can do (arm movement, gripper open/close,
  base motion, control mode)
- **Task description**: the natural language instruction for the episode
- **Success criteria**: how the environment determines task completion

### Step 2: Random Rollouts

```bash
python 02_random_rollouts.py
```

Runs the robot with random actions to see what happens (spoiler: nothing
useful, but it helps you understand the action space). Saves a video to
`/tmp/cabinet_random_rollouts.mp4`.

### Step 3: Teleoperate and Collect Demonstrations

```bash
# Mac users: use mjpython instead of python
python 03_teleop_collect_demos.py
```

Control the robot yourself using the keyboard to open cabinet doors. This
gives you intuition for the task difficulty and generates demonstration data.

**Keyboard controls:**
| Key | Action |
|-----|--------|
| Ctrl+q | Reset simulation |
| spacebar | Toggle gripper (open/close) |
| up-right-down-left | Move horizontally in x-y plane |
| .-; | Move vertically |
| o-p | Rotate (yaw) |
| y-h | Rotate (pitch) |
| e-r | Rotate (roll) |
| b | Toggle arm/base mode (if applicable) |
| s | Switch active arm (if multi-armed robot) |
| = | Switch active robot (if multi-robot environment) |              

### Step 4: Download Pre-collected Dataset

```bash
python 04_download_dataset.py
```

Downloads the official OpenCabinet demonstration dataset from the RoboCasa
servers. This includes both human demonstrations and MimicGen-expanded data
across diverse kitchen scenes.

### Step 5: Play Back Demonstrations

```bash
python 05_playback_demonstrations.py
```

Visualize the downloaded demonstrations to see how an expert opens cabinet
doors. This is the data your policy will learn from.

### Step 5b: Augment the Dataset with Handle Features

```bash
python 05b_augment_handle_data.py
```

The raw LeRobot parquet files do **not** include the cabinet handle position or
door hinge information, which are crucial for state-only policies. This script:
- Replays each demo’s saved MuJoCo state sequence from `extras/episode_*`
- Finds the cabinet’s handle body and door joints
- Computes per‑timestep features:
  - `observation.handle_pos` (3D world position)
  - `observation.handle_to_eef_pos` (handle position relative to end effector)
  - `observation.door_openness` (normalized door joint value)
  - `observation.handle_xaxis` (handle x‑axis direction)
  - `observation.hinge_direction` (+1 right‑opening, ‑1 left‑opening)
- Writes **augmented parquet files** to `dataset_path/augmented/`

Training auto‑detects these augmented files and includes them in the state
vector, so you don’t have to modify training scripts beyond running this step.

### Step 6: Train a Policy

```bash
python 06_train_policy.py
```

This repo now trains a **1D Conv U-Net diffusion policy** directly from
low-dimensional state observations (no video). Key updates from the starter code:
- **State augmentation**: uses the handle features produced by `05b_augment_handle_data.py`
- **Normalization**: saves mean/std to the checkpoint and reuses them at eval
- **State history**: optional temporal stacking via `--state_history`
- **MPS support**: Apple Silicon Macs use `mps` automatically when available

Example command that trained the current model:

```bash
python cabinet_door_project/06_train_policy.py \
  --epochs 100 --batch_size 64 --lr 1e-4 \
  --diffusion_steps 50 \
  --unet_channels 128 --unet_channel_mults 1,2,4,8 \
  --state_history 6 \
  --max_episodes -1
```

Argument overview:
- `--epochs` number of training epochs
- `--batch_size` batch size per step
- `--lr` AdamW learning rate
- `--diffusion_steps` number of diffusion timesteps
- `--unet_channels` base channel width for the 1D U-Net
- `--unet_channel_mults` channel multipliers per U-Net level
- `--state_history` number of past states to stack per input
- `--max_episodes` limit dataset episodes (`-1` = all episodes)

On Apple Silicon Macs (M‑series), the script will use the MPS backend when available.

### Step 7: Evaluate Your Policy

```bash
python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt
python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt --split target --num_rollouts 20
```

Evaluation updates:
- **Success criterion**: counts success if **any one cabinet door** is open
- **State alignment**: uses the same state keys and normalization from training
- **Action ordering**: reorders LeRobot actions to the env’s expected layout
- **Video output**: defaults to `cabinet_door_project/eval_videos/eval_<timestamp>.mp4`
- **Forced layout/style**: use `--layout_id` and `--style_id`

Common eval flags:
- `--max_steps` max steps per episode
- `--num_rollouts` number of episodes
- `--split` pretrain or target
- `--layout_id` and `--style_id` to lock a specific kitchen scene
- `--stochastic` to enable stochastic diffusion sampling

Example layout/style test:

```bash
python cabinet_door_project/07_evaluate_policy.py \
  --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt \
  --layout_id 27 --style_id 21 \
  --num_rollouts 1 --max_steps 1200
```

The model trained with the command above achieved a **6/20** success rate in
your evaluation run, using the “one door open” success criterion.

---

## Key Concepts

### The OpenCabinet Task

- **Goal**: Open a kitchen cabinet door
- **Fixture**: `HingeCabinet` (a cabinet with hinged doors)
- **Initial state**: Cabinet door is closed; robot is positioned nearby
- **Success**: `fixture.is_open(env)` returns `True`
- **Horizon**: 500 timesteps at 20 Hz control frequency (25 seconds)
- **Scene variety**: 2,500+ kitchen layouts/styles for generalization

### Observation Space (PandaOmron)

| Key | Shape | Description |
|-----|-------|-------------|
| `robot0_agentview_left_image` | (256, 256, 3) | Left shoulder camera |
| `robot0_agentview_right_image` | (256, 256, 3) | Right shoulder camera |
| `robot0_eye_in_hand_image` | (256, 256, 3) | Wrist-mounted camera |
| `robot0_gripper_qpos` | (2,) | Gripper finger positions |
| `robot0_base_pos` | (3,) | Base position (x, y, z) |
| `robot0_base_quat` | (4,) | Base orientation quaternion |
| `robot0_base_to_eef_pos` | (3,) | End-effector pos relative to base |
| `robot0_base_to_eef_quat` | (4,) | End-effector orientation relative to base |

### Action Space (PandaOmron)

| Key | Dim | Description |
|-----|-----|-------------|
| `end_effector_position` | 3 | Delta (dx, dy, dz) for the end-effector |
| `end_effector_rotation` | 3 | Delta rotation (axis-angle) |
| `gripper_close` | 1 | 0 = open, 1 = close |
| `base_motion` | 4 | (forward, side, yaw, torso) |
| `control_mode` | 1 | 0 = arm control, 1 = base control |

### Dataset Format (LeRobot)

Datasets are stored in LeRobot format:
```
dataset/
  meta/           # Episode metadata (task descriptions, camera info)
  videos/         # MP4 videos from each camera
  data/           # Parquet files with actions, states, rewards
  extras/         # Per-episode metadata
```

---

## Architecture Diagram

```
                    RoboCasa Stack
                    ==============

  +-------------------+     +-------------------+
  |   Kitchen Scene   |     |   OpenCabinet     |
  |  (2500+ layouts)  |     |   (Task Logic)    |
  +--------+----------+     +--------+----------+
           |                         |
           v                         v
  +------------------------------------------------+
  |              Kitchen Base Class                 |
  |  - Fixture management (cabinets, fridges, etc)  |
  |  - Object placement (bowls, cups, etc)          |
  |  - Robot positioning                            |
  +------------------------+-----------------------+
                           |
                           v
  +------------------------------------------------+
  |              robosuite (Backend)                |
  |  - MuJoCo physics simulation                   |
  |  - Robot models (PandaOmron, GR1, Spot, ...)   |
  |  - Controller framework                        |
  +------------------------+-----------------------+
                           |
                           v
  +------------------------------------------------+
  |              MuJoCo 3.3.1 (Physics)            |
  |  - Contact dynamics, rendering, sensors        |
  +------------------------------------------------+
```

## References

- [RoboCasa Paper & Website](https://robocasa.ai/)
- [RoboCasa GitHub](https://github.com/robocasa/robocasa)
- [robosuite Documentation](https://robosuite.ai/)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
