# PPO Implementation in PyTorch

A readable and well-documented implementation of the Proximal Policy Optimization (PPO-Clip) algorithm in PyTorch.

## Key Features

* **Generalized Advantage Estimation (GAE):** to reduce variance in policy updates.
* **Vectorized Environments:** Uses `gym.vector.SyncVectorEnv` for efficient, parallel data collection.
* **KL-Divergence Early Stopping:** to help prevent destructive policy updates but get the most out of each rollout.
* **Modern PyTorch Features:** Leverages `torch.compile` for just-in-time compilation of the actor and critic networks.
* **Training Stability:** Includes standard techniques like advantage normalization, gradient clipping and entropy regularization.
* **Experiment Tracking:** Logs key metrics to Weights and Biases

## Setup and Installation

1.  **Install Python & PyTorch:**
    - python >= 3.7 (3.11 tested)
    - pytorch >= 2.0 (2.6 tested)

1.  **System Dependencies (for video rendering):**
    ```bash
    sudo apt-get install -y xvfb ffmpeg
    ```

2.  **Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

To start training on the default `HalfCheetah-v5` environment, run:
```bash
python ppo.py
```