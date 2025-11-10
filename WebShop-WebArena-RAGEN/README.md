
  RAGEN-Implementation-for-Web-Navigation-Tasks 1.pdf    Presentation file
 
  presentation link
  https://teams.microsoft.com/l/meetingrecap?driveId=b%21lpKbtN_tQkKxtiwlnJghCZr6jH1n5YJHiA27kQNInRFHnPywmj3uSb5GyiDc1NoG&driveItemId=01UPQNR4RFM3CWZTPAARDLKFVU2W7CRVHN&sitePath=https%3A%2F%2Fnortheastern-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fkirmani_s_northeastern_edu%2FESVmxWzN4ARGtRa01b4o1O0BXc1LEV87F1g-jQwfuOwmlA&fileUrl=https%3A%2F%2Fnortheastern-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fkirmani_s_northeastern_edu%2FESVmxWzN4ARGtRa01b4o1O0BXc1LEV87F1g-jQwfuOwmlA&threadId=19%3Aa5bdb901a3da4d12bc1b6bc7e2c6e146%40thread.v2&organizerId=2083de5e-d64f-4d05-ad43-d72315fd9f60&tenantId=a8eec281-aaa3-4dae-ac9b-9a398b9215e7&callId=8a61cfcf-a3d2-40e9-ab95-140c0e7e0cd5&threadType=GroupChat&meetingType=Unknown&subType=RecapSharingLink_RecapChiclet


# WebShop-WebArena-RAGEN

A toolkit for agent-based web navigation and simulated shopping, featuring reinforcement learning, modular environments, experiment configuration, and result analysis.

## About

WebShop-WebArena-RAGEN provides code and setup for building, training, and evaluating web navigation agents. It supports structured experiments, agent models, environment wrappers, utility scripts, and evaluation tools for reproducible research and development.

---

## Project Structure

- `agents/` — agent logic and architectures
- `algorithms/` — planning modules (astar, MCTS, etc.)
- `environments/` — WebShop & WebArena interfaces
- `evaluation/` — metrics, analysis, logs
- `experiments/` — configs, outputs, results
- `models/` — model declarations and helpers
- `training/` — training scripts and config files
- `utils/` — general-purpose helper utilities

---

## Installation

git clone https://github.com/KirmaniMeharAli/neu-self-improve-ai.git
cd neu-self-improve-ai/WebShop-WebArena-RAGEN

text

(Optional) Set up a Python environment:
python -m venv venv
source venv/bin/activate # or .\venv\Scripts\activate on Windows



Install dependencies:
pip install -r requirements.txt

or manually: numpy, torch, etc.
text

---

## Usage

To train the agent:
python train_complete.py



WebArena-specific training:
python train_webarena.py




Modify configurations by editing `config.yaml` or experiment-specific JSON files in `experiments/configs/`.

---

## Configuration

- `config.yaml` — contains training hyperparameters and environment settings
- Experiment setup and output paths can be customized in `experiments/configs/`

---

## Results & Experiments

- Training outputs, logs, and evaluation files are saved in:  
  - `experiments/outputs/`
  - `experiments/results/`
  - Metrics, plots: `evaluation/`

---
