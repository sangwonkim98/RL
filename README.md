# PPO-based Sequential Document Navigation for Multi-hop QA Evidence Retrieval

Research MVP for training an RL agent to select document chunks as evidence for multi-hop QA questions.

## Project Structure
- `data_generator.py`: Generates synthetic `data.json`.
- `env.py`: Gymnasium custom environment `DocNavEnv`.
- `train.py`: PPO training using Stable-Baselines3.
- `baselines.py`: Baseline policies (Random, Greedy Similarity).
- `eval.py`: Evaluation script comparing PPO and baselines.
- `plot_results.py`: Generates plots from `metrics.json`.
- `data.json`: Synthetic dataset (automatically generated).

## Setup
Install dependencies:
```bash
pip install gymnasium stable-baselines3 scikit-learn matplotlib numpy
```

## Running the MVP

### 1. Generate Synthetic Data
```bash
python -m ppo_navigation.data_generator
```

### 2. Train the PPO Agent
```bash
python -m ppo_navigation.train
```

### 3. Evaluate and Compare with Baselines
```bash
python -m ppo_navigation.eval
```

### 5. (Optional) HotpotQA Preprocessing
If you have a HotpotQA-style dataset (JSON format), you can convert it to the environment's format:

```bash
# To test with a dummy sample:
python3 -m ppo_navigation.preprocess_hotpot --dummy --input test_hotpot.json

# To process real HotpotQA data:
python3 -m ppo_navigation.preprocess_hotpot --input your_hotpot_file.json --output ppo_navigation/hotpot_data.json
```

Then, update the `data_path` in `train.py` or `eval.py` to point to the new file.


## RL Environment Details
- **Observation Space**: Flattened vector containing:
    - Question TF-IDF embedding (64-dim)
    - Document TF-IDF embeddings (5 docs * 64-dim)
    - Visited mask (5-dim binary)
    - Current step (1-dim)
- **Action Space**: `Discrete(6)`
    - 0-4: Select document index
    - 5: STOP action
- **Rewards**:
    - `+1`: First-time selection of a support document.
    - `+3`: When all support documents are found.
    - `+5`: If STOP after all support documents found.
    - `-3`: If STOP too early (missed support docs).
    - `-0.2`: Irrelevant document selected.
    - `-0.5`: Repeated selection of same doc.
    - `-0.05`: Step penalty.
