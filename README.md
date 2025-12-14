
# Routing and Spectrum Allocation with DQN  
CS 258 — Final Project (Fall 2025)

---

## How to Execute

### Training  
Run the DQN agent for each capacity setting:

```bash
cd src
python dqn_runner.py --capacity 20
python dqn_runner.py --capacity 10
```

Each run trains an agent using `data/train/`, saves a model under `models/`, and generates training plots under `plots/`.

### Evaluation  
(Will be completed once evaluation.py is finalized.)

After training, evaluation will be run using:

```bash
python evaluation.py
```

This evaluates the trained models on `data/eval` using deterministic predictions.

---

### 2. Hyperparameter Tuning (Systematic Search)

The baseline tuning command format is:

```bash
python dqn_runner.py --capacity 20 --lr <value> --exploration_fraction <value> --timesteps 40000 --tag <label>
```

Substitute the values with:

- `<value>` for learning rate: **0.0005**, **0.001**, **0.002**  
- `<value>` for exploration fraction: **0.1**, **0.2**  
- `<label>` any short identifier, e.g. `lr5e-4_exp0.1`

This creates **6 total tuning runs**.

Each tuning run:
- trains for 40,000 timesteps,
- saves tagged models to `../models/`,
- saves tagged plots to `../plots/`.

Based on comparing 10-episode moving-average reward and blocking rate, the best configuration was:

- **learning_rate = 1e-3**  
- **exploration_fraction = 0.1**

These values are now the defaults used for all final training runs.

---

## Environment

The environment simulates the Routing and Spectrum Allocation (RSA) problem.  
Each **episode** corresponds to **one request file (~100 requests)**, as required:

> One simulation (episode) corresponds to one request file with 100 requests.

Each step (time slot) processes exactly one request.

---

## State Representation & Transitions

### Observation Vector (35 features)

- **12 link utilizations:** (# wavelengths used) / capacity  
- **12 free wavelength ratios:** (# free wavelengths) / capacity  
- **3 request features:** normalized source, destination, holding time  
- **8 path availability flags:** 1 if a path has an available wavelength, else 0  

### State Transitions (per request)

1. Release expired lightpaths.  
2. Attempt routing + first-fit wavelength allocation while enforcing:  
   - wavelength continuity  
   - capacity limits  
   - uniqueness of wavelength on each link  
3. Update link occupancy and active lightpaths.

Link state is maintained in `LinkState` objects that store capacity, utilization, wavelength occupancy, and expiration times.

---

## Action Representation

The agent selects one of 8 discrete routing actions:

| Actions | Meaning |
|--------|---------|
| 0–1 | Paths P1–P2 for requests 0 → 3 |
| 2–3 | Paths P3–P4 for 0 → 4 |
| 4–5 | Paths P5–P6 for 7 → 3 |
| 6–7 | Paths P7–P8 for 7 → 4 |

Invalid actions for a given (src, dst) automatically result in failed allocation.

---

## Reward Function

- **0** → successful lightpath allocation  
- **–1** → request blocked  

Additionally, the environment exposes:

```python
info["blocked"] = 0 or 1
```

This allows computing the **episodic blocking rate**, the main optimization objective:

\[
B = rac{	ext{blocked requests}}{	ext{total requests}}.
\]

---

## Additional Constraints

- Wavelength continuity  
- Capacity constraint  
- Wavelength conflict avoidance  
- First-fit spectrum allocation  
- Episodes fixed to one request file (~100 steps)  

---

## Training Setup

### Algorithm
Deep Q‑Network (DQN) from Stable‑Baselines3.

### Hyperparameters (initial configuration)

```python
learning_rate = 1e-3
batch_size = 64
buffer_size = 50000
learning_starts = 1000
gamma = 0.99
target_update_interval = 1000
exploration_fraction = 0.2
exploration_final_eps = 0.05
```

Training runs use **200,000 timesteps per capacity**, giving ~2000 episodes (each ≈100 requests).

### Hyperparameter Tuning  

We performed grid search over:

- Learning rate ∈ {5e‑4, 1e‑3, 2e‑3}
- Exploration fraction ∈ {0.1, 0.2}

Each configuration was trained for **40,000 timesteps** and evaluated using:

- 10‑episode moving‑average reward  
- 10‑episode moving‑average blocking rate  

\[
B = frac{	ext{blocked requests}}{	ext{total requests}}
\]
---

## Results  
*(Plots and analysis will be added once evaluation.py is completed and all models are trained.)*

Expected structure:

### Capacity = 20  
- Learning curve (avg reward vs episode)  
- Training blocking rate (avg B vs episode)  
- Evaluation blocking rate (avg B vs episode)  

### Capacity = 10  
- Same 3 plots as above  

Each will include a brief explanation of trends and observations.

---

## Files Included

- `rsaenv.py` — custom RSA environment  
- `nwutil.py` — graph and request structures  
- `dqn_runner.py` — training pipeline  
- `evaluation.py` — evaluation pipeline (coming soon)  
- `models/` — trained agents (saved)  
- `plots/` — generated figures  
- `data/train`, `data/eval` — datasets

---

## Notes  
left space for:  
- hyperparameter tuning summary  
- evaluation results  
- final observations and discussion  

