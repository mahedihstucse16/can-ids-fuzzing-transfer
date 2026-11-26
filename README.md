# Fuzzing attacks in training cause cross-vehicle false alarms in CAN intrusion detectors

Analysis code for the paper *Fuzzing Attacks in Training Data Induce Cross-Vehicle False Alarms in
Controller Area Network Intrusion Detectors: A Controlled Case Study*
(Hassan, Azad, Alam, Elahe, Sadi, Sayeed; submitted to *Results in Engineering*).

Every number, table and figure in the paper is recomputed by this code from three
public HCRL datasets. Nothing is transcribed by hand.

## Data

Download the following files from the Hacking and Countermeasure Research Lab (HCRL)
and put them in one folder:

| Dataset | Files | Source |
|---|---|---|
| Car-Hacking (Hyundai YF Sonata) | `DoS_dataset.csv`, `Fuzzy_dataset.csv`, `gear_dataset.csv`, `RPM_dataset.csv` | https://ocslab.hksecurity.net/Datasets/car-hacking-dataset |
| Survival Analysis (Sonata, Soul, Spark) | `Sonata/`, `Soul/`, `Spark/` folders (free-driving, flooding, fuzzy, malfunction) | https://sites.google.com/hksecurity.net/hcrl/Datasets/survival-ids |
| OTIDS / CAN-intrusion (KIA Soul) | `DoS_attack_dataset.txt`, `Attack_free_dataset.txt`, `Fuzzy_attack_dataset.txt`, `Impersonation_attack_dataset.txt` (label audit only) | https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset |

## Environment

```
python -m pip install -r requirements.txt
```
`song_dcnn.py` additionally needs PyTorch (not pinned; any recent version).

The paper's models were all fitted under Python 3.13 with the exact versions pinned in
`requirements.txt` (scikit-learn 1.9.1). Random forests with class weights are not
bit-for-bit identical across scikit-learn versions, so use the pinned versions to
reproduce the tables exactly.

## Pipeline

1. **Feature cache.** `CAN_SHIELD_Reproduction.ipynb` parses the raw captures, removes
   remote frames, derives labels and computes the 18 features (10 raw fields and 8
   protocol-level features) over each complete capture in timestamp order. It writes
   `X_*.npy`, `y_*.npy` and `ts_*.npy` to `results_v2/cache/`, together with the data
   audits that involve no model (duplication, identifier overlap, correlation matrix).
2. **Experiments.**
   ```
   python rev3_pipeline.py results_v2/cache rev3_out e2 e3 e4 e5 mech mech2 e1cv explain event
   python rev3_pipeline.py results_v2/cache rev3_out hyper e1split dedup big
   python rev3_pipeline.py results_v2/cache rev3_out engabl baselines calib2 ifdiag boot fznorm song songcnn
   python rev3_vocab.py results_v2/cache rev3_out
   ```
   One protocol for every run: training-partition scaling and class weighting for every
   representation, no synthetic oversampling, ten draws (seeds 0-9). Every unit is
   check-pointed as JSON, so an interrupted run resumes.
3. **Results file.** `rev3_vocab.py` first writes the OTIDS identifier census used by the
   false-positive analysis. Then `python rev3_aggregate.py rev3_out rev3_results.json` collects every
   checkpoint into one file, the single source of every number quoted in the paper.
4. **Tables, figures and supplement.**
   ```
   python rev3_tables.py rev3_results.json latex/ supp_csv/
   python -c "import json,rev3_numbers as n; json.dump(n.build(json.load(open('rev3_results.json'))), open('numbers.json','w'))"
   python rev3_supplementary.py results_v2 supp_csv/ supplementary.tex "<title>" numbers.json manuscript.aux latex/
   ```
5. **Label audit.** `python audit_impersonation.py Impersonation_attack_dataset.txt`
   reproduces the impersonation row of the OTIDS label audit.

## Second vehicle set (Survival Analysis)
```
python surv_parse.py <survival>/dataset surv
python surv_cache.py surv
python surv_pipeline.py surv surv_out e2 e3
python surv_aggregate.py surv_out surv_out/surv_results.json
```
`rev3_aggregate.py` picks up `../surv_out/surv_results.json` when it exists.

## Optional: Song et al. (2020) network in PyTorch
`song_export.py` writes the 29-identifier windows and the exact training and held-out windows of
every draw used by the `song` and `songcnn` stages; `song_dcnn.py` trains a PyTorch
re-implementation of Song et al.'s reduced Inception-ResNet on the same windows (GPU recommended;
`Song_DCNN_Colab.ipynb` runs it on Google Colab):
```
python song_export.py results_v2/cache song_windows.npz
python song_dcnn.py song_windows.npz rev3_out/checkpoints/
```
Its results are written in the checkpoint format read by `rev3_aggregate.py`.

## Experiments

| Stage | Experiment |
|---|---|
| `e1cv`, `e1split` | E1 within-vehicle evaluation: five-fold CV and three splits |
| `e2` | E2 composition ablation at a constant 120,000 training frames |
| `mech` | E2 mechanism variants: fuzzing frames given legitimate identifiers or real payloads; identifier removed |
| `mech2` | Controls: identifier + DLC only and payload only (raw 9), with and without fuzzing |
| `e3` | E3 dose-response, 0 to 40,000 frames from the fuzzing capture |
| `e4` | E4 reverse transfer, KIA Soul to Sonata |
| `e5` | E5 classifier generality, four learners |
| `big`, `hyper` | Robustness: about 1,000,000 training frames; forest size and depth |
| `event` | Event-level false alarms and detection delay on the complete OTIDS captures |
| `engabl`, `baselines`, `calib2`, `ifdiag`, `boot` | Feature ablation; tuned, neural and benign-only baselines; interleaved and temporally separated target calibration; isolation-forest score diagnostic; block bootstrap |
| `fznorm` | Control: the fuzzing capture's legitimate frames only (injected frames removed), 30,000 frames |
| `song`, `songcnn` | Song et al. (2020) 29-frame identifier-grid input, with a multilayer perceptron and random forest (`song`) and a compact NumPy convolutional network (`songcnn`), with and without fuzzing |
| `dedup` | Shared 891,068-frame tail of the Car-Hacking captures counted once |
| `explain` | Monte Carlo Shapley, impurity and permutation importance |

## Licence and citation

MIT licence (see `LICENSE`). Please cite the paper; citation metadata are in
`CITATION.cff`. An archived copy of each release is deposited on Zenodo.
