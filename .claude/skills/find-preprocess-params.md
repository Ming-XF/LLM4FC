---
name: find-preprocess-params
description: >
  Find optimal max_windows_per_subject / max_subjects for a dataset preprocessing
  script. Balances total samples ≤ target_total, class difference ≤ max_diff,
  min windows per subject ≥ min_windows, and maximizes subject coverage.
  Works for classification, regression (age), and multi-output regression (future FC) tasks.
---

# Find Preprocessing Parameters

This skill automates the process of finding optimal `max_windows_per_subject` and
`max_subjects` parameters for EEG preprocessing scripts in this codebase.

## Prerequisites

The target preprocessing file must have these features (all current files under
`data/*/` already follow this pattern):

- A top-level `_resolve_param(val, pos)` function
- `max_windows_per_subject` and `max_subjects` parameters in the main `*_preprocess()` function
- Access to source EDF/annotation data

## Step 1 — Identify the task type

Open the target preprocessing file and determine the task type from the labels:

| Task type | Label type | How classes are determined | Example files |
|---|---|---|---|
| **Classification** | Discrete (0/1/...) | `pos = (subj_label == 1)` — already correct | `disease_*.py`, `gender_*.py` |
| **Regression** (age) | Continuous float | **Needs median binarization fix** — see Step 1a | `age_*.py` |
| **Multi-output regression** (future FC) | Matrix | No class concept — only flat int caps | `futurefc_*.py` |

### Step 1a — Check for the regression binarization bug

For **regression** files (age prediction), check whether the window/subject cap
sections use `pos=(subj_label == 1)`. This is **wrong** for continuous ages
(23–96 is never `== 1`). The fix is to add a median binarization step:

```python
# Add BEFORE the max_windows_per_subject block:
_median_age = np.median(labels)
_binary_labels = (labels >= _median_age).astype(int)  # 1=old(pos), 0=young(neg)

# Then in the cap loop, replace:
#   pos=(subj_label == 1)
# with:
#   pos=(_binary_labels[idx[0]] == 1)

# In the max_subjects block, replace:
#   subj_labels = np.array([np.bincount(labels[subject_ids == s].astype(int)).argmax() ...])
# with:
#   subj_binary = np.array([int(labels[subject_ids == s].mean() >= _median_age) ...])
#   pos_subjs = unique_subjs[subj_binary == 1]
#   neg_subjs = unique_subjs[subj_binary == 0]
```

Reference: `data/caueeg/age_caueeg.py` (already fixed).

## Step 2 — Scan raw data to get per-subject window counts and labels

Run a scan script to collect `(n_windows, label, subject_id)` for every subject
**without full EDF loading** (use `preload=False` for speed):

```bash
python3 << 'PYEOF'
import json, os, mne, warnings, numpy as np, re
warnings.filterwarnings('ignore')

# Adjust these paths for the target dataset
ANNOTATION = "../data/CAUEEG/caueeg-dataset/annotation.json"
SIGNAL_DIR = "../data/CAUEEG/caueeg-dataset/signal/edf"
HZ = 200
WINDOW_SEC = 60
N_CHANNELS = 19

with open(ANNOTATION) as f:
    ann = json.load(f)

samples = ann['data']
records = []
for s in samples:
    serial = s['serial']
    # Extract subject ID — check the _process_one_sample_* worker for the exact regex
    subj_id = int(re.findall(r'\d+', serial)[0])
    # Extract label — depends on task type:
    label = float(s['age'])   # for regression; use s['label'] for classification
    edf_path = os.path.join(SIGNAL_DIR, f"{serial}.edf")
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        n_windows = raw.n_times // (HZ * WINDOW_SEC)
        records.append({'subj_id': subj_id, 'label': label, 'windows': n_windows, 'serial': serial})
    except Exception as e:
        print(f"Skip {serial}: {e}")

# Expand to per-window arrays (each subject contributes `windows` rows)
all_subj_ids = []
all_labels = []
for r in records:
    all_subj_ids.extend([r['subj_id']] * r['windows'])
    all_labels.extend([r['label']] * r['windows'])
all_subj_ids = np.array(all_subj_ids)
all_labels = np.array(all_labels, dtype=np.float32)

print(f"Subjects: {len(records)}, Total windows: {len(all_labels)}")

# For regression: binarize at median
median = np.median(all_labels)
binary_labels = (all_labels >= median).astype(int)
print(f"Median label: {median:.1f}")
print(f"  Pos class (old/high):  {binary_labels.sum()} windows, "
      f"{(binary_labels == 1).sum()} windows over {len(np.unique(all_subj_ids[binary_labels == 1]))} subjects")
print(f"  Neg class (young/low): {(binary_labels == 0).sum()} windows, "
      f"{(binary_labels == 0).sum()} windows over {len(np.unique(all_subj_ids[binary_labels == 0]))} subjects")
print(f"  Class diff: {abs(binary_labels.sum() - (len(binary_labels) - binary_labels.sum()))}")

# Stats per subject
uniq = np.unique(all_subj_ids)
print(f"Windows/subject: min={min((all_subj_ids == s).sum() for s in uniq)}, "
      f"max={max((all_subj_ids == s).sum() for s in uniq)}, "
      f"mean={len(all_labels) / len(uniq):.1f}")

# Save for parameter search
np.savez('/tmp/scan_data.npz', subj_ids=all_subj_ids, labels=all_labels,
         binary_labels=binary_labels, median=median)
print("Saved to /tmp/scan_data.npz")
PYEOF
```

**For classification tasks:** remove the median binarization and use labels
directly (they are already 0/1).

**For multi-output regression:** there is no class concept — only flat integer
caps are applicable. Use a single int for both parameters.

## Step 3 — Simulate parameter combinations

Copy the `_resolve_param` logic and run a grid search over parameter combinations:

```bash
python3 << 'PYEOF'
import numpy as np

# Load scan data
d = np.load('/tmp/scan_data.npz')
all_subj_ids = d['subj_ids']
all_labels = d['labels']

# For regression: use stored binary_labels and median
binary_labels = d['binary_labels']
median = float(d['median'])
TASK_TYPE = 'regression'   # or 'classification'

# For classification: binary_labels = all_labels.astype(int) directly
if TASK_TYPE == 'classification':
    binary_labels = all_labels.astype(int)

# Constraints — adjust these for the target task
TARGET_TOTAL = 20000     # max total samples
MAX_CLASS_DIFF = 1000    # max imbalance between classes
MIN_WINDOWS = 5          # min windows per subject

# The _resolve_param function from the preprocessing file
def _resolve_param(val, pos):
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return val
    return val[0] if pos else val[1]

def simulate(ws_cap, sj_cap):
    """Simulate the exact preprocessing cap logic."""
    subj_ids = all_subj_ids.copy()
    labels = all_labels.copy()
    bin_labels = binary_labels.copy()

    # max_windows_per_subject
    if ws_cap is not None:
        unique_subjs = np.unique(subj_ids)
        keep_mask = np.ones(len(subj_ids), dtype=bool)
        for subj in unique_subjs:
            idx = np.where(subj_ids == subj)[0]
            subj_binary = bin_labels[idx[0]]
            cap = _resolve_param(ws_cap, pos=(subj_binary == 1))
            if cap is not None and len(idx) > cap:
                sample_idx = np.linspace(0, len(idx) - 1, cap, dtype=int)
                keep_mask[idx] = False
                keep_mask[idx[sample_idx]] = True
        subj_ids = subj_ids[keep_mask]
        labels = labels[keep_mask]
        bin_labels = bin_labels[keep_mask]

    # max_subjects
    if sj_cap is not None:
        unique_subjs = np.unique(subj_ids)
        if TASK_TYPE == 'regression':
            subj_binary = np.array([int(labels[subj_ids == s].mean() >= median)
                                    for s in unique_subjs])
        else:
            subj_binary = np.array([np.bincount(labels[subj_ids == s].astype(int)).argmax()
                                    for s in unique_subjs])
        pos_subjs = unique_subjs[subj_binary == 1]
        neg_subjs = unique_subjs[subj_binary == 0]

        n_pos_limit = _resolve_param(sj_cap, pos=True)
        n_neg_limit = _resolve_param(sj_cap, pos=False)

        if isinstance(sj_cap, (int, np.integer)):
            n_pos_limit = sj_cap // 2
            n_neg_limit = sj_cap // 2

        n_pos = len(pos_subjs) if n_pos_limit is None else min(n_pos_limit, len(pos_subjs))
        n_neg = len(neg_subjs) if n_neg_limit is None else min(n_neg_limit, len(neg_subjs))
        kept_pos = pos_subjs[:n_pos]
        kept_neg = neg_subjs[:n_neg]
        kept_subjs = np.concatenate([kept_pos, kept_neg])

        keep_mask = np.isin(subj_ids, kept_subjs)
        subj_ids = subj_ids[keep_mask]
        labels = labels[keep_mask]

    # Stats
    pos_mask = bin_labels == 1 if len(bin_labels) > 0 else np.zeros(0, dtype=bool)
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    total = n_pos + n_neg

    uniq = np.unique(subj_ids)
    n_subjects = len(uniq)
    min_w = min((subj_ids == s).sum() for s in uniq) if n_subjects > 0 else 0

    return {
        'total': int(total), 'pos': int(n_pos), 'neg': int(n_neg),
        'diff': int(abs(n_pos - n_neg)), 'subjects': n_subjects,
        'min_w': int(min_w),
    }

# Baseline
base = simulate(None, None)
print(f"Baseline (no caps): total={base['total']}, pos={base['pos']}, neg={base['neg']}, "
      f"diff={base['diff']}, subjs={base['subjects']}, min_w={base['min_w']}")

# ---- Search ----
print(f"\nSearching... (target total≤{TARGET_TOTAL}, diff≤{MAX_CLASS_DIFF}, min_w≥{MIN_WINDOWS})")

# Determine param ranges from baseline
max_w = max((all_subj_ids == s).sum() for s in np.unique(all_subj_ids))
pos_subj_count = len(np.unique(all_subj_ids[binary_labels == 1]))
neg_subj_count = len(np.unique(all_subj_ids[binary_labels == 0]))

best = None
for w_pos in list(range(MIN_WINDOWS, max_w + 1)):
    for s_pos in list(range(50, pos_subj_count + 10, 10)):
        r = simulate((w_pos, None), (s_pos, None))
        if r['total'] <= TARGET_TOTAL and r['diff'] <= MAX_CLASS_DIFF and r['min_w'] >= MIN_WINDOWS:
            if best is None or r['subjects'] > best['subjects']:
                best = {**r, 'w_pos': w_pos, 's_pos': s_pos}

if best:
    print(f"\nBest: max_windows_per_subject=({best['w_pos']}, None), max_subjects=({best['s_pos']}, None)")
    print(f"  subjects={best['subjects']}, total={best['total']}, "
          f"pos={best['pos']}, neg={best['neg']}, diff={best['diff']}, min_w={best['min_w']}")
else:
    print("No solution found — relax constraints or handle both classes")

# Show top-N options sorted by subject count
print("\n--- Top options (by subject count) ---")
all_opts = []
for w_pos in range(MIN_WINDOWS, min(max_w + 1, 25)):
    for s_pos in range(50, pos_subj_count + 10, 10):
        r = simulate((w_pos, None), (s_pos, None))
        if r['total'] <= TARGET_TOTAL and r['diff'] <= MAX_CLASS_DIFF and r['min_w'] >= MIN_WINDOWS:
            all_opts.append({**r, 'w_pos': w_pos, 's_pos': s_pos})
all_opts.sort(key=lambda x: (-x['subjects'], x['diff']))
for opt in all_opts[:10]:
    print(f"  ws=({opt['w_pos']}, None), sj=({opt['s_pos']}, None): "
          f"subjs={opt['subjects']}, total={opt['total']}, "
          f"pos={opt['pos']}, neg={opt['neg']}, diff={opt['diff']:4d}")
PYEOF
```

**For classification**, set `TASK_TYPE = 'classification'`. The labels are already
discrete so no median binarization is needed.

**For multi-output regression** (future FC), skip the per-class tuple logic.
Only use flat integers: `max_windows_per_subject=N`, `max_subjects=M`.

## Step 4 — Apply the parameters

Once optimal parameters are found, update the `__main__` block in the
preprocessing file:

```python
if __name__ == '__main__':
    disease_xxx_preprocess("../data/XXX/", hz=200,
                           max_windows_per_subject=(W_POS, None),   # None = no cap for neg class
                           max_subjects=(S_POS, None))
```

Use `None` (not a large int like 999) to mean "no limit" — the existing
`_resolve_param` function handles `None` correctly.

Then run:

```bash
python -m data.xxx.xxx_preprocess
```

## Step 5 — Verify the output

```bash
python3 -c "
import numpy as np
d = np.load('../data/XXX/output.npz', allow_pickle=True)
labels = d['labels']
subj_ids = d['subject_id']
print(f'Samples: {len(labels)}, Subjects: {len(np.unique(subj_ids))}')
# For regression, also show class split at median
median = np.median(labels)
print(f'Old: {(labels >= median).sum()}, Young: {(labels < median).sum()}, diff: {abs((labels >= median).sum() - (labels < median).sum())}')
print(f'Windows/subject: min={(subj_ids == np.unique(subj_ids)[0]).sum() if len(np.unique(subj_ids)) > 0 else 0}')
"
```

## Common constraint presets

| Scenario | `TARGET_TOTAL` | `MAX_CLASS_DIFF` | `MIN_WINDOWS` |
|---|---|---|---|
| Small GPU (≤16GB) | 10,000 | 500 | 5 |
| Medium GPU (24–48GB) | 20,000 | 1,000 | 5 |
| Large GPU (≥80GB) | 50,000 | 2,000 | 3 |
| Transfer learning (few-shot) | 5,000 | 200 | 2 |
