# Raw measurement data

All numbers in the paper come from these CSVs. One row = one (object, strategy, K, frame) run.

| Column | Meaning |
|---|---|
| `obj` | YCB-Video object id (`ob_0000001` … `ob_0000021`) |
| `strategy` | `random`, `farthest`, or `dino` |
| `top_k` | number of retained hypotheses K |
| `frame` | evaluated frame |
| `time_ms` | end-to-end frame-0 registration time |
| `add_cm`, `adds_cm` | ADD / ADD-S error in cm |
| `add_pass`, `adds_pass` | error < 0.1 × object diameter |
| `rot_err`, `t_err_cm` | rotation (deg) and translation (cm) error vs ground truth |
| `diam_cm` | object diameter |
| `ok` | run completed without exception |

| File | Scope |
|---|---|
| `breadth.csv` | 21 objects × {random, farthest} × K ∈ {1,3,5,10,20,50} |
| `decisive_ob*.csv` | 8 objects × all three selectors × same K sweep |
| `anchor_full.csv` | full grid (K=252) on 5 representative objects |

Note: `anchor_full.csv` covers only 5 objects, so any comparison against the full
grid must be made on that matched subset — not against the 21-object mean.
