"""Check the configured behavior pool before starting the GPU services."""

import jb_config
from eval import load_behavior_rows, sample_behaviors


def main():
    rows = load_behavior_rows()
    picked = sample_behaviors([behavior for _, behavior, _ in rows])
    minimum = jb_config.n_behaviors_per_eval()
    expected = len(rows) if jb_config.use_full_behaviors() and not jb_config.use_stratified() else minimum
    if len(rows) < minimum:
        raise ValueError(f"Behavior pool has {len(rows)} entries; at least {minimum} required")
    if len(picked) != expected:
        raise ValueError(f"Expected {expected} behaviors per gene, got {len(picked)}")
    if len(set(picked)) != expected:
        raise ValueError("Behavior sample contains duplicate text")
    print(f"Behavior preflight passed: {len(picked)} unique behaviors per gene "
          f"from {jb_config.behaviors_path()}", flush=True)


if __name__ == "__main__":
    main()
