"""Two-judge selection helpers (70B guide judge + checklist judge)."""

JUDGE_TAG = "judge"
CHECKLIST_TAG = "checklist"


def _checklist_score(entry):
    verdict = entry.get("checklist")
    if not verdict:
        return 0.0
    return float(verdict.get("score") or 0.0)


def two_judge_cases(data):
    """{"<behavior>|judge": s70, "<behavior>|checklist": sc} over every behavior in the sidecar, or None.

    These are the 2 x N lexicase cases (behavior x judge). Both judges scored the same target response."""
    entries = (data or {}).get("behaviors") or []
    cases = {}
    for entry in entries:
        behavior = entry["behavior"]
        cases[f"{behavior}|{JUDGE_TAG}"] = float(entry.get("score") or 0.0)
        cases[f"{behavior}|{CHECKLIST_TAG}"] = _checklist_score(entry)
    return cases or None


def conservative_score(data):
    """Mean over behaviors of min(70B judge score, checklist score): a response counts only as much as the
    stricter judge credits it. Target failures count as 0. Returns 0.0 for a missing sidecar."""
    entries = (data or {}).get("behaviors") or []
    if not entries:
        return 0.0
    return sum(min(float(e.get("score") or 0.0), _checklist_score(e)) for e in entries) / len(entries)


def judge_agreement(data, threshold=0.5):
    """(both_pass, only_judge, only_checklist, neither) fractions of behaviors, for logging."""
    entries = (data or {}).get("behaviors") or []
    if not entries:
        return (0.0, 0.0, 0.0, 0.0)
    counts = [0, 0, 0, 0]
    for e in entries:
        j = float(e.get("score") or 0.0) >= threshold
        c = _checklist_score(e) >= threshold
        counts[0 if (j and c) else 1 if j else 2 if c else 3] += 1
    return tuple(n / len(entries) for n in counts)


def zscores(values):
    """Population z-scores; all zeros when the values do not vary (nothing to rank on)."""
    values = [float(v) for v in values]
    if not values:
        return []
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return [0.0 if sd < 1e-12 else (v - mean) / sd for v in values]


def zscore_blend(judge_means, checklist_means):
    """Equal-weight blend of the two judges after standardizing each across the population (the checklist judge is
    much stricter and spreads less than the 70B judge, so a raw average is dominated by the 70B judge)."""
    return [(a + b) / 2 for a, b in zip(zscores(judge_means), zscores(checklist_means))]
