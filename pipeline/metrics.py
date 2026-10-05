#Ishaan

"""Calculate all metrics
Baselines, the hard CNN and the soft CNN are all scored by the same code, so
a difference between them can't come from two slightly different formulas.

Four numbers per model, nothing else:
    balanced_accuracy   is the 1-5 rating right? (every rating counts equally)
    auc                 does P(4) + P(5) rank suspicious nodules above the rest?
    ce_vs_readers       do the probabilities match what the radiologists said?
    ece                 when it says 70%, is it right 70% of the time?

Inputs, always per NODULE:
    probs  (N, 5) float   predicted probability of rating 1..5 (rows sum to 1)
    hard   (N,)   int     median reader rating, 1..5
    soft   (N, 5) float   share of readers who gave each rating
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

EPS = 1e-7


def suspicious(hard):
    # Binary ground truth: median rating 4-5 = suspicious, 1-3 = not.
    return (np.asarray(hard) >= 4).astype(int)


def p_suspicious(probs):
    # Binary score from either model: P(4) + P(5), the chance the median rating is 4 or 5.
    return probs[:, 3] + probs[:, 4]


def ece(probs, hard, n_bins=10):
    """Expected calibration error, top-label version.

    Bin nodules by the model's confidence (its top probability). In each bin
    compare average confidence with how often the top class was right. ECE is
    the size-weighted average gap: 0 = says 70% and is right 70% of the time.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) + 1) == hard
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return total


def scores(probs, hard, soft):
    """The four numbers reported for every model and baseline."""
    pred = probs.argmax(axis=1) + 1
    y = suspicious(hard)
    return {
        "balanced_accuracy": balanced_accuracy_score(hard, pred),
        "auc": roc_auc_score(y, p_suspicious(probs)) if 0 < y.sum() < len(y) else float("nan"),
        # the soft-label metric: lower = closer to the whole panel of readers
        "ce_vs_readers": -(soft * np.log(np.clip(probs, EPS, 1))).sum(axis=1).mean(),
        "ece": ece(probs, hard),
    }


def confidence_by_spread(probs, spread):
    """Figure 1: mean top probability for each level of reader disagreement."""
    conf = probs.max(axis=1)
    return {int(s): (float(conf[spread == s].mean()), int((spread == s).sum()))
            for s in np.unique(spread)}


def bootstrap_diff(metric, a_runs, b_runs, hard, soft, n=2000, seed=0, groups=None):
    """95% CI for metric(model A) - metric(model B) on the same nodules.

    a_runs, b_runs: (runs, N, 5) -- predictions of each model on the same N
            nodules. With folds this is (1, 1885, 5): every nodule's prediction
            from the run that tested it, pooled.
    metric: function(probs, hard, soft) -> float, e.g.
            lambda p, h, s: scores(p, h, s)["ce_vs_readers"]
    groups: the patient_id of each nodule. Pass it with folds: one patient can
            have up to 13 nodules, and those aren't independent, so each
            resample draws whole patients with replacement, not single nodules.
    Each resample scores both models on the SAME drawn nodules and takes A - B.
    Paired, so how hard the drawn nodules happen to be cancels out. If the
    interval contains 0, you can't claim a difference.
    """
    rng = np.random.default_rng(seed)
    N = len(hard)
    if groups is None:
        members = [np.array([i]) for i in range(N)]
    else:
        _, which = np.unique(np.asarray(groups), return_inverse=True)
        members = [np.flatnonzero(which == g) for g in range(which.max() + 1)]
    diffs = []
    for _ in range(n):
        i = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        a = np.mean([metric(p[i], hard[i], soft[i]) for p in a_runs])
        b = np.mean([metric(p[i], hard[i], soft[i]) for p in b_runs])
        diffs.append(a - b)
    return float(np.mean(diffs)), tuple(np.percentile(diffs, [2.5, 97.5]).round(4))
