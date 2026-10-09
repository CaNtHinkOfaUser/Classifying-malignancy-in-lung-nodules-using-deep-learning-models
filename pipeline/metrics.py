#Ishaan

"""Calculate all metrics
Baselines, the hard CNN and the soft CNN are all scored by the same code, so
a difference between them can't come from two slightly different formulas.

Four numbers per model, nothing else:
    balanced_accuracy   is the 1-5 rating right? (every rating counts equally)
    auc                 does P(4) + P(5) rank suspicious nodules above the rest?
    ce_vs_readers       do the probabilities match what the radiologists said?
    ece                 when it says 70%, is it right 70% of the time?

Extras, outside scores() so the four headline numbers stay as they are:
    quadratic_kappa     agreement with the median rating, the measure radiologists'
                        agreement with each other is usually reported in
    youden_threshold    picks the suspicious cut-off on VAL...
    binary              ...which is then applied unchanged to TEST: sensitivity,
                        specificity and the rest

Inputs, always per NODULE:
    probs  (N, 5) float   predicted probability of rating 1..5 (rows sum to 1)
    hard   (N,)   int     median reader rating, 1..5
    soft   (N, 5) float   share of readers who gave each rating
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score

EPS = 1e-7


def suspicious(hard_labels):
    # Binary ground truth: median rating 4-5 = suspicious, 1-3 = not.
    return (np.asarray(hard_labels) >= 4).astype(int)


def p_suspicious(predicted_probabilities):
    # Binary score from either model: P(4) + P(5), the chance the median rating is 4 or 5.
    return predicted_probabilities[:, 3] + predicted_probabilities[:, 4]


def ece(probs, hard, n_bins=10):
    """Expected calibration error (ece), top-label version.

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



def quadratic_kappa(probs, hard):
    """Agreement between the predicted rating and the median rating, beyond chance.

    1 = perfect, 0 = no better than chance. Quadratic weights: calling a 5 a 4
    costs far less than calling it a 1.
    """
    pred = probs.argmax(axis=1) + 1
    return cohen_kappa_score(hard, pred, weights="quadratic", labels=[1, 2, 3, 4, 5])


def youden_threshold(score, y):
    """The cut-off that maximises sensitivity + specificity - 1. Pick it on VAL, apply it to TEST.

    score: p_suspicious(probs);  y: suspicious(hard).
    """
    best_t, best_j = 0.5, -1.0
    for t in np.unique(score):
        pred = score >= t
        j = pred[y == 1].mean() + (~pred[y == 0]).mean() - 1
        if j > best_j:
            best_t, best_j = t, j
    return float(best_t)


def binary(score, y, threshold):
    """Suspicious-vs-not at a fixed cut-off: what a clinician would ask.

    score: p_suspicious(probs);  y: suspicious(hard);  threshold: from youden_threshold on VAL.
    """
    pred = (np.asarray(score) >= threshold).astype(int)
    y = np.asarray(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "threshold": threshold,
        "sensitivity": tp / max(tp + fn, 1),     # share of suspicious nodules caught
        "specificity": tn / max(tn + fp, 1),     # share of non-suspicious ones correctly cleared
        "precision": tp / max(tp + fp, 1),       # of those flagged, share really suspicious
        "npv": tn / max(tn + fn, 1),             # of those cleared, share really not suspicious
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
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
