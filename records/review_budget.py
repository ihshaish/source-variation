"""Error capture against review budget on ASRS, from the stored predictions of
the word2vec BiLSTM models. Two ways of ranking held-out records for review
are compared per training: records where the narrative and synopsis models
disagree first (lowest synopsis confidence first within that group), and
lowest synopsis confidence alone. For budgets from 1% to 30% of the held-out
set, the fraction of the synopsis model's errors captured is recorded, with
the area under each curve. Reads preds_syn_w2vview_s*.npz and
preds_narr_w2vview_s*.npz; writes review_budget.json.

    python3 review_budget.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
trapezoid = getattr(np, "trapezoid", None) or np.trapz

budgets = [round(b, 3) for b in np.arange(0.01, 0.301, 0.01)]
result = {"budgets": budgets}
for training in (0, 1, 2):
    synopsis = np.load(os.path.join(HERE, f"preds_syn_w2vview_s{training}.npz"))
    narrative = np.load(os.path.join(HERE, f"preds_narr_w2vview_s{training}.npz"))
    y = synopsis["y"]
    probs_syn = synopsis["probs"]
    probs_narr = narrative["probs"]
    class_syn = (probs_syn >= 0.5).astype(int)
    class_narr = (probs_narr >= 0.5).astype(int)
    errors = class_syn != y
    n = len(y)
    n_errors = errors.sum()
    disagree = class_syn != class_narr
    confidence = np.abs(probs_syn - 0.5)
    by_disagreement = np.lexsort((confidence, ~disagree))
    by_confidence = np.argsort(confidence)
    capture_disagreement = []
    capture_confidence = []
    for budget in budgets:
        k = int(budget * n)
        capture_disagreement.append(round(float(errors[by_disagreement[:k]].sum() / n_errors), 4))
        capture_confidence.append(round(float(errors[by_confidence[:k]].sum() / n_errors), 4))
    auc_disagreement = float(trapezoid(capture_disagreement, budgets))
    auc_confidence = float(trapezoid(capture_confidence, budgets))
    result[f"seed{training}"] = {"disagreement_first": capture_disagreement,
                                 "low_confidence": capture_confidence,
                                 "auc_disagreement": round(auc_disagreement, 4),
                                 "auc_confidence": round(auc_confidence, 4)}
    print(f"training {training + 1}: area under the curve {auc_disagreement:.4f} against {auc_confidence:.4f}; "
          f"capture at 10% {capture_disagreement[9]:.3f} against {capture_confidence[9]:.3f}")
json.dump(result, open(os.path.join(HERE, "review_budget.json"), "w"), indent=1)
