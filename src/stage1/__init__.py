"""Stage 1 input plumbing over the validated frozen ESM-2 representation cache.

WHAT THIS PACKAGE IS
--------------------
Everything needed to feed the Task C frozen cache into a token-level Stage 1
model, and nothing else:

    interface.py  the exact Stage 1 input contract (Stage1Batch / Stage1Model)
    data.py       provenance verification, cohort binding, dataset + collate
    targets.py    train-only target standardisation and its inverse
    losses.py     the two candidate objectives, and the MSE metric they are
                  not to be confused with
    metrics.py    Spearman / MAE / RMSE / MSE with explicit undefined handling
    protocol.py   the fixed pilot training+evaluation protocol (not tuned)
    mock.py       a deliberately trivial consumer that proves tensors flow

WHAT THIS PACKAGE IS NOT
------------------------
It is NOT a Stage 1 model. The team's real token-level Stage 1 implementation
remains an external dependency; see docs/ESM_PROGRESS.md. `mock.py` exists only
to prove the interface is connectable and must never be reported as a result.
"""
