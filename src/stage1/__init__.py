"""Stage 1: frozen-ESM WT/Mut representation comparison (paired_delta /
branched_projection / unified_reference_delta).

See README_STAGE1.md for the full design. This package is intentionally
independent of the legacy M1-M4 path (model.py / train.py / dataset.py /
src/embeddings/*) -- it reuses only the HGVS parser in dataset.py and the
frozen ESMEncoder in src/embeddings/esm_encoder.py, both by explicit import,
never by copy-paste of their internals.
"""
