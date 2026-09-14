"""dump_stage1_cache.py -- build the Stage 1 raw ESM hidden-state cache.

Thin CLI wrapper around src.stage1.cache (mirrors dump_diff_emb.py's shape).
NOT run by this task -- requires downloading the ESM-2 650M checkpoint.

    python dump_stage1_cache.py --manifest data/split_manifest.csv \
        --wt-seq data/wt_sequence.txt --out data/stage1/raw_cache.pt --layers 33
"""

from src.stage1.cache import main

if __name__ == "__main__":
    main()
