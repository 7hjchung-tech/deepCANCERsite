import sys
from pathlib import Path

# deepRAD51C/ を sys.path に追加 — "import esm" と "from src.embeddings import ..." 양쪽 해결
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
