from pathlib import Path
import yaml


def load_config(path=None) -> dict:
    """src/config.yaml 로드. path 생략 시 이 파일 기준 상대경로로 탐색."""
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
