import json
from pathlib import Path
from typing import Any, Dict


def load_project_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_setting(config: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    section_values = config.get(section, {})
    if not isinstance(section_values, dict):
        return default
    return section_values.get(key, default)


def get_path(config: Dict[str, Any], key: str, default: str) -> Path:
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        return Path(default)
    return Path(paths.get(key, default))
