"""Helper utilities for the self-improving LLM agent."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML file and return as dictionary.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary with YAML content
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        file_path: Path to output YAML file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file and return as dictionary.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary with JSON content
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        file_path: Path to output JSON file
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def get_env_var(key: str, default: Optional[str] = None) -> str:
    """
    Get environment variable value.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} not set and no default provided")
    return value


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_list(items: List[Any], separator: str = "\n") -> str:
    """
    Format list as string with separator.

    Args:
        items: List of items
        separator: Separator between items

    Returns:
        Formatted string
    """
    return separator.join(str(item) for item in items)


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.

    Args:
        text: Input string
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def validate_json_schema(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that dictionary contains all required keys.

    Args:
        data: Dictionary to validate
        required_keys: List of required keys

    Returns:
        True if valid, False otherwise
    """
    return all(key in data for key in required_keys)


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON string, return None if invalid.

    Args:
        text: JSON string

    Returns:
        Parsed dictionary or None
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text (handles markdown code blocks).

    Args:
        text: Text potentially containing JSON

    Returns:
        Extracted JSON dictionary or None
    """
    # Try to find JSON in markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        json_str = text[start:end].strip()
        return safe_json_parse(json_str)

    # Try to find JSON in regular code blocks
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        json_str = text[start:end].strip()
        return safe_json_parse(json_str)

    # Try to parse entire text as JSON
    return safe_json_parse(text)


# from pathlib import Path
# import yaml

# def load_yaml(path: str) -> dict:
#     config_path = Path(path)
#     if not config_path.exists():
#         raise FileNotFoundError(f"Config file not found: {config_path}")

#     with open(config_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)

# def get_project_root()->Path:
#     return Path(__file__).resolve().parent.parent.parent
