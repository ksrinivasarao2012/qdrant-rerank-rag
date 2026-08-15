"""
Prompt registry loader with variant support.

Every prompt used by the pipeline lives in ``backend/prompts/system_prompts.yaml``
rather than inline in Python. This module is the only thing that reads that file.

Each prompt group holds several named *variants* so competing wordings can exist
side by side and be swept by the evaluation harness. A variant declares only the
keys that differ from its group's ``defaults``; this module merges the two.

Templating uses ``string.Template`` ($variable) instead of ``str.format``
({variable}) so that literal curly braces inside prompt text -- JSON output
examples, LaTeX, code snippets -- need no escaping, and a missing variable
degrades to the raw placeholder instead of raising mid-request.

Usage:
    from backend.core import prompts

    prompts.get("answer")                        # active variant
    prompts.get("answer", "cot_v2")              # a specific variant
    prompts.render("answer", "user", query=q)    # render a template
    prompts.variants("answer")                   # ['cot_v2', 'strict_v1']
    prompts.fingerprint("answer", "cot_v2")      # 'a3f9c1e0' -- tag eval rows
"""

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompts.yaml"

# Keys that describe a variant rather than being part of the prompt itself.
# Excluded from the fingerprint so annotating a variant does not invalidate
# eval rows produced by identical prompt text.
_METADATA_KEYS = {"notes"}

# Top-level registry keys that are not prompt groups.
_RESERVED = {"schema_version", "version"}


@lru_cache(maxsize=1)
def _registry() -> Dict[str, Any]:
    """Loads and caches the YAML registry. Parsed once per process."""
    try:
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Prompt registry not found at {PROMPTS_PATH}. "
            "It must ship alongside the backend package."
        )
    except yaml.YAMLError as e:
        raise ValueError(f"Prompt registry at {PROMPTS_PATH} is not valid YAML: {e}")

    groups = [k for k in data if k not in _RESERVED]
    logger.info(
        f"Loaded prompt registry schema v{data.get('schema_version', '?')} "
        f"({len(groups)} groups: {', '.join(sorted(groups))}) from {PROMPTS_PATH.name}"
    )
    return data


def schema_version() -> Any:
    """Structural revision of the registry file itself."""
    return _registry().get("schema_version")


def _group(name: str) -> Dict[str, Any]:
    """Returns the raw group block, with a helpful error if it is missing."""
    registry = _registry()
    if name not in registry or name in _RESERVED:
        available = sorted(k for k in registry if k not in _RESERVED)
        raise KeyError(
            f"Prompt group '{name}' not found in {PROMPTS_PATH.name}. Available: {available}"
        )
    return registry[name]


def variants(group: str) -> List[str]:
    """
    Every variant name defined for a group, sorted.

    This is what the evaluation harness iterates over to sweep prompt versions.
    """
    return sorted((_group(group).get("variants") or {}).keys())


def active(group: str) -> str:
    """The variant name production uses, per the group's ``active:`` key."""
    block = _group(group)
    name = block.get("active")
    if not name:
        raise KeyError(f"Prompt group '{group}' has no 'active:' key set.")
    return name


@lru_cache(maxsize=64)
def get(group: str, variant: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns the fully merged config for one variant: group defaults overlaid
    with that variant's overrides.

    Includes prompt text plus the model settings it was tuned against
    (model, temperature, max_tokens). Passing ``variant=None`` resolves the
    group's active variant.

    The returned dict is cached and shared -- treat it as read-only.
    """
    block = _group(group)
    name = variant or active(group)

    available = block.get("variants") or {}
    if name not in available:
        raise KeyError(
            f"Variant '{name}' not found in prompt group '{group}'. "
            f"Available: {sorted(available)}"
        )

    merged = dict(block.get("defaults") or {})
    merged.update(available[name] or {})
    merged["_variant"] = name
    return merged


def render(
    group: str,
    key: str = "user",
    variant: Optional[str] = None,
    **kwargs: Any
) -> str:
    """
    Renders a prompt template, substituting $variables with the given kwargs.

    Unknown placeholders are left intact rather than raising, so a typo in a
    variable name shows up visibly in the prompt instead of taking down a request.
    """
    config = get(group, variant)
    if key not in config:
        raise KeyError(
            f"Prompt '{group}.{config['_variant']}.{key}' not found in {PROMPTS_PATH.name}. "
            f"Available keys: {sorted(k for k in config if not k.startswith('_'))}"
        )
    return Template(config[key]).safe_substitute(**kwargs)


@lru_cache(maxsize=64)
def fingerprint(group: str, variant: Optional[str] = None) -> str:
    """
    Short stable hash of a variant's merged content.

    Tag every evaluation row with this. Renaming a variant is cheap, but editing
    its wording changes the fingerprint -- which is what stops last week's scores
    from being silently compared against text that no longer exists.

    Descriptive metadata (``notes``) is excluded, so documenting a variant does
    not invalidate results.
    """
    config = {
        k: v for k, v in get(group, variant).items()
        if k not in _METADATA_KEYS and not k.startswith("_")
    }
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def describe(group: str) -> List[Dict[str, Any]]:
    """
    Summary of every variant in a group: name, whether it is active, its
    fingerprint, model settings, and notes. Useful for eval reports.
    """
    current = active(group)
    rows = []
    for name in variants(group):
        cfg = get(group, name)
        rows.append({
            "variant": name,
            "active": name == current,
            "fingerprint": fingerprint(group, name),
            "model": cfg.get("model"),
            "temperature": cfg.get("temperature"),
            "notes": (cfg.get("notes") or "").strip(),
        })
    return rows


def reload() -> None:
    """Drops all caches so the next access re-reads the file. Handy while tuning."""
    _registry.cache_clear()
    get.cache_clear()
    fingerprint.cache_clear()
    logger.info("Prompt registry caches cleared; will reload on next access.")
