"""Prompt-template helpers used by the meeting-minutes interface."""

import json
import re
from pathlib import Path


PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
PROMPT_VARIABLES = ("contexte", "mots_particuliers", "transcript")


def render_prompt(template, values):
    """Replace known ``{variables}`` without treating other braces as errors."""

    return PLACEHOLDER_PATTERN.sub(
        lambda match: values.get(match.group(1), match.group(0)), template
    )


def prompt_variables(template):
    """Return the variables used in a template, in their first-use order."""

    return list(dict.fromkeys(PLACEHOLDER_PATTERN.findall(template)))


def format_special_terms(terms):
    """Turn a one-term-per-line field into a clear prompt block."""

    cleaned_terms = [term.strip() for term in terms.splitlines() if term.strip()]
    if not cleaned_terms:
        return "Aucun terme particulier n'a été fourni."
    return "\n".join(f"- {term}" for term in cleaned_terms)


def load_saved_templates(path):
    """Load user templates, returning an empty mapping for a new or invalid file."""

    path = Path(path)
    if not path.is_file():
        return {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(content, dict):
        return {}
    return {
        name: template
        for name, template in content.items()
        if isinstance(name, str) and isinstance(template, str)
    }


def save_templates(path, templates):
    """Persist user templates as readable UTF-8 JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(templates, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
