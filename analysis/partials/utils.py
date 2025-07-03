from typing import Literal, Any, Optional
import re
import os

type PartialHTML = Literal["root", "plot", "candidate_information"]

def get_partial_html(partial: PartialHTML) -> str:
    with open(os.path.join(os.path.dirname(__file__), f"{partial}.html"), "r") as file:
        return file.read()

def replace_placeholders(partial: PartialHTML, values: dict[str, Any]) -> str:
    """Construct HTML from a partial template with given keyword arguments."""
    return re.sub(r"\{\{(\w+)\}\}", lambda m: values.get(m.group(1), m.group(0)), get_partial_html(partial))

def generate_attribute_list(items: dict[str, Any], flags: Optional[dict[str, bool]] = None) -> str:
    """Generate a list of attributes from a dictionary."""
    return f"<div class='attr-list'>{''.join([f"<div class='{"flag" if flags and flags.get(key, None) else ""}'><span>{key.replace("_", " ")}:</span> <span>{value}</span></div>" for key, value in items.items()])}</div>"