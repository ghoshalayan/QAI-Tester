"""Pattern packs — curated knowledge primers for common UX families.

Loaded into AKB the first time a target_url is detected as belonging
to one of the families. Each pack is a ``PatternPack`` dataclass with
match heuristics + a list of named rules. The runtime queries the
AKB; pattern rules surface alongside BRD chunks + recon notes +
disputes in the agent's submodule prompt.

Authoring style
---------------
Rules are short, specific, and actionable. Bias toward *how the
agent should change behavior* over generic "here's how Salesforce
works" prose.

Examples of good rules:
- "On SAP Fiori, click targets often live in shadow DOM under
  ``ui5-shellbar``; coord-click usually beats DOM resolution."
- "Salesforce dialogs have role='dialog' but the Save button is
  inside a slot — fuzzy resolver finds it as 'Save Changes'."

Examples of bad rules (avoid):
- "Salesforce is a CRM platform" — too generic, doesn't change
  agent behavior.
"""

from .registry import PATTERN_PACKS, PatternPack, autoload_pack
from .registry import detect_pack as detect_pattern_pack

__all__ = [
    "PATTERN_PACKS",
    "PatternPack",
    "autoload_pack",
    "detect_pattern_pack",
]
