"""String-key helpers.

The wedge / line tuples in ``plot_objs`` carry a string label so external
code (notably :class:`InteractiveConnectome` and the notebook examples) can
key into them. We keep that format stable: ``"<label-or-classname> #<id>"``
for the full key, with :func:`display_label` stripping the id and any
``_N`` group suffix for human-facing text.

``id(obj)`` is used (not ``hash(obj)``) so two distinct Python objects can
never collide -- some Nengo objects override ``__hash__`` based on their
parameters, which is unsafe for keying.
"""

from __future__ import annotations

import re

_DISPLAY_LABEL_RE = re.compile(r"^(.*?)(?:\s*#-?\d+|_\d+(?:\s*#-?\d+)?)$")


def get_key(obj) -> str:
    """Stable per-object id: ``"<label or class name> #<id(obj)>"``.

    Falls back to the type name when ``obj.label`` is missing or blank.
    Returns ``"<root>"`` for ``None`` so synthetic roots have a placeholder.
    """
    if obj is None:
        return "<root>"
    label = getattr(obj, "label", None)
    name = label if (label and label.strip()) else type(obj).__name__
    return f"{name} #{id(obj)}"


def display_label(key: str) -> str:
    """Strip ``" #<hash>"`` and / or trailing ``"_<N>"`` from a key string."""
    m = _DISPLAY_LABEL_RE.match(key)
    return m.group(1) if m else key
