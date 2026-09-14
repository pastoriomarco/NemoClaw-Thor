#!/usr/bin/env python3
"""Apply vLLM PR #52805's XGrammar termination handling.

The pinned Flash-Next preview predates the upstream fix.  Keep this as a small,
exact source transform so the Thor overlay can be reproduced without rebasing
the model-specific preview onto an unrelated vLLM release.
"""

from pathlib import Path
import sys


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new, 1)


if len(sys.argv) != 2:
    raise SystemExit(f"usage: {sys.argv[0]} PATH/TO/backend_xgrammar.py")

path = Path(sys.argv[1])
source = path.read_text()

if "Tokens after termination are ignored." in source:
    print(f"{path}: PR #52805 termination fix already present")
    raise SystemExit(0)

source = replace_once(
    source,
    """        if self._is_terminated:\n            return False\n        for token in tokens:\n""",
    """        if self._is_terminated:\n            return True\n        for token in tokens:\n""",
    "accept-after-termination",
)
source = replace_once(
    source,
    """            self.num_processed_tokens += 1\n        self._is_terminated = self.matcher.is_terminated()\n        return True\n""",
    """            self.num_processed_tokens += 1\n            self._is_terminated = self.matcher.is_terminated()\n            if self._is_terminated:\n                break\n        return True\n""",
    "stop-token-batch",
)
source = replace_once(
    source,
    """        accepted_tokens = []\n        for token in tokens:\n            if self.matcher.accept_token(token):\n                accepted_tokens.append(token)\n            else:\n""",
    """        if self._is_terminated:\n            return []\n\n        accepted_tokens = []\n        for token in tokens:\n            if self.matcher.accept_token(token):\n                accepted_tokens.append(token)\n                if self.matcher.is_terminated():\n                    break\n            else:\n""",
    "validate-after-termination",
)
source = replace_once(
    source,
    """    def reset(self):\n        self.num_processed_tokens = 0\n        self.matcher.reset()\n""",
    """    def reset(self):\n        self.matcher.reset()\n        self.num_processed_tokens = 0\n        self._is_terminated = False\n""",
    "reset-termination-state",
)
source = replace_once(
    source,
    """        Returns True if the FSM was advanced successfully.\n        Returns False if the FSM failed to advance.\n""",
    """        Returns True if all grammar-constrained tokens were accepted.\n        Tokens after termination are ignored. Returns False if the FSM\n        failed to advance.\n""",
    "docstring",
)

path.write_text(source)
print(f"{path}: applied vLLM PR #52805 termination fix")
