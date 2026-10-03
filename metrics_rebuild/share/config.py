from __future__ import annotations

from typing import Optional

from . import lemma_implication
from . import verus_runner


def set_verus_binary(path: Optional[str]) -> None:
    verus_runner.set_verus_binary(path)
    lemma_implication.set_lemma_verus_binary(path)

__all__ = ["set_verus_binary"]
