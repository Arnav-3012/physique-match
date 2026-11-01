from typing import Protocol

from physique_schemas.models import GapProfile, Scan


class Analyzer(Protocol):
    """Fixed interface (CLAUDE.md): geometric v1 and Siamese v2 both implement it."""

    def compare(self, goal: Scan, current: Scan) -> GapProfile: ...
