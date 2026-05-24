"""Plot helpers — keep titles labeled Simulated/Measured/Hybrid. Phase 6."""

from __future__ import annotations

from typing import Literal

Provenance = Literal["Simulated", "Measured", "Hybrid"]


def title_with_provenance(base: str, provenance: Provenance) -> str:
    return f"{base}  ·  {provenance}"
