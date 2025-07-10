from dataclasses import dataclass

from resonate.models.ResonateSegment import ResonateSegment
from resonate.models.ResonateTiming import ResonateTiming


@dataclass
class ResonateResult:
    timing: ResonateTiming
    segments: list[ResonateSegment]
