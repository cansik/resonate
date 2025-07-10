from dataclasses import dataclass


@dataclass
class ResonateSegment:
    start_ts: float
    end_ts: float
    text: str
