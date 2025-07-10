import csv
import io
from typing import Sequence

from resonate.models.ResonateSegment import ResonateSegment


def segments_to_csv(segments: Sequence[ResonateSegment]) -> str:
    """
    Given a sequence of ResonateSegment, returns CSV text with columns:
      start,end,text
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    # Write header

    writer.writerow(["start", "end", "text"])
    # Write each segment row
    for seg in segments:
        writer.writerow([seg.start_ts, seg.end_ts, seg.text])
    return buffer.getvalue()
