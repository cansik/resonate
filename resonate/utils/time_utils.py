def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to a timestamp string in hh:mm:ss format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def samples_to_seconds(samples: float, sample_rate: float) -> float:
    return samples / sample_rate
