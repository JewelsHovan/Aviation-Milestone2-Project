# interval_generation.py
from datetime import datetime, timedelta
import math
import pandas as pd
import secrets

def generate_chunk_intervals(
    start_dt: datetime, 
    end_dt: datetime, 
    chunk_hours: float = 1.0
) -> list[tuple[int, int]]:
    """
    Generate a list of (start_timestamp, end_timestamp) pairs 
    that cover [start_dt, end_dt) in increments of chunk_hours.
    """
    intervals = []
    current_dt = start_dt
    
    while current_dt < end_dt:
        interval_start = int(current_dt.timestamp())
        next_dt = current_dt + pd.Timedelta(hours=chunk_hours)
        if next_dt > end_dt:
            next_dt = end_dt
        interval_end = int(next_dt.timestamp())
        
        intervals.append((interval_start, interval_end))
        current_dt = next_dt
    
    return intervals

def generate_sample_intervals(
    start_dt: datetime,
    end_dt: datetime,
    n_samples: int,
    interval_hours: float = 24.0
) -> list[tuple[int, int]]:
    """
    Generate n random intervals of length interval_hours between start_dt and end_dt.
    Each interval is (start_timestamp, end_timestamp).
    """
    total_window_seconds = (end_dt - start_dt).total_seconds()
    interval_seconds = interval_hours * 3600
    
    if interval_seconds > total_window_seconds:
        raise ValueError("interval_hours is too large for the given date range.")
    
    intervals = []
    # The latest start point is total_window_seconds - interval_seconds
    max_start_seconds = total_window_seconds - interval_seconds
    
    for _ in range(n_samples):
        offset_sec = secrets.SystemRandom().uniform(0, max_start_seconds)
        interval_start_dt = start_dt + timedelta(seconds=offset_sec)
        interval_end_dt = interval_start_dt + timedelta(seconds=interval_seconds)
        
        intervals.append((int(interval_start_dt.timestamp()), int(interval_end_dt.timestamp())))
    
    return intervals
