"""
This module provides utility functions for generating time intervals.
These intervals are used for retrieving data in chunks or samples.
"""
from datetime import datetime, timedelta
import pandas as pd
import secrets

def generate_chunk_intervals(
    start_dt: datetime, 
    end_dt: datetime, 
    chunk_hours: float = 1.0
) -> list[tuple[int, int]]:
    """
    Generate a list of time intervals (start_timestamp, end_timestamp) pairs.
    These intervals cover the range [start_dt, end_dt) in increments of chunk_hours.

    Args:
        start_dt (datetime): The starting datetime for the interval generation.
        end_dt (datetime): The ending datetime for the interval generation.
        chunk_hours (float): The duration of each interval in hours. Defaults to 1.0.

    Returns:
        list[tuple[int, int]]: A list of tuples, where each tuple contains the start and end timestamps (Unix epoch time) of an interval.
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
    Generate n random time intervals of length interval_hours between start_dt and end_dt.
    Each interval is represented as a tuple (start_timestamp, end_timestamp).

    Args:
        start_dt (datetime): The starting datetime for the sampling window.
        end_dt (datetime): The ending datetime for the sampling window.
        n_samples (int): The number of random intervals to generate.
        interval_hours (float): The length of each interval in hours. Defaults to 24.0.

    Returns:
        list[tuple[int, int]]: A list of tuples, where each tuple contains the start and end timestamps (Unix epoch time) of a sampled interval.

    Raises:
        ValueError: If interval_hours is too large for the given date range.
    """
    total_window_seconds = (end_dt - start_dt).total_seconds()
    interval_seconds = interval_hours * 3600
    
    # Check if the interval is too large for the given date range.
    if interval_seconds > total_window_seconds:
        raise ValueError("interval_hours is too large for the given date range.")
    
    intervals = []
    max_start_seconds = total_window_seconds - interval_seconds
    
    # Generate the intervals.
    for _ in range(n_samples):
        offset_sec = secrets.SystemRandom().uniform(0, max_start_seconds)
        interval_start_dt = start_dt + timedelta(seconds=offset_sec)
        interval_end_dt = interval_start_dt + timedelta(seconds=interval_seconds)
        
        intervals.append((int(interval_start_dt.timestamp()), int(interval_end_dt.timestamp())))
    
    return intervals
