# time_utils.py
from datetime import datetime, timedelta
import secrets

def parse_date(date_str: str) -> datetime:
    """
    Parse a date string in either YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.
    """
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format for {date_str}.")

def sample_dates(start_dt: datetime, end_dt: datetime, n: int) -> list[datetime]:
    """
    Sample n random datetime objects between start_dt and end_dt (inclusive).
    """
    if start_dt >= end_dt:
        raise ValueError("start_dt must be earlier than end_dt.")
    
    total_seconds = (end_dt - start_dt).total_seconds()
    random_dates = []
    for _ in range(n):
        offset = secrets.SystemRandom().uniform(0, total_seconds)
        random_dates.append(start_dt + timedelta(seconds=offset))
    return random_dates
