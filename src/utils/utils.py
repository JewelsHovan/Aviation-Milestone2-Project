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

def sample_dates(start, stop, n):
    """
    Sample n random datetime objects between start and stop.

    Parameters:
        start (datetime): The beginning of the interval.
        stop (datetime): The end of the interval.
        n (int): Number of random datetime samples to generate.

    Returns:
        list of datetime: A list containing n random datetime objects between start and stop.
    """
    if start >= stop:
        raise ValueError("start must be earlier than stop.")

    # Compute the total duration in seconds between start and stop
    total_seconds = (stop - start).total_seconds()

    # Generate n random datetime objects
    random_dates = [
        start + timedelta(seconds=secrets.SystemRandom().uniform(0, total_seconds))
        for _ in range(n)
    ]
    return random_dates

# Example usage:
if __name__ == "__main__":
    start_date = datetime(2025, 1, 1)
    stop_date = datetime(2025, 12, 31)
    samples = sample_dates(start_date, stop_date, 5)
    for d in samples:
        print(d)
