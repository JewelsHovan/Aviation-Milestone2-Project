from data_loader import OpenSkyLoader
from datetime import datetime, timedelta
import pandas as pd
import os
import logging
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_data_dir(base_dir='data'):
    """Create data directory if it doesn't exist"""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def get_random_dates(start_date: datetime, end_date: datetime, n_samples: int = 100) -> list:
    """
    Generate random dates between start_date and end_date
    
    Args:
        start_date: Start date
        end_date: End date
        n_samples: Number of random dates to generate
        
    Returns:
        List of random datetime objects
    """
    date_range = (end_date - start_date).days
    random_dates = []
    
    for _ in range(n_samples):
        # Generate random number of days to add
        days_to_add = secrets.SystemRandom().randint(0, date_range)
        # Generate random hour (0-23)
        random_hour = secrets.SystemRandom().randint(0, 23)
        
        # Create random datetime
        random_date = start_date + timedelta(days=days_to_add)
        random_date = random_date.replace(hour=random_hour, minute=0, second=0)
        random_dates.append(random_date)
    
    # Sort dates for better organization
    return sorted(random_dates)

def process_sample(sample_time: datetime, airport: str = 'KATL', data_dir: str = 'data', sample_duration: int = 1, limit: int = 1000):
    """
    Process and save data for a single time sample
    
    Args:
        sample_time: Start time for the sample
        airport: Airport ICAO code
        data_dir: Directory to save data
        sample_duration: Duration in hours to sample
        limit: Maximum number of records to return per sample
    """
    # Calculate end time
    end_time = sample_time + timedelta(hours=sample_duration)
    time_key = sample_time.strftime('%Y%m%d_%H%M')
    
    logger.info(f"Processing sample for {time_key}...")
    
    # Initialize loader and get data
    loader = OpenSkyLoader(request_delay=2.0)
    try:
        data = loader.get_flight_data(
            start_time=sample_time,
            end_time=end_time,
            airport=airport,
            region='georgia',
            limit=limit
        )
        
        if data.empty:
            logger.warning(f"No data found for {time_key}")
            return
        
        # Save the data
        output_file = os.path.join(data_dir, f'flight_data_{time_key}.csv')
        data.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(data)} records for {time_key}")
        
    except Exception as e:
        logger.error(f"Error processing {time_key}: {str(e)}")

if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = ensure_data_dir()
    
    # Set date range for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Generate random dates
    random_dates = get_random_dates(start_date, end_date, n_samples=100)
    
    logger.info(f"Processing {len(random_dates)} random samples from "
               f"{start_date.date()} to {end_date.date()}")
    
    # Process each random date
    for sample_time in random_dates:
        process_sample(sample_time, data_dir=data_dir)
        
    logger.info("Processing complete!")
