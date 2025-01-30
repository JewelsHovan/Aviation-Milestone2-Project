"""
Flight Data Sampler
------------------
Collects sample flight data from OpenSky for each month in the past year.
Uses the OpenSkyLoader to fetch both state vectors and flight information.
"""

from data_loader import OpenSkyLoader
from datetime import datetime, timedelta
import pandas as pd
import logging
from pathlib import Path
import secrets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_sample_dates(months_back: int = 12, samples_per_month: int = 2) -> list[datetime]:
    """
    Generate random sample dates for each month going back from current date
    
    Args:
        months_back: Number of months to go back
        samples_per_month: Number of random days to sample per month
    
    Returns:
        List of datetime objects representing sample dates
    """
    current_date = datetime.now()
    sample_dates = []
    
    for month_offset in range(months_back):
        # Get first day of each month
        current_month = current_date - timedelta(days=current_date.day - 1) - timedelta(days=30 * month_offset)
        
        # Get number of days in the month
        if month_offset == 0:  # Current month
            days_in_month = current_date.day - 1  # Don't include today
        else:
            next_month = current_month.replace(day=28) + timedelta(days=4)
            days_in_month = (next_month - timedelta(days=next_month.day)).day
        
        # Get random days from the month
        random_days = sorted(secrets.SystemRandom().sample(range(1, days_in_month + 1), min(samples_per_month, days_in_month)))
        
        # Create datetime objects for each sample day
        for day in random_days:
            sample_date = current_month.replace(day=day)
            sample_dates.append(sample_date)
    
    return sorted(sample_dates)

def collect_flight_samples(
    output_dir: str = "flight_samples",
    region: str = None,
    samples_per_month: int = 2,
    sample_duration_hours: int = 1
) -> None:
    """
    Collect flight samples for each month in the past year
    
    Args:
        output_dir: Directory to save the sample data
        region: Region name from config to filter data
        samples_per_month: Number of samples to collect per month
        sample_duration_hours: Duration of each sample in hours
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize loader
    loader = OpenSkyLoader(default_region=region)
    
    # Get sample dates
    sample_dates = get_sample_dates(samples_per_month=samples_per_month)
    
    for sample_date in sample_dates:
        try:
            # Format date for filenames
            date_str = sample_date.strftime("%Y%m%d_%H%M")
            
            # Set time window
            start_time = sample_date.replace(hour=secrets.SystemRandom().randint(6, 20), minute=0)  # Random hour between 6AM and 8PM
            end_time = start_time + timedelta(hours=sample_duration_hours)
            
            logger.info(f"Collecting sample for {start_time} to {end_time}")
            
            # Get state vectors
            state_vectors = loader.load_state_vectors(
                start_time=start_time,
                end_time=end_time,
                selected_columns=[
                    'time', 'icao24', 'callsign', 'lat', 'lon',
                    'velocity', 'heading', 'geoaltitude', 'onground'
                ]
            )
            
            if not state_vectors.empty:
                state_vectors.to_csv(
                    output_path / f"state_vectors_{date_str}.csv",
                    index=False
                )
                logger.info(f"Saved {len(state_vectors)} state vectors")
            
            # Get flight data
            flights = loader.load_flights(
                start_time=start_time,
                end_time=end_time
            )
            
            if not flights.empty:
                flights.to_csv(
                    output_path / f"flights_{date_str}.csv",
                    index=False
                )
                logger.info(f"Saved {len(flights)} flights")
            
        except Exception as e:
            logger.error(f"Failed to collect sample for {sample_date}: {str(e)}")
            continue

def analyze_samples(data_dir: str = "flight_samples") -> None:
    """
    Analyze the collected samples and print summary statistics
    
    Args:
        data_dir: Directory containing the sample data
    """
    data_path = Path(data_dir)
    
    # Collect all CSV files
    state_vector_files = list(data_path.glob("state_vectors_*.csv"))
    flight_files = list(data_path.glob("flights_*.csv"))
    
    logger.info("\nSample Analysis:")
    logger.info(f"Found {len(state_vector_files)} state vector samples")
    logger.info(f"Found {len(flight_files)} flight samples")
    
    # Analyze state vectors
    if state_vector_files:
        total_states = 0
        unique_aircraft = set()
        
        for file in state_vector_files:
            df = pd.read_csv(file)
            total_states += len(df)
            unique_aircraft.update(df['icao24'].unique())
        
        logger.info(f"\nState Vector Statistics:")
        logger.info(f"Total positions: {total_states}")
        logger.info(f"Unique aircraft: {len(unique_aircraft)}")
    
    # Analyze flights
    if flight_files:
        total_flights = 0
        unique_airports = set()
        
        for file in flight_files:
            df = pd.read_csv(file)
            total_flights += len(df)
            if 'estdepartureairport' in df.columns:
                unique_airports.update(df['estdepartureairport'].dropna())
            if 'estarrivalairport' in df.columns:
                unique_airports.update(df['estarrivalairport'].dropna())
        
        logger.info(f"\nFlight Statistics:")
        logger.info(f"Total flights: {total_flights}")
        logger.info(f"Unique airports: {len(unique_airports)}")

if __name__ == "__main__":
    # Collect samples
    collect_flight_samples(
        output_dir="flight_samples",
        region="georgia",  # Optional: specify region from config
        samples_per_month=2,
        sample_duration_hours=1
    )
    
    # Analyze results
    analyze_samples("flight_samples")
