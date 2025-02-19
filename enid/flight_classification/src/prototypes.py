import numpy as np
from dtaidistance import dtw_ndim
from scipy.interpolate import interp1d


# Commercial Flight: Straight path with minimal fluctuations
def commercial_prototype():
    t = np.linspace(0, 30, 30)  # 30 minutes, sampled every minute
    lats = np.linspace(33.7490, 33.7550, 30)  # Small change in latitude
    lons = np.linspace(-84.3880, -84.3800, 30)  # Small change in longitude
    altitudes = np.full(30, 35000)  # Constant altitude
    speeds = np.full(30, 450)  # Constant speed

    return list(zip(lats, lons, altitudes, speeds))

# Training Flight: Erratic and abstract path (e.g., zigzag or grid-like)
def training_prototype():
    t = np.linspace(0, 30, 30)
    # Make sure the first and last coordinates are the same
    lats = 33.7500 + 0.01 * np.random.randn(30)  # Random noise to simulate erratic lat movements
    lons = -84.3870 + 0.01 * np.random.randn(30)  # Random noise to simulate erratic lon movements
    lats[0] = lats[-1] = 33.7500  # Set the first and last lat the same
    lons[0] = lons[-1] = -84.3870  # Set the first and last lon the same
    altitudes = np.linspace(5000, 5000, 30)  # Constant low altitude
    speeds = np.linspace(100, 120, 30)  # Gradual speed change

    return list(zip(lats, lons, altitudes, speeds))


# Surveillance Flight: Grid-like path with periodic back-and-forth movement
def surveillance_prototype():
    t = np.linspace(0, 30, 30)
    lats = 33.7500 + 0.005 * np.floor(t / 5)  # Grid-like lat pattern
    lons = -84.3870 + 0.005 * (t % 5)  # Grid-like lon pattern
    altitudes = np.full(30, 12000)  # Constant moderate altitude
    speeds = np.full(30, 200)  # Constant moderate speed

    return list(zip(lats, lons, altitudes, speeds))

# Cargo Flight: Slight, realistic variations in altitude and speed
def cargo_prototype():
    t = np.linspace(0, 30, 30)
    lats = 33.7485 + 0.008 * np.sin(0.1 * t)  # Small lat variations
    lons = -84.3875 + 0.008 * np.cos(0.1 * t)  # Small lon variations
    altitudes = np.linspace(28000, 30000, 30)  # Gradual increase in altitude
    speeds = np.linspace(330, 350, 30)  # Gradual speed increase

    return list(zip(lats, lons, altitudes, speeds))

# Emergency Flight: Rapid directional changes with varying altitude
def emergency_prototype():
    t = np.linspace(0, 30, 30)
    lats = 33.7500 + np.cumsum(np.random.randn(30)) * 0.005  # Cumulative random lat changes
    lons = -84.3870 + np.cumsum(np.random.randn(30)) * 0.005  # Cumulative random lon changes
    altitudes = np.linspace(8000, 8500, 30)  # Gradual altitude increase
    speeds = np.linspace(160, 200, 30)  # Gradual speed increase

    return list(zip(lats, lons, altitudes, speeds))

# Private Flight: Relaxed, gradual turns but steady trajectory overall
def private_prototype():
    t = np.linspace(0, 30, 30)
    lats = 33.7495 + 0.004 * np.sin(0.05 * t)  # Slight curvature
    lons = -84.3875 + 0.004 * np.cos(0.05 * t)  # Slight curvature
    altitudes = np.linspace(2000, 2200, 30)  # Gradual altitude change
    speeds = np.linspace(150, 160, 30)  # Gradual speed increase

    return list(zip(lats, lons, altitudes, speeds))

# Aerobatic Flight: Sharp turns and rapid altitude changes
def aerobatic_prototype():
    t = np.linspace(0, 30, 30)
    lats = 33.7500 + 0.02 * np.random.randn(30)  # Frequent large latitude changes
    lons = -84.3870 + 0.02 * np.random.randn(30)  # Frequent large longitude changes
    altitudes = np.linspace(1000, 1200, 30)  # Rapid altitude changes
    speeds = np.linspace(400, 450, 30)  # High and increasing speed

    return list(zip(lats, lons, altitudes, speeds))


# Smooth Path Function (for optional smoothing)
def smooth_path(path, window_size=3):
    smoothed_path = []
    for i in range(len(path)):
        # Smooth over a sliding window
        start = max(0, i - window_size // 2)
        end = min(len(path), i + window_size // 2 + 1)
        smoothed_point = np.mean(path[start:end], axis=0)
        smoothed_path.append(smoothed_point)
    return smoothed_path


def normalize_path(path, num_points=100, weights=None):
    """Resamples a path with latitude, longitude, altitude, and velocity."""
    path = np.array(path)
    x = np.linspace(0, 1, len(path))

    # Interpolate each feature separately
    f_lat = interp1d(x, path[:, 0], kind='linear', fill_value="extrapolate")
    f_lon = interp1d(x, path[:, 1], kind='linear', fill_value="extrapolate")
    f_alt = interp1d(x, path[:, 2], kind='linear', fill_value="extrapolate")
    f_vel = interp1d(x, path[:, 3], kind='linear', fill_value="extrapolate")  # Velocity feature

    x_new = np.linspace(0, 1, num_points)
    resampled_path = np.column_stack((f_lat(x_new), f_lon(x_new), f_alt(x_new), f_vel(x_new)))

    if weights is not None:
        resampled_path *= weights  # Apply feature weights

    return resampled_path



# Function to calculate DTW distance
def calculate_dtw_distance(path1, path2, window=5):
    """Computes DTW distance with a warping constraint."""
    return dtw_ndim.distance(np.array(path1), np.array(path2), window=window)



def classify_flight_path(path, speed_weight=2.0, altitude_weight=2.0):
    """Classifies a flight path using weighted DTW.""" 
    # Define prototype flight paths
    prototypes = {
        'Commercial Flight': commercial_prototype(),
        'Training Flight': training_prototype(),
        'Surveillance Flight': surveillance_prototype(),
        'Private Flight': private_prototype(),
    }

    # Define weights (higher weight on speed)
    feature_weights = np.array([1.0, 1.0, altitude_weight, speed_weight])  

    # Normalize and apply weights
    norm_path = normalize_path(path, weights=feature_weights)
    norm_prototypes = {label: normalize_path(prototype, weights=feature_weights) for label, prototype in prototypes.items()}

    # Compute DTW distances
    distances = {label: calculate_dtw_distance(norm_path, prototype) for label, prototype in norm_prototypes.items()}

    # Find the best match
    best_match = min(distances, key=distances.get)

    print(distances)
    return best_match



import matplotlib.pyplot as plt
def plot_prototypes():
    # Define the prototypes
    prototypes = {
        'Commercial Flight': commercial_prototype(),
        'Training Flight': training_prototype(),
        'Surveillance Flight': surveillance_prototype(),
        'Emergency Flight': emergency_prototype(),
    }

    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot each prototype path
    for label, path in prototypes.items():
        lats, lons = zip(*[(lat, lon) for lat, lon, _, _ in path])  # Unpack only lat, lon
        plt.plot(lons, lats, label=label, marker='o', markersize=5)
    
        # Customize the plot
        plt.title('Flight Prototypes')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.show()


def test_prototypes():
    prototypes = {
        'Commercial Flight': commercial_prototype(),
        'Training Flight': training_prototype(),
        'Surveillance Flight': surveillance_prototype(),
        'Cargo Flight': cargo_prototype(),
        'Emergency Flight': emergency_prototype(),
        'Private Flight': private_prototype(),
        'Aerobatic Flight': aerobatic_prototype(),
    }

    smoothed_prototypes = {label: smooth_path(prototype) for label, prototype in prototypes.items()}
    
    # Calculate DTW distance between input path and each prototype
    distances = {label: calculate_dtw_distance(prototype, prototype) for label, prototype in smoothed_prototypes.items()}
    
    # Find the minimum distance
    min_distance = min(distances.values())

    # Find all labels that match the minimum distance
    best_matches = [label for label, dist in distances.items() if dist == min_distance]

    print("Best matches:", best_matches)

# Call the function to plot the prototypes
# test_prototypes()
plot_prototypes()


