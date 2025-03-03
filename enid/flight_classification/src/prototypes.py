
from dtaidistance import dtw_ndim
from scipy.interpolate import interp1d
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# # # Commercial Flight: Straight path with minimal fluctuations
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



def dtw_distance(path1, path2, window =  5, slope=1):
    len_path1 = len(path1)
    len_path2 = len(path2)

    # If lengths differ significantly, increase the window size for more flexibility
    if abs(len_path1 - len_path2) > 5:
        window = max(window, int(abs(len_path1 - len_path2) / 2))

    return dtw_ndim.distance(path1, path2, window=window)
    





# Function to perform DTW matching with parameter tuning
def match_flight_to_prototypes(flight_data, speed_weight=2.0, altitude_weight=2.0):
    prototypes = {
        "Commercial": commercial_prototype(),
        "Training": training_prototype(),
        "Surveillance": surveillance_prototype(),
        "Private": private_prototype(),
        # Add other prototypes here
    }

    # Define weights (higher weight on speed)
    feature_weights = np.array([1.0, 1.0, altitude_weight, speed_weight])  

    # Normalize and apply weights
    norm_path = normalize_path(flight_data, weights=feature_weights)
    norm_prototypes = {label: normalize_path(prototype, weights=feature_weights) for label, prototype in prototypes.items()}

    # Scaling flight_data (optional, for tuning)
    scaler = StandardScaler()
    scaled_flight_data = scaler.fit_transform(norm_path)

    results = {}
    silhouette_scores = {}

    # Tuning parameters for DTW
    window_sizes = [20, 30]  # Example window sizes for tuning

    # Perform DTW matching for each prototype with parameter tuning
    for label, prototype in norm_prototypes.items():
        scaled_prototype = scaler.transform(prototype)

        # Tuning DTW parameters
        tuned_results = {}
        for window in window_sizes:                     
                distance = dtw_distance(scaled_flight_data, scaled_prototype, window=window)
                print("Distance:", distance)
                tuned_results[window] = distance

                # Compute silhouette score for clustering (optional)
                combined_data = np.vstack([scaled_flight_data, scaled_prototype])
                silhouette = silhouette_score(combined_data, [0] * len(scaled_flight_data) + [1] * len(scaled_prototype))
                silhouette_scores[window] = silhouette

        # Get best tuned match based on minimum DTW distance
        best_tuned_params = min(tuned_results, key=tuned_results.get)
        results[label] = tuned_results[best_tuned_params]

        # Calculate confidence score (1 / (1 + DTW distance))
        confidence_score = 1 / (1 + results[label])

    # Get the overall best match based on minimum DTW distance
    best_match = min(results, key=results.get)
    
    return best_match, results, confidence_score, silhouette_scores


