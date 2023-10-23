import numpy as np

# Assuming data_list contains your list of 100 data points with 12 features each
data_list = [
    [feature1_value, feature2_value, ..., feature12_value],  # Data point 1
    [feature1_value, feature2_value, ..., feature12_value],  # Data point 2
    ...
    [feature1_value, feature2_value, ..., feature12_value]   # Data point 100
]

# Convert the list to a NumPy array
data_array = np.array(data_list)

# Initialize empty arrays for lower and upper bounds for each feature
lower_bounds = np.zeros(12)
upper_bounds = np.zeros(12)

# Calculate lower and upper bounds for each feature based on IQR
for i in range(12):  # Loop through each feature
    feature_data = data_array[:, i]  # Extract the data for the current feature
    Q1 = np.percentile(feature_data, 25)  # Calculate the 25th percentile
    Q3 = np.percentile(feature_data, 75)  # Calculate the 75th percentile
    IQR = Q3 - Q1  # Calculate the interquartile range

    # Define lower and upper bounds for the current feature
    lower_bounds[i] = Q1 - 1.5 * IQR
    upper_bounds[i] = Q3 + 1.5 * IQR

# Identify outliers for each feature
outliers = ((data_array < lower_bounds) | (data_array > upper_bounds))

# Print indices of outliers for each feature
for feature_index, feature_outliers in enumerate(outliers.T):
    print(f'Outliers for Feature {feature_index + 1}:')
    print(np.nonzero(feature_outliers)[0])