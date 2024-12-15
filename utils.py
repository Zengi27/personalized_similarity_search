import numpy as np
import random


def mahalanobis_distance(x, y, metric_matrix):
    """
    Calculate the Mahalanobis distance between two vectors.

    Parameters:
    x (list or ndarray): The first vector.
    y (list or ndarray): The second vector.
    metric_matrix (ndarray): The metric matrix, typically the inverse of the covariance matrix.

    Returns:
    float: The Mahalanobis distance between the vectors x and y.
    """
    
    # Convert lists to NumPy arrays
    # x = np.array(x)
    # y = np.array(y)
    
    # Compute the difference between the two vectors
    diff = x - y
    # Compute the Mahalanobis distance
    distance = np.dot(np.dot(diff, metric_matrix), diff)
    # Compute the square root of the distance
    distance = np.sqrt(distance)
    
    return distance


def euclidean_distance(x, y):
    """
    Calculate the Euclidean distance between two vectors.

    Parameters:
    x (list or ndarray): The first vector.
    y (list or ndarray): The second vector.

    Returns:
    float: The Euclidean distance between the vectors x and y.
    """
    

    # x = np.array(x)
    # y = np.array(y)
    
    distance = np.sqrt(np.sum((x - y)**2))
    
    return distance


def cosine_distance_norm(x, y):
    """
    Calculate the cosine distance between two normalized vectors.

    Parameters:
    x (list or ndarray): The first normalized vector.
    y (list or ndarray): The second normalized vector.

    Returns:
    float: The cosine distance between the vectors x and y.
    """
    
    # x = np.array(x)
    # y = np.array(y)
    
    # Calculate cosine similarity as dot product of the vectors
    cosine_similarity = np.dot(x, y)
    
    # Cosine distance is 1 minus the cosine similarity
    distance = 1 - cosine_similarity
    return distance


# Vectorized Euclidean distance function
def vectorized_euclidean_distances(target_vector, feature_vectors):
    delta = feature_vectors - target_vector
    distances = np.sqrt(np.sum(delta ** 2, axis=1))

    return distances


# Vectorized Mahalanobis distance function
def vectorized_mahalanobis_distances(target_vector, feature_vectors, inv_cov_matrix):
    delta = feature_vectors - target_vector
    distances = np.sqrt(np.sum(np.dot(delta, inv_cov_matrix) * delta, axis=1))
    return distances


# Vectorized cosine distance function for normalized vectors, using np.sum
def vectorized_cosine_distances_normalized(target_vector, feature_vectors):
    # Calculate cosine similarities using the dot product
    cosine_similarities = np.dot(feature_vectors, target_vector)

    # cosine_similarities = np.sum(feature_vectors * target_vector, axis=1)
    cosine_distances = 1 - cosine_similarities

    return cosine_distances


def calculate_scaling_factor(A):
    """
    Calculate the scaling factor based on the minimum eigenvalue of a matrix.

    Parameters:
    A (ndarray): The input square matrix.

    Returns:
    float: The scaling factor, calculated as the reciprocal of the square root of the minimum eigenvalue of A.
    """
    
    # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Find the minimum eigenvalue
    min_eigenvalue = np.min(eigenvalues)

    # Calculate the scaling factor
    scaling_factor = 1 / np.sqrt(min_eigenvalue)

    return scaling_factor


def generate_random_pairs(dataset, num_pairs):
    """
    Generate random pairs from a dataset.

    Parameters:
    dataset (list or ndarray): The dataset from which pairs are generated.
    num_pairs (int): The number of pairs to generate.

    Returns:
    tuple: A tuple containing two elements:
        - random_pairs (list of lists): Pairs of elements from the dataset.
        - y (list): Labels where 1 indicates a positive pair and -1 indicates a negative pair.
    
    """
    
    # Choose one random index for the first element of the pair
    random_pairs = []
    y = []
    for _ in range(num_pairs):
        # Choose one random index for the first element of the pair
        idx1 = random.choice(range(len(dataset)))
        # Choose two other distinct random indices for the second elements of the pairs
        idx2, idx3 = random.sample([i for i in range(len(dataset)) if i != idx1], 2)
        
        # Create two pairs with the same first index and different second indices
        pair1 = [dataset[idx1], dataset[idx2]]
        pair2 = [dataset[idx1], dataset[idx3]]
        
        random_pairs.append(pair1)
        random_pairs.append(pair2)
        
        y.append(1)
        y.append(-1)
        
    return random_pairs, y


def generate_random_pairs_2(dataset, num_pairs, num_closest):
    random_pairs = []
    y = []
    
    for _ in range(num_pairs):
        # Choose one random index for the first element of the pair
        idx1 = random.choice(range(len(dataset)))
        
        # Calculate distances from dataset[idx1] to all other vectors
        distances = [euclidean_distance(dataset[idx1], vector) for vector in dataset]

        # Get the indices of the closest num_closest vectors (excluding idx1)
        closest_indices = np.argsort(distances)[:num_closest + 1]  # +1 to include idx1 itself
        closest_indices = [idx for idx in closest_indices if idx != idx1]  # Exclude idx1
        
        # Ensure there are enough closest indices
        if len(closest_indices) < 1:
            raise ValueError("Not enough distinct vectors in the dataset to form pairs.")
        
        # Choose one index for the second element of the pair from the closest vectors
        idx2 = random.choice(closest_indices)
        
        # Choose idx3 randomly from the entire dataset, ensuring it's not idx1
        idx3 = random.choice([i for i in range(len(dataset)) if i != idx1])
        
        # Create two pairs with the same first index and different second indices
        pair1 = [dataset[idx1], dataset[idx2]]
        pair2 = [dataset[idx1], dataset[idx3]]
        
        random_pairs.append(pair1)
        random_pairs.append(pair2)
        
        y.append(1)   # Positive pair
        y.append(-1)  # Negative pair
        
    return random_pairs, y


def nearest_indices_mahalanobis(anchor_image, dataset, start_matrix, num_indices=50):
    """
    Find the nearest indices to an anchor image using Mahalanobis distance.

    Parameters:
    anchor_image (list or ndarray): The anchor image vector.
    dataset (list of ndarray): The dataset of vectors.
    start_matrix (ndarray): The metric matrix used to calculate Mahalanobis distance.
    num_indices (int): The number of nearest indices to return.

    Returns:
    ndarray: Indices of the nearest vectors in the dataset.
    """
    distances = np.array([mahalanobis_distance(anchor_image, vector, start_matrix) for vector in dataset])
    nearest_indices = np.argsort(distances)[:num_indices]
    
    # Get the distances for these nearest indices
    nearest_distances = distances[nearest_indices]
    
    return nearest_indices, nearest_distances


def nearest_indices_euclidean(anchor_image, dataset, num_indices=50):
    """
    Find the nearest indices to an anchor image using Euclidean distance.

    Parameters:
    anchor_image (list or ndarray): The anchor image vector.
    dataset (list of ndarray): The dataset of vectors.
    num_indices (int): The number of nearest indices to return.

    Returns:
    ndarray: Indices of the nearest vectors in the dataset.
    """
    
    distances = np.array([euclidean_distance(anchor_image, vector) for vector in dataset])
    nearest_indices = np.argsort(distances)[:num_indices]
    
    # Get the distances for these nearest indices
    nearest_distances = distances[nearest_indices]
    
    
    return nearest_indices, nearest_distances


def percentage_common_elements(array1, array2):
    """
    Calculate the percentage of common elements between two arrays.

    Parameters:
    array1 (list or ndarray): The first array.
    array2 (list or ndarray): The second array.

    Returns:
    float: The percentage of elements in array1 that are also present in array2.
    """
    
    # Convert arrays to sets to remove duplicates and for faster intersection
    set1 = set(array1)
    set2 = set(array2)
    # Find the intersection of both sets
    common_elements = set1.intersection(set2)
    # Calculate the percentage of common elements relative to the size of the first set
    if len(set1) == 0:  # Avoid division by zero if the first array is empty
        return 0.0
    
    percentage = (len(common_elements) / len(set1)) * 100
    print("Percentage of elements included", percentage, "%")
    
    return percentage


def get_nearest_distances(distances, num_nearest):
    """
    Get the indices and distances of the nearest vectors in the dataset.
    
    Parameters:
    - distances: A list of Mahalanobis distances between the anchor image and the dataset.
    - num_nearest: The number of nearest distances to return.
    
    Returns:
    - nearest_distances: A list of the nearest distances.
    - nearest_indices: A list of indices in the dataset corresponding to the nearest distances.
    """
    # Sort the distances and get the indices of the nearest num_nearest ones
    nearest_indices = np.argsort(distances)[:num_nearest]
    
    # Get the nearest distances using the indices
    nearest_distances = [distances[i] for i in nearest_indices]
    
    return nearest_distances, nearest_indices