import numpy as np
from collections import Counter

def predict_knn(new_point, data, k):
    """
    new_point: list/array of feature values (e.g., [5.1, 3.5])
    data: list of tuples -> (features, class_label)
          example: [([5.1, 3.5], 'A'), ([6.2, 2.8], 'B')]
    k: number of nearest neighbors
    """

    # 1. Normalize data + new_point
    features = np.array([row[0] for row in data], dtype=float)
    labels = [row[1] for row in data]

    # Compute min and max for each column
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)

    # Avoid division by zero
    ranges = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)

    # Normalize dataset
    norm_features = (features - min_vals) / ranges

    # Normalize new point
    new_point = np.array(new_point, dtype=float)
    norm_new_point = (new_point - min_vals) / ranges

    # 2. Compute Euclidean distances
    distances = np.linalg.norm(norm_features - norm_new_point, axis=1)

    # 3. Select k nearest neighbors
    k_indices = distances.argsort()[:k]
    k_labels = [labels[i] for i in k_indices]

    # 4. Majority vote prediction
    prediction = Counter(k_labels).most_common(1)[0][0]

    return prediction

#Sample data
data = [
    ([5.1, 3.5], "A"),
    ([4.9, 3.0], "A"),
    ([6.2, 2.9], "B"),
    ([7.0, 3.2], "B")
]

new_point = [5.8, 3.0]

print(predict_knn(new_point, data, k=3))
