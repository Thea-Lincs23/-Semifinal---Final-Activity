import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def logistic_predict(z, threshold=0.5):
    # 1. Compute probability
    prob = sigmoid(z)

    # 2. Classification decision
    classification = 1 if prob >= threshold else 0

    # 3. Print results
    print("Probability:", prob)
    print("Predicted class:", classification)

    return prob, classification

# Example
z = 1.72
logistic_predict(z)
