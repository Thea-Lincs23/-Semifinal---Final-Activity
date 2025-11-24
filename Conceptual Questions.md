--- 1. Classification Basics ---

a. Difference between classification and regression
•Classification predicts a category/class label (e.g., spam/not spam, disease/no disease).
•Regression predicts a continuous numeric value (e.g., price, temperature, sales amount).

b. Give two examples each of:
• Binary classification
Email spam detection (Spam / Not Spam)
Medical diagnosis (Positive / Negative)
• Multiclass classification
Classifying fruits (Apple / Banana / Mango)
Predicting type of vehicle (Car / Truck / Motorcycle / Bus)

c. Define the following evaluation metrics:

• Accuracy-
The proportion of correct predictions out of all predictions

• Precision-
Of the items the model predicted as positive, how many were actually positive.

• Recall-
 Out of all the actual positive items, how many the model was able to find.

• F1 Score-
The harmonic mean of precision and recall; balances the two.

• Confusion Matrix-
A table showing TP, TN, FP, and FN to evaluate classification performance.

--- 2. Logistic Regression ---

a. Why is logistic regression considered a classification algorithm, not a regression algorithm?
- Because it outputs probabilities of classes using the sigmoid function and classifies data into discrete categories, not continuous values.
b. What is the role of the sigmoid function in logistic regression?
-Converts any number into a probability between 0 and 1, which is used to decide the class.
c. List two advantages and two disadvantages of logistic regression.
• Advantages

Simple to implement and interpret.

Works well when classes are linearly separable.

• Disadvantages

Not suitable for complex, non-linear relationships (unless features are transformed).

Sensitive to outliers and multicollinearity.

3. K-Nearest Neighbors (KNN)
a. What does it mean that KNN is a non-parametric and lazy learning algorithm?
• Non-parametric:
It makes no assumptions about the data distribution. No equation/model is learned.
• Lazy learning:
It does not train a model.
It stores the dataset, and during prediction, it calculates distances only when needed, making prediction slower.

b. Describe the steps of how KNN classifies a new data point.
1. Calculate the distance (e.g., Euclidean) between the new point and all existing data points.
2. Select the K nearest neighbors (smallest distances).
3. Check the majority class among those K neighbors.
4. Assign the majority class as the prediction.

c. Explain how choosing a small K vs. a large K affects the model.
• Small K (e.g., K = 1 or 3):
Very sensitive to noise.
Higher chance of overfitting.
More flexible but less stable.

• Large K (e.g., K = 15 or 25):
More stable and smooth decision boundaries.
Can underfit (too generalized).
Less sensitive to outliers.

d. Why is feature scaling (normalization/standardization) important for KNN?
- Feature scaling prevents large-valued features from overpowering small-valued features, ensuring accurate distance-based classification.
