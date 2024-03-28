import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns


# Assuming 'data' contains the input data
df = pd.read_csv("data_final.csv")


# Extract features and labels
X = df.iloc[:, 1:-1].values  # Exclude the Unique_Frame and Fingering columns
y = df.iloc[:, -1].values  # Fingering column

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

# Train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Train SVM classifier
svm_classifier = SVC(kernel="linear")
svm_classifier.fit(X_train, y_train)

# Evaluate classifiers
knn_accuracy = knn_classifier.score(X_test, y_test)
svm_accuracy = svm_classifier.score(X_test, y_test)

knn_predictions = knn_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)

print("KNN Classifier Accuracy:", knn_accuracy)
print("SVM Classifier Accuracy:", svm_accuracy)
# Classification report
print("KNN Classifier Metrics:")
print(classification_report(y_test, knn_predictions))

print("SVM Classifier Metrics:")
print(classification_report(y_test, svm_predictions))

# Confusion matrix
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))

print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))

# Calculate confusion matrices
knn_conf_matrix = confusion_matrix(y_test, knn_predictions)
svm_conf_matrix = confusion_matrix(y_test, svm_predictions)

# Plot confusion matrices
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(knn_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.subplot(1, 2, 2)
sns.heatmap(svm_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()


dump(knn_classifier, "knn_classifier.joblib")
dump(svm_classifier, "svm_classifier.joblib")
