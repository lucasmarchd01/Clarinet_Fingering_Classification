import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Step 1: Load the data
data = pd.read_csv("combined_data_filtered.csv")

# Step 2: Extract features and labels
X = data.pivot(index="Frame", columns=["Hand", "Landmark"], values=["X", "Y", "Z"])
X = X.to_numpy().reshape(len(X), -1)  # Flatten the data
y = data["Fingering"].values

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Support Vector Machine (SVM) classifier
svm_clf = SVC(kernel="linear")
svm_clf.fit(X_train, y_train)

# Step 5: Train KMeans clustering
kmeans = KMeans(n_clusters=len(np.unique(y_train)))
kmeans.fit(X_train)

# Step 6: Predictions
svm_preds = svm_clf.predict(X_test)
kmeans_preds = kmeans.predict(X_test)

# Step 7: Evaluation (optional)
svm_accuracy = accuracy_score(y_test, svm_preds)
kmeans_accuracy = accuracy_score(y_test, kmeans_preds)

print("SVM Accuracy:", svm_accuracy)
print("KMeans Accuracy:", kmeans_accuracy)
