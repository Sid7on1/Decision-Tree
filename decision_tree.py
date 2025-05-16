import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset (assumes 'heart.csv' is in the same folder)
df = pd.read_csv('heart.csv')

# Data overview
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Separate features and target
X = df.drop('target', axis=1)  # target is the label column
y = df['target']

# Feature scaling (important for some features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=10)

# Train the model
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler for future use
joblib.dump(clf, 'decision_tree_heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")