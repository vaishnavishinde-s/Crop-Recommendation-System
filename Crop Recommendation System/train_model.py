import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('Crop_recommendation.csv')

# Features and target
X = data.drop('label', axis=1)
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open('crop_model.pkl', 'wb'))

print("Model saved as crop_model.pkl")