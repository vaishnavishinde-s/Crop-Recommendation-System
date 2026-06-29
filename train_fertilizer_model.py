# =====================================
# 🧪 Fertilizer Recommendation Model
# =====================================

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==============================
# 1. Load Dataset
# ==============================

data = pd.read_csv('fertilizer_recommendation_dataset.csv')

print("Dataset Loaded ✅")
print(data.head())

# ==============================
# 2. Drop unnecessary column
# ==============================

data = data.drop(columns=['Remark'])

# ==============================
# 3. Encode Categorical Data
# ==============================

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

data['Soil'] = le_soil.fit_transform(data['Soil'])
data['Crop'] = le_crop.fit_transform(data['Crop'])
data['Fertilizer'] = le_fert.fit_transform(data['Fertilizer'])

# ==============================
# 4. Features & Labels
# ==============================

X = data.drop('Fertilizer', axis=1)
y = data['Fertilizer']

# ==============================
# 5. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. Train Model
# ==============================

model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)

print("Model Trained ✅")

# ==============================
# 7. Evaluate
# ==============================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# ==============================
# 8. Save Model + Encoders
# ==============================

pickle.dump(model, open('models/fertilizer_model.pkl', 'wb'))
pickle.dump(le_soil, open('models/soil_encoder.pkl', 'wb'))
pickle.dump(le_crop, open('models/crop_encoder.pkl', 'wb'))
pickle.dump(le_fert, open('models/fertilizer_encoder.pkl', 'wb'))

print("Model & Encoders Saved ✅")