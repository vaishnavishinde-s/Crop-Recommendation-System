import pickle

model = pickle.load(open('crop_model.pkl', 'rb'))

# Example input
input_data = [[90, 40, 40, 25, 80, 6.5, 200]]

prediction = model.predict(input_data)

print("Recommended Crop:", prediction[0])