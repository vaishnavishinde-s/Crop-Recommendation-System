# 🌱 Crop Recommendation System

## Overview

Choosing the right crop is one of the most important decisions in farming. This project is a **Crop Recommendation System** that uses Machine Learning to suggest the most suitable crop based on soil nutrients and environmental conditions.

The user enters values such as **Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall**, and the system predicts the crop that is most likely to grow well under those conditions.

This project was built to explore how Machine Learning can be applied to solve real-world agricultural problems and support smarter farming decisions.

---

## Features

* 🌾 Recommends the best crop based on input conditions.
* 🤖 Uses a trained Machine Learning model for prediction.
* 💻 Simple and easy-to-use web interface.
* ⚡ Instant prediction results.
* 📱 Responsive design that works on different screen sizes.

---

## Technologies Used

* **Python**
* **Flask**
* **HTML**
* **CSS**
* **Bootstrap**
* **Pandas**
* **NumPy**
* **Scikit-learn**

---

## Project Structure

```text
Crop-Recommendation-System/
│── static/
│   ├── css/
│   ├── images/
│
│── templates/
│   ├── index.html
│   └── result.html
│
│── model/
│   ├── crop_model.pkl
│   └── label_encoder.pkl
│
│── app.py
│── requirements.txt
│── README.md
└── dataset.csv
```

---

## How It Works

1. Open the application in your browser.
2. Enter the required soil and weather values.
3. Click the **Predict** button.
4. The model processes the data.
5. The recommended crop is displayed on the screen.

---

## Input Parameters

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature (°C)
* Humidity (%)
* Soil pH
* Rainfall (mm)

---

## Output

The application predicts the most suitable crop for the given conditions.

**Example**

```
Input:
Nitrogen: 90
Phosphorus: 42
Potassium: 43
Temperature: 21°C
Humidity: 82%
pH: 6.5
Rainfall: 203 mm

Output:
Recommended Crop: Rice 🌾
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Crop-Recommendation-System.git
```

Move into the project folder:

```bash
cd Crop-Recommendation-System
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Activate it:

**Linux/macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Future Improvements

* Add fertilizer recommendations.
* Integrate real-time weather data.
* Support multiple languages.
* Store prediction history.
* Deploy the application online.
* Build a mobile version.

---

## Learning Outcomes

While building this project, I learned:

* How to preprocess agricultural datasets.
* How to train and evaluate Machine Learning models.
* How to integrate a trained model with Flask.
* How to build a simple web application for ML projects.
* How to connect the frontend with backend logic.

---

## Contributing

Contributions are welcome. Feel free to fork the repository, create a new branch, and submit a pull request if you have ideas for improvements.

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Vaishnavi Shinde**

Computer Engineering Student | Python Developer | Machine Learning Enthusiast

If you found this project useful, consider giving it a ⭐ on GitHub!
