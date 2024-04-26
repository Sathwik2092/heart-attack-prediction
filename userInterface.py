import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the heart dataset
data = pd.read_csv('/content/heart.csv')

# Split features and target
x = data.iloc[:, :-1]
y = data.iloc[:, -1:]

# Standardize features
stsc = StandardScaler()
x = stsc.fit_transform(x)

# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Training logistic regression model
lr = LogisticRegression(random_state=88)
lr.fit(x_train, y_train.values.ravel())

# Model evaluation
train_score = lr.score(x_train, y_train)
test_score = lr.score(x_test, y_test)
print("Train accuracy:", train_score)
print("Test accuracy:", test_score)

# Prompting user for input
print("\nEnter the following details to predict heart disease risk:")
age = float(input("Age: "))
sex = float(input("Sex (0 for female, 1 for male): "))
cp = float(input("Chest Pain Type: "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Serum Cholesterol (mg/dl): "))
fbs = float(input("Fasting Blood Sugar (> 120 mg/dl, 1 = true; 0 = false): "))
restecg = float(input("Resting Electrocardiographic Results: "))
thalach = float(input("Maximum Heart Rate Achieved: "))
exang = float(input("Exercise Induced Angina (1 = yes; 0 = no): "))
oldpeak = float(input("ST Depression Induced by Exercise Relative to Rest: "))
slope = float(input("Slope of the Peak Exercise ST Segment: "))
ca = float(input("Number of Major Vessels Colored by Fluoroscopy: "))
thal = float(input("Thalassemia: "))

# Create a DataFrame from user input
user_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Standardize user input
user_data_scaled = stsc.transform(user_data)

# Predicting outcome
prediction = lr.predict(user_data_scaled)
probability = lr.predict_proba(user_data_scaled)

# Output prediction
if prediction[0] == 0:
    print("\nPrediction: No Heart Disease")
else:
    print("\nPrediction: Heart Disease Detected")
print("Probability of Heart Disease:", probability[0][1])
