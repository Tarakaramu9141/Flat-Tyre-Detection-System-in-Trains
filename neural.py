# Neural Network for Predicting Flat Wheels

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data = pd.read_csv('Graph_Data.csv')

# Prepare the data
X = data[['Axle_Number', 'Wheel_Impact']].values
y = data['Wheel_Impact'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.1)

# Predicting flat wheels
predictions = model.predict(X_test)
predicted_flat_wheels = (predictions > 0.5).astype(int)

# Output the results
output = pd.DataFrame({'Axle_Number': X_test[:, 0], 'Predicted_Flat_Wheel': predicted_flat_wheels.flatten()})
output.to_csv('predicted_flat_wheels.csv', index=False)


