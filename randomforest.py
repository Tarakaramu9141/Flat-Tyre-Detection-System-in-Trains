import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('Balanced_Wheel_Impact_Data.csv')

# Assuming the columns are named 'Axle_Number', 'Wheel_Impact_Right(KN)', 'Wheel_Impact_Left(KN)', and 'Flat_Wheel'
axle_data = df[['Axle_Number', 'Wheel_Impact_Right(KN)', 'Wheel_Impact_Left(KN)']]
labels = df['Flat_Wheel'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(axle_data, labels, test_size=0.3, random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required at each leaf node
}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_rf = grid_search.best_estimator_

# Evaluate the model
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print classification report
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy of the best model: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Flat Wheel', 'Flat Wheel']))

# Cross-Validation Score
cv_scores = cross_val_score(best_rf, axle_data, labels, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")

# Function to predict if a wheel is flat using the trained model
def predict_flat_wheel_from_csv(file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    
    # Check if required columns exist in the CSV
    required_columns = ['Axle_Number', 'Wheel_Impact_Right(KN)', 'Wheel_Impact_Left(KN)']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"The input file must contain the following columns: {', '.join(required_columns)}")
    
    # Make predictions for the entire dataset
    predictions = best_rf.predict(data[required_columns])
    
    # Add predictions to the DataFrame
    data['Flat_Wheel_Prediction'] = predictions
    
    # Filter rows where the prediction indicates a flat wheel (1 indicates flat)
    flat_wheel_rows = data[data['Flat_Wheel_Prediction'] == 1]
    
    if not flat_wheel_rows.empty:
        print("Flat wheels detected in the following rows:")
        print(flat_wheel_rows[['Axle_Number', 'Wheel_Impact_Right(KN)', 'Wheel_Impact_Left(KN)']])
        
    else:
        print("No flat wheels detected.")
        

# Example usage
csv_file_path = 'sample.csv'  # Replace with your actual CSV file path
predict_flat_wheel_from_csv(csv_file_path)



