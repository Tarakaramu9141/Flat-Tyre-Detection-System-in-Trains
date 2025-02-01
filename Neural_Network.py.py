import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class FlatWheelDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_data(self, data):
        """
        Advanced data preparation with multiple feature engineering techniques
        """
        # Create additional features
        data['Impact_Difference'] = np.abs(data['Wheel_Impact_Left(KN)'] - data['Wheel_Impact_Right(KN)'])
        data['Total_Impact'] = data['Wheel_Impact_Left(KN)'] + data['Wheel_Impact_Right(KN)']
        
        # Define advanced flat wheel criteria
        def advanced_flat_wheel_criteria(row):
            conditions = [
                # Impact force criteria
                (row['Wheel_Impact_Left(KN)'] >= 250) & (row['Wheel_Impact_Left(KN)'] <= 310),
                (row['Wheel_Impact_Right(KN)'] >= 250) & (row['Wheel_Impact_Right(KN)'] <= 310),
                # Impact difference criteria
                row['Impact_Difference'] > 50,  # Significant impact variation
                row['Total_Impact'] > 480  # High total impact
            ]
            return int(any(conditions))
        
        # Apply advanced flat wheel detection
        data['Potential_Flat_Wheel'] = data.apply(advanced_flat_wheel_criteria, axis=1)
        
        # Select features
        features = [
            'Wheel_Impact_Left(KN)', 
            'Wheel_Impact_Right(KN)', 
            'Axle_Number',
            'Impact_Difference',
            'Total_Impact'
        ]
        
        # Prepare X and y
        X = data[features].values
        y = data['Potential_Flat_Wheel'].values
        
        return X, y, data
    
    def create_model(self, input_shape):
        """
        Create a more robust neural network model
        """
        model = Sequential([
            # Input layer with increased complexity
            Dense(64, activation='relu', input_shape=input_shape, 
                  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Hidden layers with increased depth and regularization
            Dense(32, activation='relu', 
                  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(16, activation='relu', 
                  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Advanced optimization
        optimizer = Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(), 
                     tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train_with_cross_validation(self, X, y, n_splits=5):
        """
        Perform stratified cross-validation
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        # Callbacks for improved training
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=0.00001
        )
        
        for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}")
            
            # Split data
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Compute class weights
            class_weights = class_weight.compute_class_weight(
                'balanced', 
                classes=np.unique(y_train), 
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            # Create and train model
            model = self.create_model((X_train_scaled.shape[1],))
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=100,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate model
            y_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
            print("\nClassification Report:")
            print(classification_report(y_val, y_pred))
            
            # Store best model
            if fold == n_splits:
                self.model = model
            
            cv_scores.append(history)
        
        return cv_scores
    
    def detect_flat_wheels(self, data):
        """
        Advanced flat wheel detection
        """
        # Prepare features
        X, y, processed_data = self.prepare_data(data)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        predictions = self.model.predict(X_scaled).flatten()
        
        # Add predictions to dataframe
        processed_data['Flat_Wheel_Probability'] = predictions
        
        # Advanced filtering
        flat_wheels = processed_data[
            (processed_data['Potential_Flat_Wheel'] == 1) | 
            (processed_data['Flat_Wheel_Probability'] > 0.5)
        ]
        
        return flat_wheels
    
    def visualize_results(self, data):
        """
        Comprehensive visualization of results
        """
        # Prepare data
        X, y, processed_data = self.prepare_data(data)
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        y_pred_proba = self.model.predict(X_scaled).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        # Feature Importance Visualization
        feature_names = [
            'Wheel_Impact_Left(KN)', 
            'Wheel_Impact_Right(KN)', 
            'Axle_Number',
            'Impact_Difference',
            'Total_Impact'
        ]
        
        plt.figure(figsize=(10, 4))
        plt.title('Feature Distributions for Flat Wheels')
        for i, feature in enumerate(feature_names):
            plt.subplot(2, 3, i+1)
            sns.histplot(data=processed_data, x=feature, hue='Potential_Flat_Wheel', 
                         multiple='stack', palette='Set2')
            plt.title(feature)
        plt.tight_layout()
        plt.show()
def predict_flat_wheel_from_csv(file_path, detector):
    """
    Predict which axle numbers have flat wheels based on a CSV file.
    :param file_path: Path to the CSV file.
    :param detector: Instance of FlatWheelDetector with a trained model.
    """
    try:
        # Load the data from the CSV file
        new_data = pd.read_csv(file_path)

        # Check if required columns exist in the input file
        required_columns = ['Axle_Number', 'Wheel_Impact_Right(KN)', 'Wheel_Impact_Left(KN)']
        if not all(col in new_data.columns for col in required_columns):
            raise ValueError(f"The input file must contain the following columns: {', '.join(required_columns)}")

        # Prepare features for prediction
        X_new, _, processed_new_data = detector.prepare_data(new_data)
        X_scaled = detector.scaler.transform(X_new)

        # Make predictions using the trained model
        predictions = detector.model.predict(X_scaled).flatten()
        processed_new_data['Flat_Wheel_Probability'] = predictions

        # Identify axle numbers with a high probability of flat wheels
        flat_wheel_candidates = processed_new_data[
            processed_new_data['Flat_Wheel_Probability'] > 0.5
        ]
        
        if not flat_wheel_candidates.empty:
            print("\nFlat wheels detected for the following axle numbers:")
            print(flat_wheel_candidates[['Axle_Number', 'Wheel_Impact_Right(KN)', 'Wheel_Impact_Left(KN)', 'Flat_Wheel_Probability']])
        else:
            print("\nNo flat wheels detected in the provided data.")
        
        return flat_wheel_candidates
    except Exception as e:
        print(f"An error occurred during prediction: {e}")


def main():
    # Load data
    csv_path = 'D:/cris-project/Balanced_Wheel_Impact_Data.csv'
    new_csv_path = 'D:/cris-project/sample.csv'  # Path to the new CSV file for prediction
    data = pd.read_csv(csv_path)
    
    # Create detector
    detector = FlatWheelDetector()
    
    try:
        # Prepare data
        X, y, _ = detector.prepare_data(data)
        
        # Train with cross-validation
        detector.train_with_cross_validation(X, y)
        
        # Detect flat wheels
        flat_wheels = detector.detect_flat_wheels(data)
        
        print("\nPotential Flat Wheel Candidates:")
        print(flat_wheels[['Axle_Number', 'Wheel_Impact_Left(KN)', 'Wheel_Impact_Right(KN)', 'Flat_Wheel_Probability']])
        
        # Predict from a new CSV file
        print("\n--- Predicting Flat Wheels from New CSV File ---")
        predict_flat_wheel_from_csv(new_csv_path, detector)
        
        # Visualize results
        detector.visualize_results(data)
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()


