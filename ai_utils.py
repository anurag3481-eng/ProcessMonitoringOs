import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from datetime import datetime, timedelta

class ResourcePredictor:
    """Predictive analytics for system resource usage"""
    
    def __init__(self, history_size=60):
        self.history_size = history_size
        self.cpu_model = None
        self.mem_model = None
        self.disk_model = None
        self.min_samples_for_prediction = 30
    
    def can_predict(self, data):
        """Check if we have enough data to make predictions"""
        return len(data) >= self.min_samples_for_prediction
    
    def predict_next_values(self, data, steps=5):
        """Predict the next values using ARIMA model"""
        if not self.can_predict(data):
            return None
            
        try:
            # Use a simple ARIMA model for prediction
            model = ARIMA(data, order=(1, 0, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return forecast
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def get_predictions(self, cpu_history, mem_history, disk_history):
        """Get predictions for CPU, memory and disk usage"""
        predictions = {
            'cpu': None,
            'memory': None,
            'disk': None,
            'prediction_time': None
        }
        
        # Only make predictions if we have enough data
        if self.can_predict(cpu_history):
            cpu_pred = self.predict_next_values(cpu_history)
            mem_pred = self.predict_next_values(mem_history)
            disk_pred = self.predict_next_values(disk_history)
            
            if cpu_pred is not None and mem_pred is not None and disk_pred is not None:
                # Generate time labels for predictions (5 minutes into future)
                now = datetime.now()
                future_times = [(now + timedelta(minutes=i)).strftime("%H:%M") 
                               for i in range(1, 6)]
                
                predictions = {
                    'cpu': cpu_pred.tolist(),
                    'memory': mem_pred.tolist(),
                    'disk': disk_pred.tolist(),
                    'times': future_times,
                    'prediction_time': now.strftime("%H:%M:%S")
                }
        
        return predictions


class AnomalyDetector:
    """Anomaly detection for system resource usage"""
    
    def __init__(self):
        self.model = None
        self.min_samples_for_training = 50
        self.is_trained = False
        self.last_training_time = None
        self.training_interval = 10  # Train every 10 updates
        self.update_count = 0
    
    def should_train(self, data_size):
        """Determine if the model should be trained"""
        if not self.is_trained and data_size >= self.min_samples_for_training:
            return True
        
        if self.is_trained and self.update_count >= self.training_interval:
            self.update_count = 0
            return True
            
        self.update_count += 1
        return False
    
    def train(self, cpu_data, mem_data, disk_data):
        """Train the anomaly detection model"""
        if len(cpu_data) < self.min_samples_for_training:
            return False
            
        try:
            # Combine the data into a single feature matrix
            X = np.column_stack((cpu_data, mem_data, disk_data))
            
            # Train an Isolation Forest model
            self.model = IsolationForest(contamination=0.05, random_state=42)
            self.model.fit(X)
            
            self.is_trained = True
            self.last_training_time = datetime.now().strftime("%H:%M:%S")
            return True
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def detect_anomalies(self, cpu_data, mem_data, disk_data):
        """Detect anomalies in the current resource usage"""
        if not self.is_trained:
            return None
            
        try:
            # Get the most recent data point
            cpu_current = cpu_data[-1]
            mem_current = mem_data[-1]
            disk_current = disk_data[-1]
            
            # Make prediction
            X = np.array([[cpu_current, mem_current, disk_current]])
            prediction = self.model.predict(X)
            
            # Calculate anomaly score (negative = anomaly)
            score = self.model.score_samples(X)[0]
            
            # Determine if this is an anomaly
            is_anomaly = prediction[0] == -1
            
            return {
                'is_anomaly': is_anomaly,
                'score': score,
                'cpu': cpu_current,
                'memory': mem_current,
                'disk': disk_current,
                'detection_time': datetime.now().strftime("%H:%M:%S")
            }
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return None 
