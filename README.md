This repository provides a system for monitoring and analyzing system resources (CPU, memory, and disk usage) as well as individual process details. It includes two primary modules:

ResourcePredictor: A predictive model to forecast resource usage.
AnomalyDetector: An anomaly detection system to identify unusual patterns in resource consumption.
Process Utilities: Functions to interact with processes on the system, including retrieving details, killing processes, and changing process priority.
Table of Contents
Installation
Usage
Resource Prediction
Anomaly Detection
Process Monitoring
Classes and Functions
ResourcePredictor
AnomalyDetector
Process Utilities
Dependencies
License
Installation
To use this system, you must have Python installed along with the required dependencies. You can install them using pip:

bash
Copy
pip install numpy pandas scikit-learn statsmodels psutil
Usage
Resource Prediction
The ResourcePredictor class predicts future system resource usage (CPU, memory, disk) based on historical data. It uses a simple ARIMA model to forecast the next 5 values for each resource.

python
Copy
from ai_utils import ResourcePredictor
import numpy as np

# Initialize the predictor
predictor = ResourcePredictor()

# Example historical data (you should provide real data)
cpu_history = np.random.rand(60) * 100  # Example CPU usage history
mem_history = np.random.rand(60) * 100  # Example memory usage history
disk_history = np.random.rand(60) * 100  # Example disk usage history

# Get predictions
predictions = predictor.get_predictions(cpu_history, mem_history, disk_history)
print(predictions)
Anomaly Detection
The AnomalyDetector class detects anomalies in the current resource usage by using an Isolation Forest model. It needs historical resource data to be trained and can then identify abnormal resource consumption patterns.

python
Copy
from ai_utils import AnomalyDetector
import numpy as np

# Initialize the anomaly detector
detector = AnomalyDetector()

# Example historical data (you should provide real data)
cpu_history = np.random.rand(60) * 100  # Example CPU usage history
mem_history = np.random.rand(60) * 100  # Example memory usage history
disk_history = np.random.rand(60) * 100  # Example disk usage history

# Train the anomaly detector (ensure sufficient data is provided)
if detector.should_train(cpu_history):
    detector.train(cpu_history, mem_history, disk_history)

# Detect anomalies in the current usage
anomaly = detector.detect_anomalies(cpu_history, mem_history, disk_history)
print(anomaly)
Process Monitoring
The process_utils module provides functions to interact with processes. It can retrieve process details, terminate processes, or change their priority.

Get Process Details
python
Copy
from process_utils import get_process_details

# Get details of a process by its PID
pid = 1234  # Replace with the PID of the process you're interested in
details = get_process_details(pid)
if details:
    print(details)
else:
    print(f"Process {pid} not found or access denied.")
Kill a Process
python
Copy
from process_utils import kill_process

# Kill a process by its PID
pid = 1234  # Replace with the PID of the process you want to terminate
success, message = kill_process(pid)
print(message)
Change Process Priority
python
Copy
from process_utils import change_process_priority

# Change the priority of a process
pid = 1234  # Replace with the PID of the process
priority = 10  # Set the desired priority (higher numbers mean lower priority)
success, message = change_process_priority(pid, priority)
print(message)
Classes and Functions
ResourcePredictor
__init__(self, history_size=60): Initializes the ResourcePredictor with a specified history size (default is 60).
can_predict(self, data): Checks if there is enough historical data to make a prediction.
predict_next_values(self, data, steps=5): Predicts the next values for a given resource (CPU, memory, disk) using the ARIMA model.
get_predictions(self, cpu_history, mem_history, disk_history): Generates predictions for CPU, memory, and disk usage based on the historical data.
AnomalyDetector
__init__(self): Initializes the AnomalyDetector with default parameters.
should_train(self, data_size): Checks whether the model should be retrained based on the data size and training interval.
train(self, cpu_data, mem_data, disk_data): Trains the anomaly detection model using the historical resource usage data.
detect_anomalies(self, cpu_data, mem_data, disk_data): Detects anomalies in the current resource usage.
Process Utilities
get_process_details(pid): Retrieves detailed information about a process given its PID.
kill_process(pid): Terminates a process given its PID.
change_process_priority(pid, priority): Changes the priority of a process given its PID and the desired priority.
Dependencies
numpy: Used for numerical operations.
pandas: Used for data manipulation.
scikit-learn: Used for machine learning algorithms (Isolation Forest for anomaly detection).
statsmodels: Used for time series analysis (ARIMA model for prediction).
psutil: Used for retrieving system and process information.
