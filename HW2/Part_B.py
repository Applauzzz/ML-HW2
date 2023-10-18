import mlrose_hiive as mlrose
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

def train_nn_with_ro(X_train_scaled, y_train, algorithm, max_iters):
    # Neural Network parameters
    hidden_nodes = [64]
    activation = 'relu'
    algorithm = algorithm
    max_attempts = 200
    
    nn_model = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation=activation,
                                    algorithm=algorithm, max_iters=max_iters,
                                    bias=True, is_classifier=True, learning_rate=0.1,
                                    early_stopping=True, max_attempts=max_attempts,
                                    random_state=0, curve=True)
    
    nn_model.fit(X_train_scaled, y_train)
    return nn_model, nn_model.fitness_curve

# ... [Your existing code for loading and preprocessing data here] ...
data = pd.read_csv('./daily_weather.csv')
del data['number']
data = data.dropna()

# Creating target column for high humidity
data['high_humidity_label'] = (data['relative_humidity_3pm'] > 24.99) * 1

# Features for prediction
morning_features = ['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am',
                    'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am',
                    'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am']

X = data[morning_features]
y = data['high_humidity_label']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize variables for plotting
iterations = list(range(200, 2001, 100))
accuracies = []
training_accuracies = []
times = []

# Train with Backprop
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=0)
mlp.fit(X_train_scaled, y_train)
y_pred_backprop = mlp.predict(X_test_scaled)
accuracy_backprop = accuracy_score(y_test, y_pred_backprop)
print(f"Backprop Accuracy: {accuracy_backprop * 100:.2f}%")

mlp.fit(X_train_scaled, y_train)
loss_curve_backprop = mlp.loss_curve_

nn_rhc, curve_rhc = train_nn_with_ro(X_train_scaled, y_train, 'random_hill_climb', 2000)

nn_sa, curve_sa = train_nn_with_ro(X_train_scaled, y_train, 'simulated_annealing', 2000)

nn_ga, curve_ga = train_nn_with_ro(X_train_scaled, y_train, 'genetic_alg', 2000)

# Plotting
plt.figure(figsize=(12, 6))
print(loss_curve_backprop)
print(curve_rhc)
print(curve_sa)
print(curve_ga)
max = 200
plt.plot(range(max),loss_curve_backprop[:max], label="Backpropagation", color='green')
plt.plot(range(max),curve_rhc[:max,0], label="Random Hill Climb", color='blue')
plt.plot(range(max),curve_sa[:max,0], label="Simulated Annealing", color='red')
plt.plot(range(max),curve_ga[:max,0], label="Genetic Algorithm", color='purple')

plt.title("Neural Network Training - Learning Curves")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

plt.show()