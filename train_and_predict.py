import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import gradio as gr
import argparse

# --- Step 1: Generate or load CSV ---
def generate_csv(filename, rows=100):
    np.random.seed(42)
    X1 = np.random.uniform(0, 10, rows)
    X2 = np.random.uniform(0, 10, rows)
    noise = np.random.normal(0, 3, rows)
    Y = 3.5 * X1 + 2.1 * X2 + noise
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
    df.to_csv(filename, index=False)
    print(f"[INFO] Generated {filename}")

def load_data(filename):
    df = pd.read_csv(filename)
    X = df[['X1', 'X2']].values
    Y = df['Y'].values
    return X, Y

# --- Step 2: Model ---
def build_model():
    model = keras.Sequential([
        keras.Input(shape=(2,), name="input_layer"),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, X, Y, epochs=50):
    model.fit(X, Y, epochs=epochs, verbose=0)
    return model

def predict_with_ci(model, X, num_samples=100):
    _ = model(X[:1], training=False)  # Ensure model has .input/.output
    f_model = keras.models.Model(model.inputs, model.outputs)
    preds = np.array([f_model(X, training=True).numpy().flatten() for _ in range(num_samples)])
    mean_pred = preds.mean(axis=0)
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    return mean_pred, lower, upper

# --- Step 3: Save predictions ---
def export_to_excel(X, Y, pred, lower, upper, filename='predictions.xlsx'):
    df = pd.DataFrame(X, columns=['X1', 'X2'])
    df['Actual Y'] = Y
    df['Predicted Y'] = pred
    df['Lower 95% CI'] = lower
    df['Upper 95% CI'] = upper
    df.to_excel(filename, index=False)
    print(f"[INFO] Predictions exported to {filename}")

# --- Step 4: Plot results ---
def plot_predictions(Y_actual, Y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(Y_actual, Y_pred, alpha=0.7)
    plt.plot([Y_actual.min(), Y_actual.max()], [Y_actual.min(), Y_actual.max()], 'r--')
    plt.xlabel("Actual Y")
    plt.ylabel("Predicted Y")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.show()

# --- Step 5: Web UI ---
def launch_web_ui(model):
    def predict_ui(x1, x2):
        X_input = np.array([[x1, x2]])
        mean, low, high = predict_with_ci(model, X_input, num_samples=100)
        return {
            "Predicted Y": float(mean[0]),
            "Lower 95% CI": float(low[0]),
            "Upper 95% CI": float(high[0])
        }

    gr.Interface(
        fn=predict_ui,
        inputs=[
            gr.Slider(0, 10, step=0.1, label="X1"),
            gr.Slider(0, 10, step=0.1, label="X2")
        ],
        outputs="json",
        title="Predict Y with Confidence Interval"
    ).launch()

# --- Main logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", action="store_true", help="Launch web UI")
    args = parser.parse_args()

    csv_file = "data.csv"
    model_file = "model.keras"
    excel_file = "predictions.xlsx"

    if not os.path.exists(csv_file):
        generate_csv(csv_file)

    X, Y = load_data(csv_file)

    if os.path.exists(model_file):
        try:
            model = keras.models.load_model(model_file)
            print(f"[INFO] Loaded existing model from {model_file}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}. Training a new one.")
            model = build_model()
    else:
        model = build_model()

    # Train or fine-tune model
    model = train_model(model, X, Y, epochs=50)
    model.save(model_file)
    print(f"[INFO] Model saved to {model_file}")

    # Predictions + Excel
    pred, low, high = predict_with_ci(model, X)
    export_to_excel(X, Y, pred, low, high, filename=excel_file)

    # Show plot
    plot_predictions(Y, pred)

    # Optional UI
    if args.ui:
        launch_web_ui(model)
    print("[INFO] Done.")
# --- End of script ---
# This script generates a dataset, trains a model, makes predictions, and optionally launches a web UI.
# It can be run from the command line with the --ui flag to enable the web interface.
# The script is modular, allowing for easy updates and testing of different components.
# The model is saved and can be reused without retraining, and predictions are exported to an Excel file.
# The web UI allows for interactive predictions with confidence intervals.
# The script uses TensorFlow and Keras for model building and training, and Gradio for the web interface.
# The script is designed to be run in a Python environment with the necessary libraries installed.