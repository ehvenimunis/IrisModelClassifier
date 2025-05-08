import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_model(hidden_neurons=8, activation='sigmoid'):
    model = Sequential([
        Dense(hidden_neurons, activation=activation, input_shape=(4,)),
        Dropout(0.3),
        Dense(hidden_neurons, activation=activation),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    return model

def train_and_evaluate_model(hidden_neurons, activation, learning_rate, X_train, X_test, y_train, y_test):
    model = create_model(hidden_neurons, activation)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train,
                       validation_data=(X_test, y_test),
                       epochs=100,
                       batch_size=8,
                       callbacks=[early_stopping],
                       verbose=0)
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_classes),
        'precision': precision_score(y_test, y_pred_classes, average='macro'),
        'recall': recall_score(y_test, y_pred_classes, average='macro'),
        'f1': f1_score(y_test, y_pred_classes, average='macro')
    }
    
    return metrics, history

def plot_training_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define configurations to test
    configurations = [
        {'hidden_neurons': 8, 'activation': 'sigmoid'},
        {'hidden_neurons': 16, 'activation': 'sigmoid'},
        {'hidden_neurons': 8, 'activation': 'relu'},
        {'hidden_neurons': 8, 'activation': 'tanh'},
        {'hidden_neurons': 8, 'activation': 'leaky_relu'}
    ]
    
    learning_rates = [0.01, 0.001, 0.0001]
    
    # Train and evaluate models
    for config in configurations:
        print(f"\nConfiguration: {config['hidden_neurons']} neurons, {config['activation']} activation")
        print("-" * 50)
        
        results = []
        histories = []
        
        for lr in learning_rates:
            metrics, history = train_and_evaluate_model(
                config['hidden_neurons'],
                config['activation'],
                lr,
                X_train, X_test, y_train, y_test
            )
            
            results.append({
                'Learning Rate': lr,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1']
            })
            histories.append(history)
        
        # Print results table
        results_df = pd.DataFrame(results)
        print("\nResults:")
        print(results_df.to_string(index=False))
        
        # Plot training history for each learning rate
        for i, lr in enumerate(learning_rates):
            plot_training_history(
                histories[i],
                f"Training History - {config['hidden_neurons']} neurons, {config['activation']}, lr={lr}"
            )

if __name__ == "__main__":
    main()
