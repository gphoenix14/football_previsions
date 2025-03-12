import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
import os

# Configurazione ambiente per riproducibilità
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(csv_path):
    """Carica e prepara i dati mantenendo la stessa struttura del modello bayesiano"""
    df = pd.read_csv(csv_path).dropna().drop_duplicates()
    
    # Feature engineering coerente
    df['HomeAway_Bin'] = df['HomeAway'].map({'Home': 1, 'Away': 0})
    df['WeatherCode'] = df['WeatherCondition'].map({'Sunny': 0, 'Cloudy': 1, 'Rainy': 2})
    
    # Selezione features identica
    player_attrs = ['IndiceForma','Motivazione','Affiatamento','CondizioneFisica']
    home_features = [f'HomePlayer{i}_{attr}' for i in range(1,12) for attr in player_attrs]
    away_features = [f'AwayPlayer{i}_{attr}' for i in range(1,12) for attr in player_attrs]
    
    base_features = [
        'HomeAway_Bin', 'Shots', 'ShotsOnGoal', 
        'Possession', 'xG', 'Temperature', 
        'WeatherCode', 'Humidity'
    ]
    
    all_features = base_features + home_features + away_features
    X = StandardScaler().fit_transform(df[all_features].values)
    y = df['GoalsScored'].values.astype(np.float32)
    
    return X, y

def build_deep_learning_model(input_shape):
    """Costruisce un modello deep learning con regolarizzazione avanzata"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,),
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.3),
        Dense(64, activation='relu',
              kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='exponential')  # Output non negativo per conteggi
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='poisson',  # Appropriate per dati di conteggio
        metrics=['mae', 'mse']
    )
    
    return model

def visualize_results(y_true, y_pred):
    """Visualizzazione avanzata per confronto diretto"""
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Confronto valori reali vs predetti
    plt.subplot(2, 2, 1)
    max_val = max(y_true.max(), y_pred.max())
    plt.scatter(y_true, y_pred, alpha=0.6, c='darkred', edgecolor='black', linewidth=0.5)
    plt.plot([0, max_val], [0, max_val], 'k--', lw=1)
    plt.xlabel('Valori Reali', fontsize=12)
    plt.ylabel('Previsioni DL', fontsize=12)
    plt.title('Confronto Diretto', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Distribuzione errori
    plt.subplot(2, 2, 2)
    errors = y_pred.flatten() - y_true
    plt.hist(errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Errore di Previsione', fontsize=12)
    plt.ylabel('Frequenza', fontsize=12)
    plt.title('Distribuzione degli Errori', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Andamento temporale (supponendo ordine temporale)
    plt.subplot(2, 1, 2)
    indices = np.arange(len(y_true))
    plt.plot(indices, y_true, 'o-', label='Reali', color='navy', markersize=5)
    plt.plot(indices, y_pred, 'X--', label='Previsioni DL', color='crimson', markersize=7)
    plt.xlabel('Indice Partita', fontsize=12)
    plt.ylabel('Goals', fontsize=12)
    plt.title('Andamento Previsioni vs Reali', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main(csv_path):
    # Caricamento dati
    X, y = load_and_preprocess_data(csv_path)
    
    # Split dati
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Costruzione modello
    model = build_deep_learning_model(X.shape[1])
    
    # Allenamento con early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=50, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Previsioni
    y_pred = model.predict(X).flatten()
    
    # Visualizzazione risultati
    visualize_results(y, y_pred)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    try:
        main('matches_data.csv')
        print("Addestramento completato con successo!")
    except Exception as e:
        print(f"Errore: {str(e)}")