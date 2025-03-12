import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def load_and_preprocess_data(csv_path='matches_data.csv'):
    """
    Caricamento e preprocessing dei dati con:
    - Codifica one-hot per WeatherCondition
    - Creazione feature ShotsOnGoalRatio
    - Rimozione feature ridondanti
    """
    df = pd.read_csv(csv_path).dropna().drop_duplicates().reset_index(drop=True)
    
    if 'MatchID' in df.columns:
        df.drop(columns=['MatchID'], inplace=True)
    
    # Conversione Home/Away a binario
    df['HomeAway_Bin'] = df['HomeAway'].map({'Home': 1, 'Away': 0})
    
    # One-hot encoding per WeatherCondition
    weather_dummies = pd.get_dummies(df['WeatherCondition'], prefix='Weather', drop_first=True)
    df = pd.concat([df, weather_dummies], axis=1)
    
    # Creazione nuova feature e rimozione colonne originali
    df['ShotsOnGoalRatio'] = df['ShotsOnGoal'] / df['Shots']
    df['ShotsOnGoalRatio'].replace([np.inf, -np.inf], 0, inplace=True)
    df.drop(columns=['Shots', 'ShotsOnGoal', 'WeatherCondition', 'HomeAway'], inplace=True)
    
    # Lista feature finali
    features = [
        'HomeAway_Bin',
        'ShotsOnGoalRatio',
        'Possession',
        'xG',
        'Temperature',
        'Humidity',
        'Weather_Cloudy',
        'Weather_Rainy'
    ]
    
    # Verifica presenza colonne
    for col in features:
        if col not in df.columns:
            raise ValueError(f"Colonna '{col}' mancante nel dataset!")
    
    X = df[features].values
    y = df['GoalsScored'].values
    
    return df, X, y

def build_model(input_dim):
    """Costruzione modello più semplice con regularizzazione integrata"""
    tf.keras.backend.clear_session()
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.GaussianNoise(0.1),  # Aumento dati tramite noise
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='poisson',  # Loss function per dati di conteggio
        metrics=['mae']
    )
    return model

def train_and_evaluate(csv_path='matches_data.csv'):
    """Pipeline completa con valutazione migliorata"""
    # Caricamento dati
    df, X, y = load_and_preprocess_data(csv_path)
    print(f"Dimensione dataset: {len(df)} righe")
    
    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardizzazione corretta (post-split)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Costruzione modello
    model = build_model(X_train.shape[1])
    model.summary()
    
    # Early stopping con pazienza aumentata
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Addestramento
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=8,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Valutazione
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Metriche aggiuntive
    if len(X_test) >= 2:
        y_pred = model.predict(X_test).flatten()
        print(f"R²: {r2_score(y_test, y_pred):.4f}")
        print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
        
        # Grafico predizioni vs osservati
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
        plt.xlabel('Valori Reali')
        plt.ylabel('Predizioni')
        plt.title('Confronto Predizioni vs Valori Reali')
        plt.show()
    else:
        print("Test set troppo piccolo per metriche aggiuntive")
    
    # Grafico andamento training
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Andamento Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Andamento MAE')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model, history

if __name__ == "__main__":
    model, history = train_and_evaluate()