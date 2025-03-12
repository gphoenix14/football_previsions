import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Per MAPE non c'è un'unica funzione built-in in sklearn < 1.2, la calcoliamo manualmente

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def load_and_preprocess_data(csv_path='matches_data.csv'):
    """
    1) Carica il CSV e rimuove righe NaN/duplicati
    2) Converte 'HomeAway' in binario (Home=1, Away=0)
    3) One-hot encoding di 'WeatherCondition' -> Weather_Cloudy, Weather_Rainy (Sunny come baseline)
    4) Crea 'ShotsOnGoalRatio' = ShotsOnGoal / Shots
       e rimuove eventuali righe con inf/-inf => trasformate in NaN => droppate
    5) Elimina colonne inutili
    6) Restituisce (df, X, y)
    """
    df = pd.read_csv(csv_path).dropna().drop_duplicates().reset_index(drop=True)
    
    if 'MatchID' in df.columns:
        df.drop(columns=['MatchID'], inplace=True)
    
    df['HomeAway_Bin'] = df['HomeAway'].map({'Home': 1, 'Away': 0})
    
    # One-hot encoding per il meteo
    weather_dummies = pd.get_dummies(df['WeatherCondition'], prefix='Weather')
    # Rimuoviamo la colonna 'Weather_Sunny' così Sunny diventa baseline implicita
    if 'Weather_Sunny' in weather_dummies.columns:
        weather_dummies.drop(columns=['Weather_Sunny'], inplace=True)
    df = pd.concat([df, weather_dummies], axis=1)
    
    # Creazione ShotsOnGoalRatio
    df['ShotsOnGoalRatio'] = df['ShotsOnGoal'] / df['Shots']
    
    # Converte inf/-inf in NaN, poi rimuove righe con NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Elimina colonne inutili
    df.drop(columns=['Shots', 'ShotsOnGoal', 'WeatherCondition', 'HomeAway'], inplace=True)
    
    # Elenco feature di base
    features = [
        'HomeAway_Bin',
        'ShotsOnGoalRatio',
        'Possession',
        'xG',
        'Temperature',
        'Humidity'
    ]
    # Aggiungiamo se presenti
    if 'Weather_Cloudy' in df.columns:
        features.append('Weather_Cloudy')
    if 'Weather_Rainy' in df.columns:
        features.append('Weather_Rainy')
    
    for col in features:
        if col not in df.columns:
            raise ValueError(f"Manca la colonna '{col}' nel dataset!")
    
    X = df[features].values
    y = df['GoalsScored'].values
    
    return df, X, y


def build_model(input_dim):
    """
    Costruisce un modello di rete neurale 'più serio':
    - 2 layer Dense nascosti con L2 e Dropout
    - Loss MSE
    - Metriche: MAE, MSE
    """
    tf.keras.backend.clear_session()
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(8, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation='linear')
    ])
    
    # Compilazione con MSE come loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model


def train_and_evaluate(csv_path='matches_data.csv'):
    """
    Esegue:
    - Caricamento + Preprocessing
    - train_test_split
    - StandardScaler
    - Costruzione modello + EarlyStopping
    - Training
    - Valutazione con metriche multiple
    - Creazione di grafici 'comunicativi'
    """
    df, X, y = load_and_preprocess_data(csv_path)
    print(f"Dimensione dataset (pulito): {len(df)} righe")
    
    # Suddivisione train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Creazione modello
    model = build_model(X_train.shape[1])
    model.summary()
    
    # EarlyStopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # Training
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=8,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Valutazione su test set
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test).flatten()
    
    # Calcolo metriche extra
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # MAPE manuale (attenzione a eventuali zero in y_test)
    eps = 1e-9
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + eps))) * 100
    
    print("\n=== METRICHE DI VALUTAZIONE (TEST) ===")
    print(f"MSE:    {mse:.4f}")
    print(f"RMSE:   {rmse:.4f}")
    print(f"MAE:    {mae:.4f}")
    print(f"MAPE:   {mape:.2f}%")
    print(f"R²:     {r2:.4f}")
    
    # -- GRAFICI --
    
    # 1) Andamento Loss nel training (solo 'loss')
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Andamento della Loss (MSE)")
    plt.xlabel("Epoche")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.show()
    
    # 2) Andamento MAE nel training
    plt.figure()
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title("Andamento della MAE")
    plt.xlabel("Epoche")
    plt.ylabel("MAE")
    plt.legend()
    plt.show()
    
    # 3) Confronto 'Valori Reali vs Predetti' (scatter)
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.7, label='Predizioni')
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfetta')
    plt.title("Confronto Valori Reali vs Predetti")
    plt.xlabel("Goals Reali")
    plt.ylabel("Goals Predetti")
    plt.legend()
    plt.show()
    
    # 4) Curva 'Valore Reale' vs 'Valore Predetto' per ogni campione del test
    #    (può essere utile quando l'ordine dei campioni in test ha un senso)
    plt.figure()
    plt.plot(y_test, label='Valori Reali')
    plt.plot(y_pred, label='Valori Predetti', linestyle='--')
    plt.title("Serie di Valori Reali e Predetti (Test Set)")
    plt.xlabel("Indice campione (Test)")
    plt.ylabel("Goals")
    plt.legend()
    plt.show()
    
    # 5) Distribuzione dei residui (y_test - y_pred)
    residuals = y_test - y_pred
    plt.figure()
    plt.hist(residuals, bins=8, alpha=0.7)
    plt.title("Distribuzione dei Residui (y_test - y_pred)")
    plt.xlabel("Residui")
    plt.ylabel("Frequenza")
    plt.show()
    
    return model, history


if __name__ == "__main__":
    train_and_evaluate("matches_data.csv")
