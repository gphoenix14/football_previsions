import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Configurazione ambiente per evitare warning
os.environ["PYTENSOR_FLAGS"] = "cxx="
np.seterr(all='ignore')  # Disabilita warning numerici

def load_data(csv_path):
    """Carica e prepara i dati con controllo delle dimensioni"""
    df = pd.read_csv(csv_path).dropna().drop_duplicates()
    
    # Mappature e feature engineering
    df['HomeAway_Bin'] = df['HomeAway'].map({'Home': 1, 'Away': 0})
    df['WeatherCode'] = df['WeatherCondition'].map({'Sunny': 0, 'Cloudy': 1, 'Rainy': 2})
    
    # Selezione features
    player_attrs = ['IndiceForma','Motivazione','Affiatamento','CondizioneFisica']
    home_features = [f'HomePlayer{i}_{attr}' for i in range(1,12) for attr in player_attrs]
    away_features = [f'AwayPlayer{i}_{attr}' for i in range(1,12) for attr in player_attrs]
    
    base_features = [
        'HomeAway_Bin', 'Shots', 'ShotsOnGoal', 
        'Possession', 'xG', 'Temperature', 
        'WeatherCode', 'Humidity'
    ]
    
    X = df[base_features + home_features + away_features].values
    y = df['GoalsScored'].values.astype(np.float32)
    
    return StandardScaler().fit_transform(X), y

def build_model(X, y):
    """Costruisce il modello Bayesiano con regolarizzazione"""
    with pm.Model() as model:
        # Priori regolarizzati
        intercept = pm.Normal('Intercept', mu=0, sigma=1)
        coefs = pm.Laplace('coefs', mu=0, b=0.1, shape=X.shape[1])
        
        # Linear predictor con stabilizzazione numerica
        mu = pm.math.clip(intercept + pm.math.dot(X, coefs), -10, 10)
        lambda_ = pm.math.exp(mu)
        
        # Likelihood con dimensione esplicita
        pm.Poisson('obs', mu=lambda_, observed=y, shape=y.shape[0])
        
    return model

def run_analysis(csv_path):
    """Pipeline completa di analisi"""
    X, y = load_data(csv_path)
    model = build_model(X, y)
    
    with model:
        trace = pm.sample(
            draws=50,
            tune=100,
            chains=2,
            target_accept=0.95,
            return_inferencedata=True,
            random_seed=42,
            idata_kwargs={'log_likelihood': True}
        )
        
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=['obs'],
            random_seed=42
        )
    
    # Analisi diagnostiche
    az.plot_trace(trace, compact=True)
    plt.tight_layout()
    plt.show()
    
    # Visualizzazione avanzata
    y_pred = ppc.posterior_predictive['obs'].mean(axis=(0,1)).values
    indices = np.arange(len(y))
    
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Confronto diretto
    plt.subplot(2, 1, 1)
    plt.scatter(indices, y, label='Valori Reali', color='blue', 
                alpha=0.7, s=100, marker='o', edgecolor='black')
    plt.scatter(indices, y_pred, label='Previsioni', color='red', 
                alpha=0.7, s=100, marker='X', linewidths=1.5)
    
    # Linee di connessione per evidenziare le differenze
    for i, (true, pred) in enumerate(zip(y, y_pred)):
        plt.plot([i, i], [true, pred], 'grey', alpha=0.3, linestyle='--')
    
    plt.title('Confronto Dettagliato: Valori Reali vs Previsioni', fontsize=14)
    plt.xlabel('Indice Partita', fontsize=12)
    plt.ylabel('Numero di Goal', fontsize=12)
    plt.xticks(indices, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Subplot 2: Residui
    plt.subplot(2, 1, 2)
    residuals = y_pred - y
    colors = ['red' if res >= 0 else 'blue' for res in residuals]
    plt.bar(indices, residuals, color=colors, alpha=0.6, edgecolor='black')
    
    plt.title('Analisi dei Residui (Previsioni - Valori Reali)', fontsize=14)
    plt.xlabel('Indice Partita', fontsize=12)
    plt.ylabel('Errore di Previsione', fontsize=12)
    plt.xticks(indices, rotation=45)
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        run_analysis('matches_data.csv')
        print("Analisi completata con successo!")
    except Exception as e:
        print(f"ERRORE CRITICO: {str(e)}")
        print("Consigli:")
        print("- Verificare la consistenza dei dati in input")
        print("- Ridurre il numero di feature")
        print("- Aumentare la regolarizzazione dei priori")