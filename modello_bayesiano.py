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
        # Configurazione avanzata per NUTS
        trace = pm.sample(
            draws=50,
            tune=100,
            chains=2,
            target_accept=0.95,
            return_inferencedata=True,
            random_seed=42,
            idata_kwargs={'log_likelihood': True}
        )
        
        # Campionamento predittivo protetto
        try:
            ppc = pm.sample_posterior_predictive(
                trace,
                var_names=['obs'],
                random_seed=42
            )
        except RuntimeError as e:
            raise RuntimeError("Errore nel campionamento predittivo") from e
    
    # Analisi diagnostiche
    az.plot_trace(trace, compact=True)
    plt.tight_layout()
    plt.show()
    
    # Visualizzazione risultati
    y_pred = ppc.posterior_predictive['obs'].mean(axis=(0,1)).values
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([0, y.max()], [0, y.max()], 'r--')
    plt.xlabel('Osservato')
    plt.ylabel('Predetto')
    plt.title('Confronto Previsioni vs Realtà')
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