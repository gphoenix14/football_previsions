import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def generate_csv(csv_path='matches_data.csv'):
    """
    Genera e salva il file CSV con dati fittizi relativi a match di calcio.
    """
    num_samples = 50
    np.random.seed(42)

    # Identificatori unici
    match_id = list(range(1, num_samples + 1))

    # Feature di base
    home_away = np.random.choice(['Home', 'Away'], size=num_samples)
    weather_condition = np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=num_samples)

    # Umidità e temperatura
    humidity = []
    temperature = []
    for wc in weather_condition:
        if wc == 'Rainy':
            humidity.append(np.random.randint(70, 81))
            temperature.append(np.random.randint(5, 26))
        elif wc == 'Cloudy':
            humidity.append(np.random.randint(50, 71))
            temperature.append(np.random.randint(10, 31))
        else:  # Sunny
            humidity.append(np.random.randint(30, 51))
            temperature.append(np.random.randint(15, 36))
    humidity = np.array(humidity)
    temperature = np.array(temperature)

    # Tiri
    shots = np.array([
        np.random.randint(7, 16) if ha == 'Home' else np.random.randint(5, 14)
        for ha in home_away
    ])

    # Tiri in porta
    shots_on_goal = []
    for s, wc in zip(shots, weather_condition):
        if wc == 'Rainy':
            sog = int(s * np.random.uniform(0.2, 0.5))
        elif wc == 'Cloudy':
            sog = int(s * np.random.uniform(0.3, 0.6))
        else:  # Sunny
            sog = int(s * np.random.uniform(0.4, 0.7))
        shots_on_goal.append(sog)
    shots_on_goal = np.array(shots_on_goal)

    # Possesso
    possession = np.array([
        np.random.randint(50, 66) if ha == 'Home' else np.random.randint(40, 56)
        for ha in home_away
    ])

    # xG
    xg = []
    for sog, wc in zip(shots_on_goal, weather_condition):
        if wc == 'Rainy':
            base = np.random.uniform(0.2, 1.5)
        elif wc == 'Cloudy':
            base = np.random.uniform(0.3, 2.0)
        else:
            base = np.random.uniform(0.5, 2.2)
        xg_val = np.clip(base + 0.05 * sog, 0.2, 2.5)
        xg.append(round(xg_val, 1))
    xg = np.array(xg)

    # Gol segnati
    goals_scored = []
    for xg_val, ha in zip(xg, home_away):
        adj_xg = xg_val + (0.2 if ha == 'Home' else 0)
        goals = np.clip(np.round(adj_xg + np.random.normal(0, 0.5)), 0, None)
        goals_scored.append(int(goals))
    goals_scored = np.array(goals_scored)

    data = {
        'MatchID': match_id,
        'HomeAway': home_away,
        'Shots': shots,
        'ShotsOnGoal': shots_on_goal,
        'Possession': possession,
        'xG': xg,
        'GoalsScored': goals_scored,
        'Temperature': temperature,
        'WeatherCondition': weather_condition,
        'Humidity': humidity
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"CSV '{csv_path}' creato con successo con {num_samples} campioni!")

def load_and_preprocess_data(csv_path='matches_data.csv', standardize=True):
    """
    Legge il CSV, rimuove duplicati o righe vuote,
    converte variabili categoriali in numeriche
    e (opzionalmente) standardizza le feature.
    Restituisce il DataFrame originale e le matrici (X, y).
    """
    df = pd.read_csv(csv_path).dropna().drop_duplicates().reset_index(drop=True)

    # Rimuovi la colonna MatchID se esiste
    if 'MatchID' in df.columns:
        df.drop(columns=['MatchID'], inplace=True, errors='ignore')

    # Home/Away -> 1 / 0
    df['HomeAway_Bin'] = df['HomeAway'].map({'Home': 1, 'Away': 0})

    # Weather -> numerico
    weather_map = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2}
    df['WeatherCode'] = df['WeatherCondition'].map(weather_map)

    features = [
        'HomeAway_Bin',
        'Shots',
        'ShotsOnGoal',
        'Possession',
        'xG',
        'Temperature',
        'WeatherCode',
        'Humidity'
    ]
    X = df[features].values
    y = df['GoalsScored'].values

    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return df, X, y

def run_bayesian_model(csv_path='matches_data.csv', standardize=True):
    """
    Esegue un modello di regressione Poisson bayesiana con PyMC
    e mostra i grafici di confronto e analisi.
    """
    # Caricamento e preprocessing
    df, X, y = load_and_preprocess_data(csv_path, standardize)

    print(f"Caricato il dataset '{csv_path}' con {df.shape[0]} righe totali.")
    n_features = X.shape[1]
    print(f"Numero di feature: {n_features}. Dimensione di y: {len(y)}.")

    with pm.Model() as model:
        intercept = pm.Normal("Intercept", mu=0, sigma=5)
        coefs = pm.Normal("coefs", mu=0, sigma=5, shape=n_features)

        mu = intercept + pm.math.dot(X, coefs)
        lambda_ = pm.math.exp(mu)

        # Likelihood Poisson
        goals = pm.Poisson("goals", mu=lambda_, observed=y)

        trace = pm.sample(
            draws=2000,
            tune=1000,
            target_accept=0.9,
            chains=4,
            random_seed=42
        )

    with model:
        ppc = pm.sample_posterior_predictive(
            trace,
            random_seed=42,
            return_inferencedata=False
        )

    # Media delle predizioni PPC
    y_pred = ppc['goals'].mean(axis=(0, 1))

    if len(y_pred) != len(y):
        raise ValueError(
            f"Dimension mismatch: y_pred ha shape {len(y_pred)}, mentre y ha shape {len(y)}."
        )

    # Grafico goals osservati vs goals predetti
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(y)), y, label='Goals osservati', marker='o')
    plt.scatter(range(len(y_pred)), y_pred, label='Goals predetti (media PPC)', marker='x')
    plt.xlabel('Indice Partita')
    plt.ylabel('Goals')
    plt.title('Confronto: Goals osservati vs. predetti (Bayesian Poisson)')
    plt.legend()
    plt.show()

    # Trace
    az.plot_trace(trace)
    plt.show()

    # Posterior
    az.plot_posterior(trace)
    plt.show()

    print("Modello Bayesiano completato con successo!")

if __name__ == "__main__":
    # 1) Genera il CSV
    generate_csv('matches_data.csv')

    # 2) Esegui il modello su quel CSV
    run_bayesian_model(csv_path='matches_data.csv', standardize=True)
