import pandas as pd
import numpy as np

num_samples = 100
np.random.seed(42)

# Genera identificatori unici per ciascun match
match_id = list(range(1, num_samples + 1))

# Genera feature base con dipendenze
home_away = np.random.choice(['Home', 'Away'], size=num_samples)
weather_condition = np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=num_samples)

# Genera condizioni ambientali con bias
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

# Genera statistiche di gioco con bias
shots = np.array([np.random.randint(7, 16) if ha == 'Home' else np.random.randint(5, 14) for ha in home_away])

shots_on_goal = []
for s, wc in zip(shots, weather_condition):
    if wc == 'Rainy':
        sog = int(s * np.random.uniform(0.2, 0.5))
    elif wc == 'Cloudy':
        sog = int(s * np.random.uniform(0.3, 0.6))
    else:
        sog = int(s * np.random.uniform(0.4, 0.7))
    shots_on_goal.append(sog)
shots_on_goal = np.array(shots_on_goal)

possession = np.array([np.random.randint(50, 66) if ha == 'Home' else np.random.randint(40, 56) for ha in home_away])

# Genera xG con multiple dipendenze
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

# Creiamo un dizionario iniziale per costruire il DataFrame
data = {
    'MatchID': match_id,
    'HomeAway': home_away,
    'Shots': shots,
    'ShotsOnGoal': shots_on_goal,
    'Possession': possession,
    'xG': xg,
    'Temperature': temperature,
    'WeatherCondition': weather_condition,
    'Humidity': humidity
}

# ==========
# AGGIUNTA DELLE 4 NUOVE FEATURE PER I 22 GIOCATORI
# ==========

num_players_per_team = 11
feature_names = ["IndiceForma", "Motivazione", "Affiatamento", "CondizioneFisica"]

# Generiamo casualmente i valori per ogni giocatore di entrambe le squadre
# (in totale 22 giocatori, ognuno con 4 parametri).
# useremo un range 1-10 per rendere lineare la somma totale della squadra.

all_home_features = []
all_away_features = []

for i in range(num_samples):
    # Home team (indipendentemente dal fatto che la riga sia contrassegnata "Home" o "Away",
    # qui generiamo i valori dei 11 giocatori "home")
    home_features_match = {}
    for player_id in range(1, num_players_per_team + 1):
        for feat in feature_names:
            col_name = f"HomePlayer{player_id}_{feat}"
            home_features_match[col_name] = np.random.randint(1, 11)
    all_home_features.append(home_features_match)

    # Away team
    away_features_match = {}
    for player_id in range(1, num_players_per_team + 1):
        for feat in feature_names:
            col_name = f"AwayPlayer{player_id}_{feat}"
            away_features_match[col_name] = np.random.randint(1, 11)
    all_away_features.append(away_features_match)

# Ora aggiungiamo queste colonne al nostro dizionario `data`
# per poi costruire il DataFrame finale
for key in all_home_features[0].keys():
    data[key] = [hf[key] for hf in all_home_features]

for key in all_away_features[0].keys():
    data[key] = [af[key] for af in all_away_features]

# =================
# CALCOLO DEI GOL SEGNATI CON L'INFLUENZA DELLE NUOVE FEATURE
# =================
goals_scored = []

for i in range(num_samples):
    # xG di base
    base_xg = xg[i]

    # Vantaggio casalingo
    if home_away[i] == 'Home':
        # Somma i parametri dei 11 giocatori "home"
        sum_home_features = 0
        for player_id in range(1, num_players_per_team + 1):
            for feat in feature_names:
                col_name = f"HomePlayer{player_id}_{feat}"
                sum_home_features += data[col_name][i]

        # Calcolo media dei 44 parametri (11 giocatori x 4)
        avg_home_features = sum_home_features / (num_players_per_team * len(feature_names))

        adj_xg = base_xg + 0.2  # piccolo vantaggio casa già previsto
        # Aggiunta influenza lineare dei parametri
        adj_xg += (avg_home_features * 0.05)  # fattore di scala arbitrario

    else:  # 'Away'
        # Somma i parametri dei 11 giocatori "away"
        sum_away_features = 0
        for player_id in range(1, num_players_per_team + 1):
            for feat in feature_names:
                col_name = f"AwayPlayer{player_id}_{feat}"
                sum_away_features += data[col_name][i]

        avg_away_features = sum_away_features / (num_players_per_team * len(feature_names))

        adj_xg = base_xg
        # Aggiunta influenza lineare dei parametri
        adj_xg += (avg_away_features * 0.05)

    # Generiamo i gol tenendo conto dell'adj_xg (e di un po' di rumore)
    goals = np.clip(np.round(adj_xg + np.random.normal(0, 0.5)), 0, None)
    goals_scored.append(int(goals))

# Aggiungiamo la colonna GoalsScored al DataFrame
data['GoalsScored'] = goals_scored

# Creiamo il DataFrame finale
df = pd.DataFrame(data)

# Salviamo il DataFrame in un CSV
df.to_csv('matches_data.csv', index=False)

print(f"CSV 'matches_data.csv' creato con successo con {num_samples} campioni e feature aggiuntive!") 
