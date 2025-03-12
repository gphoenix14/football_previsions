import pandas as pd
import numpy as np

num_samples = 50
np.random.seed(42)

# Genera identificatori unici
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

# Genera gol segnati con vantaggio casalingo
goals_scored = []
for xg_val, ha in zip(xg, home_away):
    adj_xg = xg_val + (0.2 if ha == 'Home' else 0)
    goals = np.clip(np.round(adj_xg + np.random.normal(0, 0.5)), 0, None)
    goals_scored.append(int(goals))
goals_scored = np.array(goals_scored)

# Crea DataFrame
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
df.to_csv('matches_data.csv', index=False)

print(f"CSV 'matches_data.csv' creato con successo con {num_samples} campioni!")