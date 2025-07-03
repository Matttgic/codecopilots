import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set\_page\_config(page\_title="🎾 Prédicteur Tennis", layout="wide")
st.title("🎾 Prédicteur de Matchs ATP")

model = joblib.load("tennis\_match\_predictor\_xgb.joblib")

def log\_prediction(input\_features, prob):
input\_features\['prediction\_prob'] = prob
input\_features\['timestamp'] = datetime.now()
log\_df = pd.DataFrame(\[input\_features])
log\_df.to\_csv("prediction\_logs.csv", mode="a", index=False, header=not os.path.exists("prediction\_logs.csv"))

def predict\_new\_match(input\_features):
input\_df = pd.DataFrame(\[input\_features])
prob = model.predict\_proba(input\_df\[\[col for col in input\_df.columns if col not in \['player\_1', 'player\_2', 'match\_date']]])\[0, 1]
log\_prediction(input\_features, prob)
return prob

def predict\_batch(df):
df = df.copy()
input\_df = df\[\[col for col in df.columns if col not in \['player\_1', 'player\_2', 'match\_date']]]
df\['prediction\_prob'] = model.predict\_proba(input\_df)\[:, 1]
df\['timestamp'] = datetime.now()
df.to\_csv("prediction\_logs.csv", mode="a", index=False, header=not os.path.exists("prediction\_logs.csv"))
return df

def retrain\_model():
if os.path.exists("train\_data.csv"):
df = pd.read\_csv("train\_data.csv")
from xgboost import XGBClassifier
from sklearn.model\_selection import train\_test\_split

```
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    joblib.dump(model, "tennis_match_predictor_xgb.joblib")
    st.success("Modèle réentraîné ✅")
else:
    st.error("Fichier train_data.csv manquant.")
```

def generate\_train\_example():
df = pd.DataFrame({
'rank\_diff': np.random.randint(-100, 100, 500),
'ace\_diff': np.random.randint(-20, 20, 500),
'df\_diff': np.random.randint(-10, 10, 500),
'win\_ratio\_diff': np.random.uniform(-1, 1, 500),
'surface\_ratio\_diff': np.random.uniform(-1, 1, 500),
'surface': np.random.randint(0, 4, 500),
'tourney\_level': np.random.randint(0, 5, 500),
'round': np.random.randint(0, 7, 500),
'target': np.random.randint(0, 2, 500)
})
df.to\_csv("train\_data.csv", index=False)
st.success("Fichier exemple train\_data.csv généré ✅")

# Tabs

tabs = st.tabs(\["🔮 Prédiction", "📊 Historique", "📁 Batch CSV", "📈 Comparaison", "🔁 Réentraîner"])

with tabs\[0]:
st.header("Entrée manuelle")
player\_1 = st.text\_input("Joueur 1", "Joueur 1")
player\_2 = st.text\_input("Joueur 2", "Joueur 2")
match\_date = st.date\_input("Date du match", datetime.today())

```
rank_diff = st.slider("Diff. classement", -200, 200, 0)
ace_diff = st.slider("Diff. aces", -50, 50, 0)
df_diff = st.slider("Diff. doubles fautes", -20, 20, 0)
win_ratio_diff = st.slider("Diff. ratio victoires", -1.0, 1.0, 0.0, step=0.01)
surface_ratio_diff = st.slider("Diff. perf surface", -1.0, 1.0, 0.0, step=0.01)
surface = st.selectbox("Surface", [("Hard", 0), ("Clay", 1), ("Grass", 2), ("Carpet", 3)], format_func=lambda x: x[0])[1]
tourney_level = st.slider("Niveau tournoi", 0, 4, 2)
round_ = st.slider("Tour (0-6)", 0, 6, 3)

if st.button("Prédire"):
    input_data = {
        'player_1': player_1,
        'player_2': player_2,
        'match_date': match_date.strftime('%Y-%m-%d'),
        'rank_diff': rank_diff,
        'ace_diff': ace_diff,
        'df_diff': df_diff,
        'win_ratio_diff': win_ratio_diff,
        'surface_ratio_diff': surface_ratio_diff,
        'surface': surface,
        'tourney_level': tourney_level,
        'round': round_
    }
    proba = predict_new_match(input_data)
    winner = player_1 if proba > 0.5 else player_2
    st.success(f"Gagnant prédit : {winner}")
    st.info(f"Probabilité de victoire : {proba:.2%}")
```

with tabs\[1]:
st.header("Historique")
if os.path.exists("prediction\_logs.csv"):
df = pd.read\_csv("prediction\_logs.csv")
df\['timestamp'] = pd.to\_datetime(df\['timestamp'])
st.dataframe(df.sort\_values(by="timestamp", ascending=False))
else:
st.info("Aucune prédiction enregistrée.")

with tabs\[2]:
st.header("Batch CSV")
uploaded\_file = st.file\_uploader("Choisir un fichier CSV", type="csv")
if uploaded\_file:
df = pd.read\_csv(uploaded\_file)
try:
result\_df = predict\_batch(df)
st.success("Prédictions terminées")
st.dataframe(result\_df\[\["player\_1", "player\_2", "match\_date", "prediction\_prob"]])
except Exception as e:
st.error(str(e))

with tabs\[3]:
st.header("Comparaison entre joueurs")
if os.path.exists("prediction\_logs.csv"):
df = pd.read\_csv("prediction\_logs.csv")
players = sorted(set(df\['player\_1'].dropna()).union(df\['player\_2'].dropna()))
p1 = st.selectbox("Joueur 1", players)
p2 = st.selectbox("Joueur 2", players, index=1)
comp = df\[((df\['player\_1'] == p1) & (df\['player\_2'] == p2)) | ((df\['player\_1'] == p2) & (df\['player\_2'] == p1))]
if not comp.empty:
fig, ax = plt.subplots()
sns.boxplot(data=comp, x='player\_1', y='prediction\_prob', ax=ax)
st.pyplot(fig)
else:
st.warning("Aucune confrontation trouvée.")

with tabs\[4]:
st.header("Réentraînement")
col1, col2 = st.columns(2)
with col1:
if st.button("Réentraîner modèle"):
retrain\_model()
with col2:
if st.button("Créer fichier d'entraînement"):
generate\_train\_example()
