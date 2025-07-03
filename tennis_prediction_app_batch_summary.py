import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Prédicteur Tennis", layout="centered")
st.title("🎾 Prédicteur de matchs de tennis")

model = joblib.load("tennis_match_predictor_xgb.joblib")

st.header("🧮 Entrée manuelle")

rank_diff = st.slider(
    "Différence de classement (J1 - J2)",
    -200, 200, 0,
    help="Classement ATP : positif si J1 est mieux classé que J2."
)

ace_diff = st.slider(
    "Aces (J1 - J2)",
    -50, 50, 0,
    help="Nombre d'aces : positif si J1 sert plus d'aces que J2."
)

df_diff = st.slider(
    "Double fautes (J1 - J2)",
    -20, 20, 0,
    help="Nombre de doubles fautes : positif si J1 fait plus d'erreurs que J2."
)

win_ratio_diff = st.slider(
    "Différence de ratio de victoires",
    -1.0, 1.0, 0.0, step=0.01,
    help="Différence entre les taux de victoires globaux de J1 et J2. Positif = J1 plus gagnant."
)

surface_ratio_diff = st.slider(
    "Diff. de perf par surface",
    -1.0, 1.0, 0.0, step=0.01,
    help="Différence des performances historiques sur la surface sélectionnée."
)

surface = st.selectbox(
    "Surface",
    options=[("Hard", 0), ("Clay", 1), ("Grass", 2), ("Carpet", 3)],
    format_func=lambda x: x[0],
    help="Type de surface du match : dure, terre battue, gazon, moquette."
)[1]

tourney_level = st.slider(
    "Niveau du tournoi (0-4)",
    0, 4, 2,
    help="0=Challenger, 1=ATP250, 2=ATP500, 3=Masters1000, 4=Grand Chelem"
)

round_ = st.slider(
    "Tour (0 = 1er tour, 6 = finale)",
    0, 6, 3,
    help="Niveau d'avancement dans le tournoi. 0=début, 6=finale."
)

if st.button("Prédire"):
    features = np.array([[rank_diff, ace_diff, df_diff, win_ratio_diff, surface_ratio_diff, surface, tourney_level, round_]])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][prediction]

    st.success(f"✅ Gagnant prédit : {'Joueur 1' if prediction == 1 else 'Joueur 2'}")
    st.info(f"Confiance du modèle : {prob:.2%}") 
