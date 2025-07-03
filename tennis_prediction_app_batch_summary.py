import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Prédicteur Tennis", layout="centered")
st.title("🎾 Prédicteur de matchs de tennis")

model = joblib.load("tennis_match_predictor_xgb.joblib")

st.header("🧮 Entrée manuelle")

rank_diff = st.slider("🔢 Différence de classement (J1 - J2)", -200, 200, 0)
st.caption("ℹ️ Positif = J1 est mieux classé que J2")

ace_diff = st.slider("🎯 Aces (J1 - J2)", -50, 50, 0)
st.caption("ℹ️ Positif = J1 sert plus d'aces que J2")

df_diff = st.slider("⚠️ Double fautes (J1 - J2)", -20, 20, 0)
st.caption("ℹ️ Positif = J1 fait plus de doubles fautes que J2")

win_ratio_diff = st.slider("📈 Différence de ratio de victoires", -1.0, 1.0, 0.0, step=0.01)
st.caption("ℹ️ Positif = J1 gagne plus souvent que J2, globalement")

surface_ratio_diff = st.slider("🎾 Diff. de perf par surface", -1.0, 1.0, 0.0, step=0.01)
st.caption("ℹ️ Positif = J1 est historiquement meilleur que J2 sur cette surface")

surface = st.selectbox(
    "🌍 Surface",
    options=[("Hard", 0), ("Clay", 1), ("Grass", 2), ("Carpet", 3)],
    format_func=lambda x: x[0]
)[1]
st.caption("ℹ️ Surface du match : dure, terre battue, gazon, moquette")

tourney_level = st.slider("🏆 Niveau du tournoi (0–4)", 0, 4, 2)
st.caption("ℹ️ 0=Challenger, 1=ATP250, 2=ATP500, 3=Masters1000, 4=Grand Chelem")

round_ = st.slider("📅 Tour (0 = 1er tour, 6 = finale)", 0, 6, 3)
st.caption("ℹ️ Niveau d'avancement dans le tournoi")

if st.button("🔮 Prédire le gagnant"):
    features = np.array([[rank_diff, ace_diff, df_diff, win_ratio_diff, surface_ratio_diff, surface, tourney_level, round_]])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][prediction]

    st.success(f"✅ Gagnant prédit : {'Joueur 1' if prediction == 1 else 'Joueur 2'}")
    st.info(f"Confiance du modèle : {prob:.2%}")
