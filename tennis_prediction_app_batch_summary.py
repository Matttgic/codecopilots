import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="🎾 Prédicteur Tennis", layout="wide")
st.title("🎾 Prédicteur de Matchs ATP")

model = joblib.load("tennis_match_predictor_xgb.joblib")

def log_prediction(input_features, prob):
    input_features['prediction_prob'] = prob
    input_features['timestamp'] = datetime.now()
    log_df = pd.DataFrame([input_features])
    log_df.to_csv("prediction_logs.csv", mode="a", index=False, header=not os.path.exists("prediction_logs.csv"))

def predict_new_match(input_features):
    input_df = pd.DataFrame([input_features])
    prob = model.predict_proba(input_df[[col for col in input_df.columns if col not in ['player_1', 'player_2', 'match_date']]])[0, 1]
    log_prediction(input_features, prob)
    return prob

def predict_batch(df):
    df = df.copy()
    input_df = df[[col for col in df.columns if col not in ['player_1', 'player_2', 'match_date']]]
    df['prediction_prob'] = model.predict_proba(input_df)[:, 1]
    df['timestamp'] = datetime.now()
    df.to_csv("prediction_logs.csv", mode="a", index=False, header=not os.path.exists("prediction_logs.csv"))
    return df

pages = st.tabs(["🔮 Prédiction", "📊 Historique", "📁 Matchs du jour", "📈 Comparaison", "🔁 Réentraîner"])

with pages[0]:
    st.header("Entrée manuelle")
    player_1 = st.text_input("Joueur 1", "Joueur 1")
    player_2 = st.text_input("Joueur 2", "Joueur 2")
    match_date = st.date_input("Date du match", datetime.today())

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

with pages[1]:
    st.header("Historique")
    if os.path.exists("prediction_logs.csv"):
        df = pd.read_csv("prediction_logs.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.dataframe(df.sort_values(by="timestamp", ascending=False))
    else:
        st.info("Aucune prédiction enregistrée.")

with pages[2]:
    st.header("📁 Prédictions pour tous les matchs du jour")
    st.markdown("""
    Envoyez un fichier `.csv` contenant **tous les matchs du jour** avec les colonnes suivantes :
    - `player_1`, `player_2`, `match_date`
    - `rank_diff`, `ace_diff`, `df_diff`, `win_ratio_diff`, `surface_ratio_diff`
    - `surface` (0=Hard, 1=Clay, 2=Grass, 3=Carpet), `tourney_level`, `round`
    """)

    uploaded_file = st.file_uploader("🔼 Déposer votre CSV des matchs du jour", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        try:
            result_df = predict_batch(df)
            st.success("✅ Prédictions générées !")
            show_table = st.radio("Afficher ou télécharger les résultats ?", ["📄 Afficher dans l'app", "📥 Télécharger un fichier tableur"])
            if show_table == "📄 Afficher dans l'app":
                st.dataframe(result_df[["player_1", "player_2", "match_date", "prediction_prob"]])
            else:
                st.download_button("📥 Télécharger les prédictions", result_df.to_csv(index=False).encode('utf-8'), "matchs_du_jour_pred.csv", "text/csv")
        except Exception as e:
            st.error(f"Erreur : {e}")

with pages[3]:
    st.header("Comparaison entre joueurs")
    if os.path.exists("prediction_logs.csv"):
        df = pd.read_csv("prediction_logs.csv")
        players = sorted(set(df['player_1'].dropna()).union(df['player_2'].dropna()))
        p1 = st.selectbox("Joueur 1", players)
        p2 = st.selectbox("Joueur 2", players, index=1)
        comp = df[((df['player_1'] == p1) & (df['player_2'] == p2)) | ((df['player_1'] == p2) & (df['player_2'] == p1))]
        if not comp.empty:
            fig, ax = plt.subplots()
            sns.boxplot(data=comp, x='player_1', y='prediction_prob', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Aucune confrontation trouvée.")

with pages[4]:
    st.header("Réentraînement")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Réentraîner modèle"):
            if os.path.exists("train_data.csv"):
                df = pd.read_csv("train_data.csv")
                from xgboost import XGBClassifier
                from sklearn.model_selection import train_test_split

                X = df.drop(columns=['target'])
                y = df['target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                joblib.dump(model, "tennis_match_predictor_xgb.joblib")
                st.success("Modèle réentraîné ✅")
            else:
                st.error("Fichier train_data.csv manquant.")
    with col2:
        if st.button("Créer fichier d'entraînement"):
            df = pd.DataFrame({
                'rank_diff': np.random.randint(-100, 100, 500),
                'ace_diff': np.random.randint(-20, 20, 500),
                'df_diff': np.random.randint(-10, 10, 500),
                'win_ratio_diff': np.random.uniform(-1, 1, 500),
                'surface_ratio_diff': np.random.uniform(-1, 1, 500),
                'surface': np.random.randint(0, 4, 500),
                'tourney_level': np.random.randint(0, 5, 500),
                'round': np.random.randint(0, 7, 500),
                'target': np.random.randint(0, 2, 500)
            })
            df.to_csv("train_data.csv", index=False)
            st.success("Fichier exemple train_data.csv généré ✅")
