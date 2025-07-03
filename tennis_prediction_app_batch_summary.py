# tennis_prediction_app_batch_summary.py

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, log_loss, brier_score_loss
)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Prediction logging
def log_prediction(input_features, prob):
    input_features['prediction_prob'] = prob
    input_features['timestamp'] = datetime.now()
    log_df = pd.DataFrame([input_features])
    log_df.to_csv("prediction_logs.csv", mode="a", index=False, header=not os.path.exists("prediction_logs.csv"))

# Predict using saved model
def predict_new_match(input_features):
    model = joblib.load("tennis_match_predictor_xgb.joblib")
    input_df = pd.DataFrame([input_features])
    prob = model.predict_proba(input_df)[0, 1]
    log_prediction(input_features, prob)
    return prob

# Predict in batch mode
def predict_batch(df):
    model = joblib.load("tennis_match_predictor_xgb.joblib")
    df = df.copy()
    df['prediction_prob'] = model.predict_proba(df)[:, 1]
    df['timestamp'] = datetime.now()
    df.to_csv("prediction_logs.csv", mode="a", index=False, header=not os.path.exists("prediction_logs.csv"))
    return df

# Streamlit App
st.set_page_config(page_title="🎾 Prédicteur Matchs Tennis", layout="wide")
st.title("🎾 Prédicteur de Matchs ATP")

tabs = st.tabs(["🔮 Prédiction", "📊 Logs des Prédictions", "📁 Prédictions en lot"])

# Tab 1 - Single Match Prediction
with tabs[0]:
    with st.form("match_form"):
        st.subheader("Entrez les statistiques du match")
        rank_diff = st.slider("Différence de classement (J1 - J2)", -200, 200, 0)
        ace_diff = st.slider("Différence d'aces (J1 - J2)", -50, 50, 0)
        df_diff = st.slider("Double fautes (J1 - J2)", -20, 20, 0)
        win_ratio_diff = st.slider("Différence de ratio de victoires", -1.0, 1.0, 0.0, step=0.01)
        surface_ratio_diff = st.slider("Diff. de perf par surface", -1.0, 1.0, 0.0, step=0.01)
        surface = st.selectbox("Surface", options=[("Hard", 0), ("Clay", 1), ("Grass", 2), ("Carpet", 3)])
        tourney_level = st.slider("Niveau du tournoi (0-4)", 0, 4, 2)
        round_ = st.slider("Tour (0 = 1er tour, 6 = finale)", 0, 6, 3)

        submitted = st.form_submit_button("Prédire")
        if submitted:
            input_data = {
                'rank_diff': rank_diff,
                'ace_diff': ace_diff,
                'df_diff': df_diff,
                'win_ratio_diff': win_ratio_diff,
                'surface_ratio_diff': surface_ratio_diff,
                'surface': surface[1],
                'tourney_level': tourney_level,
                'round': round_
            }
            proba = predict_new_match(input_data)
            st.success(f"✅ Probabilité que le Joueur 1 gagne : {proba:.2%}")

# Tab 2 - Logs Viewer
with tabs[1]:
    st.subheader("Historique des Prédictions")
    if os.path.exists("prediction_logs.csv"):
        logs_df = pd.read_csv("prediction_logs.csv")
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])

        col1, col2 = st.columns(2)
        with col1:
            min_date = logs_df['timestamp'].min().date()
            max_date = logs_df['timestamp'].max().date()
            date_range = st.date_input("Filtrer par date", [min_date, max_date], min_value=min_date, max_value=max_date)

        if len(date_range) == 2:
            logs_df = logs_df[(logs_df['timestamp'].dt.date >= date_range[0]) & (logs_df['timestamp'].dt.date <= date_range[1])]

        st.download_button("📁 Télécharger les logs filtrés en CSV", logs_df.to_csv(index=False).encode('utf-8'), "logs_filtrés.csv", "text/csv")

        col3, col4 = st.columns(2)
        with col3:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(logs_df['prediction_prob'], bins=20, kde=True, ax=ax1)
            ax1.set_title("Distribution des probabilités prédites")
            st.pyplot(fig1)

        with col4:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=logs_df, x='round', ax=ax2)
            ax2.set_title("Nombre de prédictions par round")
            st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=logs_df, x='surface', ax=ax3)
        ax3.set_title("Nombre de prédictions par surface")
        st.pyplot(fig3)

        st.dataframe(logs_df.sort_values(by="timestamp", ascending=False))
    else:
        st.warning("Aucune prédiction enregistrée pour le moment.")

# Tab 3 - Batch Upload
with tabs[2]:
    st.subheader("📥 Uploader un fichier CSV pour des prédictions en lot")
    st.markdown("Colonnes requises : `rank_diff`, `ace_diff`, `df_diff`, `win_ratio_diff`, `surface_ratio_diff`, `surface`, `tourney_level`, `round`")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        try:
            pred_df = predict_batch(input_df)
            st.success("✅ Prédictions générées avec succès !")
            st.dataframe(pred_df)

            st.markdown("### 📈 Statistiques globales des prédictions")
            col1, col2, col3 = st.columns(3)
            col1.metric("Moyenne", f"{pred_df['prediction_prob'].mean():.2%}")
            col2.metric("Min", f"{pred_df['prediction_prob'].min():.2%}")
            col3.metric("Max", f"{pred_df['prediction_prob'].max():.2%}")

            st.download_button("📥 Télécharger les résultats", pred_df.to_csv(index=False).encode('utf-8'), "predictions_batch.csv", "text/csv")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

st.markdown("---")
st.markdown("© 2025 - Modèle XGBoost ATP - Déployable sur Streamlit Cloud")
