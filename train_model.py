# train_model.py

import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Charger les fichiers CSV
dfs = [
    pd.read_csv("data/atp_matches_2021.csv"),
    pd.read_csv("data/atp_matches_2022.csv"),
    pd.read_csv("data/atp_matches_2023.csv")
]
df = pd.concat(dfs, ignore_index=True)

# Nettoyage
df = df.dropna(subset=['winner_name', 'loser_name', 'winner_rank', 'loser_rank', 'w_ace', 'l_ace', 'w_df', 'l_df', 'surface'])
df['rank_diff'] = df['winner_rank'] - df['loser_rank']
df['ace_diff'] = df['w_ace'] - df['l_ace']
df['df_diff'] = df['w_df'] - df['l_df']

# Encodage catégoriel
for col in ['surface', 'tourney_level', 'round']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Ratios de victoire globaux
def win_ratio(df, col, outcome):
    return df.groupby(col)[outcome].mean()

df['winner_win_ratio'] = df['winner_name'].map(win_ratio(df, 'winner_name', pd.Series(1, index=df.index)))
df['loser_win_ratio'] = df['loser_name'].map(win_ratio(df, 'loser_name', pd.Series(0, index=df.index)))
df['win_ratio_diff'] = df['winner_win_ratio'].fillna(0.5) - df['loser_win_ratio'].fillna(0.5)

# Ratios de surface
for surf in df['surface'].unique():
    surf_df = df[df['surface'] == surf]
    df.loc[df['surface'] == surf, 'winner_surface_ratio'] = df.loc[df['surface'] == surf, 'winner_name'].map(win_ratio(surf_df, 'winner_name', pd.Series(1, index=surf_df.index)))
    df.loc[df['surface'] == surf, 'loser_surface_ratio'] = df.loc[df['surface'] == surf, 'loser_name'].map(win_ratio(surf_df, 'loser_name', pd.Series(0, index=surf_df.index)))

df['surface_ratio_diff'] = df['winner_surface_ratio'].fillna(0.5) - df['loser_surface_ratio'].fillna(0.5)

# Features finales
features = ['rank_diff', 'ace_diff', 'df_diff', 'win_ratio_diff', 'surface_ratio_diff', 'surface', 'tourney_level', 'round']
X = df[features].fillna(0)
y = [1] * len(df)

# Split et entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Sauvegarde du modèle
joblib.dump(model, "tennis_match_predictor_xgb.joblib")
print("✅ Modèle entraîné et sauvegardé : tennis_match_predictor_xgb.joblib") 
