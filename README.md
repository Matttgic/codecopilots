🎾 Tennis Match Winner Predictor

Predict tennis match winners using historical ATP match data and a machine learning model built with XGBoost.


---

🚀 Features

Upload your own batch of matches in CSV format

Predict the winner for each match

Summarize predictions in a user-friendly Streamlit app

Model trained on 2021–2023 ATP data



---

👀 How to Use the Streamlit App

📁 Upload Batch CSV

Upload a CSV file that contains one or more matches you want to predict. The file should contain:

player_1,player_2,surface,tourney_level,round,player_1_rank,player_2_rank,player_1_ace,player_2_ace,player_1_df,player_2_df

🔄 Predict Matches

Once your CSV is uploaded, the app will automatically generate predictions for each row.

📊 Summary Tab

View a summary table with the following info:

Match: player_1 vs player_2

Surface: hard, clay, grass, etc.

Predicted Winner: model's prediction

Confidence: model probability score


🔹 Input Sliders (Single Match Tab)

Diff. de perf par surface → surface_ratio_diff

> Différence entre les taux de victoire de J1 et J2 sur la surface choisie. Valeurs entre -1.0 (J2 meilleur) et +1.0 (J1 meilleur).



Différence de ratio de victoires → win_ratio_diff

> Différence globale de performance entre J1 et J2, tous tournois et surfaces confondus.



Niveau du tournoi (0–4) → tourney_level

> Niveau de prestige du tournoi : 0 (Challenger) → 4 (Grand Chelem).



Tour (0 = 1er tour, 6 = finale) → round

> Indique le stade du tournoi (plus la valeur est élevée, plus on est proche de la finale).





---

🧠 Model Info

Algorithm: XGBoost Classifier

Features: rank diff, ace diff, double faults diff, surface performance, win ratios

Training: 2021–2023 ATP matches

Output: binary classification (player 1 wins or not)



---

📁 Project Structure

.
├── data/
│   ├── atp_matches_2021.csv
│   ├── atp_matches_2022.csv
│   └── atp_matches_2023.csv
├── train_model.py
├── tennis_match_predictor_xgb.joblib
├── tennis_prediction_app_batch_summary.py
└── requirements.txt


---

🛆 Local Installation

pip install -r requirements.txt
streamlit run tennis_prediction_app_batch_summary.py


---

🌐 Deploy on Streamlit Cloud

1. Fork this repo to your GitHub


2. Go to streamlit.io/cloud


3. Create new app → connect your GitHub repo


4. Set script path to: tennis_prediction_app_batch_summary.py


5. Click Deploy




---

📂 Batch Input Format

Upload a CSV with the following columns:

player_1,player_2,surface,tourney_level,round,player_1_rank,player_2_rank,player_1_ace,player_2_ace,player_1_df,player_2_df


---

🙌 Credits

ATP data: Jeff Sackmann's tennis_atp repo

ML by XGBoost



---

📬 Contact

Made by @Matt — pull requests welcome!

