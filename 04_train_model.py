import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

print("="*60)
print("CREAZIONE MODELLO DI MACHINE LEARNING")
print("Customer Churn Prediction con Random Forest")
print("="*60)

# 1. Carica il dataset finale (gia pulito e convertito)
print("\n[1/8] Caricamento dataset finale...")
df = pd.read_csv('Telco-Customer-Churn-FINAL.csv')
print(f"Dataset caricato: {df.shape}")
print(f"Colonne totali: {len(df.columns)}")

# 2. Separa features (X) e target (y)
print("\n[2/8] Separazione features e target...")
# La colonna Churn e' quella che vogliamo prevedere
y = df['Churn']
X = df.drop('Churn', axis=1)

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"\nDistribuzione del Churn:")
print(y.value_counts())
print(f"Percentuale Churn=1: {(y.sum()/len(y)*100):.2f}%")

# 3. Dividi in training set (80%) e test set (20%)
print("\n[3/8] Divisione in Training e Test set...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# 4. Crea e addestra il modello Random Forest
print("\n[4/8] Addestramento del modello Random Forest...")
print("Questo puo richiedere 30-60 secondi...")

model = RandomForestClassifier(
    n_estimators=100,      # 100 alberi decisionali
    max_depth=10,          # Profondita massima
    random_state=42,
    n_jobs=-1              # Usa tutti i core disponibili
)

model.fit(X_train, y_train)
print("Modello addestrato con successo!")

# 5. Fai previsioni sul test set
print("\n[5/8] Previsioni sul test set...")
y_pred = model.predict(X_test)
print(f"Previsioni completate: {len(y_pred)} clienti")

# 6. Valuta le performance
print("\n[6/8] Valutazione delle performance...")
accuracy = accuracy_score(y_test, y_pred)
print(f"\nACCURACY (Precisione generale): {accuracy*100:.2f}%")

print("\n--- CONFUSION MATRIX ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nInterpretazione:")
print(f"  Veri Negativi (predetto No, reale No): {cm[0][0]}")
print(f"  Falsi Positivi (predetto Si, reale No): {cm[0][1]}")
print(f"  Falsi Negativi (predetto No, reale Si): {cm[1][0]}")
print(f"  Veri Positivi (predetto Si, reale Si): {cm[1][1]}")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# 7. Feature Importance (quali variabili sono piu importanti?)
print("\n[7/8] Feature Importance (Top 10)...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nVariabili piu importanti per prevedere il churn:")
print(feature_importance.head(10).to_string(index=False))

# 8. Salva il modello
print("\n[8/8] Salvataggio del modello...")
with open('churn_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Modello salvato come: churn_model.pkl")

# Salva anche le colonne (per usare il modello dopo)
with open('model_columns.pkl', 'wb') as file:
    pickle.dump(X.columns.tolist(), file)
print("Colonne salvate come: model_columns.pkl")

print("\n" + "="*60)
print("MODELLO COMPLETATO E SALVATO!")
print("="*60)
print(f"\nRISULTATI FINALI:")
print(f"  - Accuracy: {accuracy*100:.2f}%")
print(f"  - Modello salvato: churn_model.pkl")
print(f"  - Pronto per la demo all'esame!")
