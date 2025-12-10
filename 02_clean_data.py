import pandas as pd
import numpy as np

print("="*60)
print("PULIZIA DEL DATASET - STEP BY STEP")
print("="*60)

# 1. Carica il dataset originale
print("\n[1/6] Caricamento dataset originale...")
df = pd.read_csv('Telco-Customer-Churn.csv')
print(f"Dataset caricato: {len(df)} righe, {len(df.columns)} colonne")

# 2. Controlla TotalCharges (ha problemi!)
print("\n[2/6] Analisi colonna TotalCharges...")
print(f"Tipo attuale: {df['TotalCharges'].dtype}")
print(f"Primi 5 valori: {df['TotalCharges'].head().tolist()}")

# Cerca valori strani (spazi vuoti)
problematic = df[df['TotalCharges'] == ' ']
print(f"Righe con spazi vuoti in TotalCharges: {len(problematic)}")

if len(problematic) > 0:
    print("\nEsempi di righe problematiche:")
    print(problematic[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']].head())

# 3. Pulisci TotalCharges
print("\n[3/6] Pulizia TotalCharges...")
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
print(f"TotalCharges convertito in: {df['TotalCharges'].dtype}")
print(f"Valori mancanti rimasti: {df['TotalCharges'].isnull().sum()}")

# 4. Rimuovi customerID (non serve per la previsione)
print("\n[4/6] Rimozione customerID...")
df = df.drop('customerID', axis=1)
print(f"customerID rimosso. Colonne rimanenti: {len(df.columns)}")

# 5. Mostra il risultato
print("\n[5/6] Controllo finale del dataset pulito...")
print(f"Dimensioni finali: {df.shape}")
print("\nTipi di dati:")
print(df.dtypes)
print("\nValori mancanti per colonna:")
print(df.isnull().sum())

# 6. Salva il dataset pulito
print("\n[6/6] Salvataggio dataset pulito...")
df.to_csv('Telco-Customer-Churn-CLEAN.csv', index=False)
print("Dataset pulito salvato come: Telco-Customer-Churn-CLEAN.csv")

print("\n" + "="*60)
print("PULIZIA COMPLETATA!")
print("="*60)