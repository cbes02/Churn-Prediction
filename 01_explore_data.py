import pandas as pd
import numpy as np

# Carica il dataset
print("Caricamento dataset...")
df = pd.read_csv('Telco-Customer-Churn.csv')

# Prime informazioni generali
print("\n" + "="*50)
print("INFORMAZIONI GENERALI")
print("="*50)
print(f"Numero di clienti (righe): {len(df)}")
print(f"Numero di caratteristiche (colonne): {len(df.columns)}")

print("\n" + "="*50)
print("NOMI DELLE COLONNE")
print("="*50)
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n" + "="*50)
print("PRIME 5 RIGHE DEL DATASET")
print("="*50)
print(df.head())

print("\n" + "="*50)
print("TIPI DI DATI")
print("="*50)
print(df.dtypes)

print("\n" + "="*50)
print("VALORI MANCANTI")
print("="*50)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("Nessun valore mancante trovato!")
else:
    print(missing[missing > 0])

print("\n" + "="*50)
print("DISTRIBUZIONE DEL CHURN (cosa vogliamo prevedere)")
print("="*50)
print(df['Churn'].value_counts())
print("\nIn percentuale:")
print(df['Churn'].value_counts(normalize=True) * 100)

print("\n" + "="*50)
print("STATISTICHE SULLE COLONNE NUMERICHE")
print("="*50)
print(df.describe())

print("\n✅ Esplorazione completata!")