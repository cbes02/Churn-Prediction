import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("CONVERSIONE CATEGORIE IN NUMERI")
print("="*60)

# 1. Carica il dataset pulito
print("\n[1/7] Caricamento dataset pulito...")
df = pd.read_csv('Telco-Customer-Churn-CLEAN.csv')
print(f"Dataset caricato: {df.shape}")

# 2. Identifica colonne categoriche (tipo object)
print("\n[2/7] Identificazione colonne categoriche...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Colonne categoriche trovate: {len(categorical_cols)}")
for col in categorical_cols:
    print(f"  - {col}: {df[col].nunique()} valori unici")

# 3. Separa colonne binarie (Yes/No) da quelle con piu valori
print("\n[3/7] Analisi tipi di categorie...")
binary_cols = []
multi_cols = []

for col in categorical_cols:
    unique_values = df[col].unique()
    if len(unique_values) == 2:
        binary_cols.append(col)
        print(f"  Binaria: {col} -> {list(unique_values)}")
    else:
        multi_cols.append(col)
        print(f"  Multipla: {col} -> {list(unique_values)}")

# 4. Converti colonne binarie (Yes/No -> 1/0, Male/Female -> 1/0)
print("\n[4/7] Conversione colonne binarie...")
for col in binary_cols:
    unique_vals = df[col].unique()
    print(f"  Converto {col}:")
    
    # Mappa i valori
    if 'Yes' in unique_vals:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        print(f"    Yes -> 1, No -> 0")
    elif 'Male' in unique_vals:
        df[col] = df[col].map({'Male': 1, 'Female': 0})
        print(f"    Male -> 1, Female -> 0")
    else:
        # Usa LabelEncoder per altre colonne binarie
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"    {unique_vals[0]} -> 0, {unique_vals[1]} -> 1")

# 5. Converti colonne multiple con One-Hot Encoding
print("\n[5/7] Conversione colonne multiple (One-Hot Encoding)...")
if len(multi_cols) > 0:
    for col in multi_cols:
        print(f"  One-Hot su {col}...")
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    print(f"  Nuove colonne create!")

# 6. Verifica finale
print("\n[6/7] Verifica finale...")
print(f"Dimensioni finali: {df.shape}")
print(f"\nTipi di dati (tutti devono essere numeri!):")
print(df.dtypes.value_counts())
print("\nPrime 3 righe del dataset convertito:")
print(df.head(3))

# 7. Salva il dataset pronto per ML
print("\n[7/7] Salvataggio dataset finale...")
df.to_csv('Telco-Customer-Churn-FINAL.csv', index=False)
print("Dataset finale salvato come: Telco-Customer-Churn-FINAL.csv")

print("\n" + "="*60)
print("CONVERSIONE COMPLETATA!")
print("="*60)
print("Il dataset e pronto per il Machine Learning!")
