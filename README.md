# Customer Churn Prediction

## Descrizione del Progetto
Questo progetto utilizza tecniche di Machine Learning per prevedere se un cliente di una compagnia telefonica abbandonerà il servizio (churn). Il modello analizza dati demografici, informazioni sui servizi e dettagli contrattuali per identificare i clienti a rischio.

## Dataset
- **Fonte**: Kaggle - Telco Customer Churn
- **Link**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Dimensioni**: 7043 clienti, 21 caratteristiche originali
- **Target**: Churn (Yes/No)

## Struttura del Progetto
```
Churn_Prediction/
├── 01_explore_data.py          # Esplorazione iniziale del dataset
├── 02_clean_data.py            # Pulizia e preprocessing
├── 03_encode_categories.py     # Conversione categorie in numeri
├── 04_train_model.py           # Addestramento del modello
├── Telco-Customer-Churn.csv    # Dataset originale
├── Telco-Customer-Churn-CLEAN.csv   # Dataset pulito
├── Telco-Customer-Churn-FINAL.csv   # Dataset pronto per ML
├── churn_model.pkl             # Modello salvato
├── model_columns.pkl           # Colonne del modello
├── README.md                   # Questo file
├── LICENSE                     # Licenza MIT
└── ACKNOWLEDGMENTS             # Riconoscimenti
```

## Requisiti
- Python 3.8+
- pandas
- numpy
- scikit-learn
- pickle

## Installazione
1. Clona la repository
2. Crea un ambiente virtuale Anaconda:
   ```bash
   conda create -n churn_env python=3.9
   conda activate churn_env
   ```
3. Installa le dipendenze:
   ```bash
   pip install pandas scikit-learn numpy
   ```

## Utilizzo
Esegui i file in ordine:

```bash
# 1. Esplora il dataset
python 01_explore_data.py

# 2. Pulisci i dati
python 02_clean_data.py

# 3. Converti le categorie
python 03_encode_categories.py

# 4. Addestra il modello
python 04_train_model.py
```

## Risultati
- **Accuracy**: 80.41%
- **Modello**: Random Forest Classifier (100 alberi)
- **Variabili più importanti**:
  1. Tenure (durata del contratto)
  2. Total Charges (costi totali)
  3. Monthly Charges (costi mensili)

## Interpretazione
Il modello identifica correttamente 8 clienti su 10 che abbandoneranno il servizio. Le variabili più importanti suggeriscono che i clienti con contratti brevi e costi elevati sono più a rischio di churn.

## Autore
Sara Rebecca Magarotto - Progetto per corso di Intelligenza Artificiale, Impresa e Società - IULM

## Data
Dicembre 2025
