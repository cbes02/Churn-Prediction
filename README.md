# Customer Churn Prediction

Un progetto di Machine Learning per predire l'abbandono dei clienti di una compagnia telefonica utilizzando tecniche di classificazione supervisionata. Il sistema analizza dati demografici, informazioni sui servizi e dettagli contrattuali per identificare i clienti a rischio di churn.

## Descrizione del Progetto

Questo progetto utilizza tecniche di Machine Learning per prevedere se un cliente di una compagnia telefonica abbandonerà il servizio (churn). Il modello analizza dati demografici, informazioni sui servizi e dettagli contrattuali per identificare i clienti a rischio.

L'obiettivo principale è fornire uno strumento predittivo che permetta all'azienda di implementare strategie di retention mirate, riducendo il tasso di abbandono e migliorando la customer satisfaction.

Il progetto segue una pipeline completa di Data Science:
- Esplorazione e analisi esplorativa dei dati (EDA)
- Pulizia e preprocessing del dataset
- Feature engineering e encoding delle variabili categoriche
- Training e validazione del modello predittivo
- Valutazione delle performance e interpretazione dei risultati

## Modello di Classificazione

Per questo progetto è stato utilizzato il *Random Forest Classifier*, un algoritmo di ensemble learning basato su alberi decisionali. Il modello è stato configurato con i seguenti parametri:

•⁠  ⁠*Numero di alberi*: 100
•⁠  ⁠*Criterio di split*: Gini impurity
•⁠  ⁠*Validation Strategy*: Train-test split (80-20)
•⁠  ⁠*Seed (random_state)*: Impostato per reproducibilità

## Dataset

Il progetto utilizza il dataset **Telco Customer Churn**, una risorsa ampiamente utilizzata nella community di Data Science per problemi di classificazione binaria.

- **Fonte**: Kaggle - Telco Customer Churn
- **Link**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Dimensioni**: 7043 clienti, 21 caratteristiche originali
- **Target**: Churn (Yes/No) - variabile binaria che indica se il cliente ha abbandonato il servizio
- **Tipologia**: Dataset strutturato con variabili numeriche e categoriche

### Caratteristiche principali del dataset:
- Informazioni demografiche (genere, età, presenza di familiari)
- Servizi sottoscritti (telefono, internet, sicurezza online, backup, ecc.)
- Dettagli contrattuali (tipo di contratto, metodo di pagamento, fatturazione)
- Metriche finanziarie (costi mensili, costi totali, durata del contratto)

## Struttura del Progetto
```
Churn_Prediction/
├── 01_explore_data.py              # Esplorazione iniziale del dataset
├── 02_clean_data.py                # Pulizia e preprocessing
├── 03_encode_categories.py         # Conversione categorie in numeri
├── 04_train_model.py               # Addestramento del modello
├── Telco-Customer-Churn.csv        # Dataset originale
├── Telco-Customer-Churn-CLEAN.csv  # Dataset pulito
├── Telco-Customer-Churn-FINAL.csv  # Dataset pronto per ML
├── churn_model.pkl                 # Modello salvato
├── model_columns.pkl               # Colonne del modello
├── README.md                       # Questo file
├── LICENSE                         # Licenza MIT
└── ACKNOWLEDGMENTS                 # Riconoscimenti
```

### Descrizione dei file Python:

1. **01_explore_data.py**: Analisi esplorativa dei dati (EDA) con statistiche descrittive, distribuzione delle variabili e identificazione di valori mancanti o anomali
2. **02_clean_data.py**: Pulizia del dataset, gestione dei missing values, rimozione di duplicati e normalizzazione dei dati
3. **03_encode_categories.py**: Conversione delle variabili categoriche in formato numerico tramite encoding (Label Encoding, One-Hot Encoding)
4. **04_train_model.py**: Training del modello Random Forest, tuning degli iperparametri, validazione e salvataggio del modello finale

## Requisiti

### Software e Librerie

- **Python 3.8+**
- **pandas** - Manipolazione e analisi dei dati
- **numpy** - Operazioni numeriche e array
- **scikit-learn** - Algoritmi di Machine Learning e metriche di valutazione
- **pickle** - Serializzazione e salvataggio del modello


## Installazione

### 1. Clona la repository
```bash
git clone https://github.com/rebeccamagarotto/Churn-Prediction.git
cd Churn-Prediction
```

### 2. Crea un ambiente virtuale Anaconda

```bash
conda create -n churn_prediction python=3.9
conda activate churn_prediction
```

### 3. Installa le librerie

```bash
pip install pandas scikit-learn numpy
```

**Alternativa con requirements.txt** (se presente):
```bash
pip install -r requirements.txt
```

### 4. Verifica l'installazione

```bash
python -c "import pandas, numpy, sklearn; print('Installazione completata con successo!')"
```

## Utilizzo

### Pipeline di esecuzione:

#### 1. Esplora il dataset

```bash
python 01_explore_data.py
```

#### 2. Pulisci i dati

```bash
python 02_clean_data.py
```

#### 3. Converti le categorie

```bash
python 03_encode_categories.py
```

#### 4. Addestra il modello

```bash
python 04_train_model.py
```

**Output**: 
- Modello addestrato salvato in `churn_model.pkl`
- File `model_columns.pkl` con le feature utilizzate
- Metriche di performance stampate a console


## Risultati

Il modello addestrato ha raggiunto le seguenti performance sul test set:

### Metriche di Performance

- **Accuracy:** 80.4%
- **Precision (Churn):** 66.6%
- **Recall (Churn):** 52.7%
- **F1-score (Churn):** 58.8%
- **ROC–AUC:** 84%
  
**Modello**: Random Forest Classifier con 100 alberi decisionali
**Validation Strategy**: Train-test split (80-20)

### Feature Importance:

Il modello ha identificato le seguenti variabili come i principali predittori del churn:

- **Tenure** – Durata della relazione tra cliente e azienda
- **MonthlyCharges** – Costo mensile del servizio
- **TotalCharges** – Spesa cumulativa sostenuta dal cliente nel tempo
- **Contract** – Tipologia di contratto sottoscritto
- **PaymentMethod** – Metodo di pagamento utilizzato
- **TechSupport / OnlineSecurity** – Presenza di servizi di supporto e sicurezza

### Altri predittori significativi:
- Tipo di contratto (month-to-month vs annuale)
- Metodo di pagamento
- Servizi aggiuntivi sottoscritti (internet, sicurezza online, backup)

## Interpretazione dei Risultati

Il modello identifica correttamente **8 clienti su 10** che abbandoneranno il servizio, fornendo uno strumento affidabile per implementare strategie di retention preventive.

### Insights principali:

1. **Durata del contratto (Tenure)**

2. **Costi elevati**

3. **Tipo di contratto**

### Applicazioni pratiche:

- **Targeting mirato**
- **Campagne di retention**
- **Ottimizzazione pricing**
- **Customer service**


## Autori

**Rebecca Magarotto 1033459**

**Chiara Beretta 1033576**

**Chiara da Lisca 1035091**

Progetto sviluppato per il corso di **Data Mining & Text Analytics** 
nel corso di laurea magistrale in **Intelligenza Artificiale, impresa e società**

Docente: Prof. Alessandro Bruno

IULM University - A.A. 2025-2026


## Streamlit

https://churn-prediction-wptsxwkdgfk44z7cwapyfp.streamlit.app/


## Licenza

Questo progetto è distribuito sotto **licenza MIT**.

La licenza MIT permette l'uso, la copia, la modifica e la distribuzione del codice sia per scopi commerciali che non commerciali, a condizione che venga mantenuta l'attribuzione originale.

## Data

**Dicembre 2025**
