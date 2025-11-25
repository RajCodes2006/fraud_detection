import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset or Creating a synthetic one
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    N_SAMPLES = 100000
    N_FRAUD = 492
    rng = np.random.RandomState(42)

    data = {
        'Time': np.arange(N_SAMPLES),
        'V1': rng.normal(0, 1, N_SAMPLES),
        'V2': rng.normal(0, 1, N_SAMPLES),
        'V3': rng.normal(0, 1, N_SAMPLES),
        **{f'V{i}': rng.normal(0, 1, N_SAMPLES) for i in range(4, 29)},
        'Amount': np.clip(rng.lognormal(2, 1, N_SAMPLES) * 10, 0.1, 2000),
        'Class': 0
    }

    df = pd.DataFrame(data)

    # Marking fraud samples
    fraud_indices = rng.choice(N_SAMPLES, N_FRAUD, replace=False)
    df.loc[fraud_indices, 'Class'] = 1
    df.loc[fraud_indices, 'V1'] = rng.uniform(-5, -2, N_FRAUD)
    df.loc[fraud_indices, 'Amount'] = rng.uniform(200, 1000, N_FRAUD)

# Features / labels
X = df.drop('Class', axis=1)
y = df['Class']

# Scale some columns
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Outlier fraction
outlier_fraction = y.value_counts()[1] / len(df)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Isolation Forest
if_model = IsolationForest(
    n_estimators=100,
    max_samples=len(X_train),
    contamination=outlier_fraction,
    random_state=42
)
if_model.fit(X_train)
if_preds = if_model.predict(X_test)
if_preds = np.where(if_preds == 1, 0, 1)

# Local Outlier Factor
lof_model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=outlier_fraction,
    novelty=False,
    n_jobs=-1
)
lof_preds = lof_model.fit_predict(X_test)
lof_preds = np.where(lof_preds == 1, 0, 1)

# Evaluation function
def evaluate(name, y_true, y_pred):
    print(name)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title(name)
    plt.show()

# Evaluate models
evaluate("Isolation Forest", y_test, if_preds)
evaluate("Local Outlier Factor", y_test, lof_preds)
