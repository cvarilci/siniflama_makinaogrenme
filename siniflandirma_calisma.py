# Gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Veri Yükleme
# -----------------------------
df = pd.read_csv("16-diabetes.csv")

print("✅ Veri ilk 5 satır:\n", df.head(), "\n")
print("✅ Veri bilgisi:\n")
print(df.info(), "\n")
print("✅ Tanımlayıcı istatistikler:\n", df.describe(), "\n")
print("✅ Null değerler:\n", df.isnull().sum(), "\n")

# -----------------------------
# 2. Eksik değer (0 değerleri) doldurma
# -----------------------------
columns_to_fill = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)

medians = {}
for col in columns_to_fill:
    median_value = X_train[X_train[col] != 0][col].median()
    medians[col] = median_value
    X_train[col] = X_train[col].replace(0, median_value)
    X_test[col] = X_test[col].replace(0, median_value)

# -----------------------------
# 3. Standartlaştırma
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 4. Modeller ve Hiperparametreler
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=5000),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "SVM": {
        "model": SVC(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=15),
        "params": {
            "max_depth": [3, 5, 7, None],
            "criterion": ["gini", "entropy"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=15),
        "params": {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7, None],
            "criterion": ["gini", "entropy"]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=15),
        "params": {
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.001, 0.01, 0.1, 1, 10]
        }
    },
    "NaiveBayes": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": np.logspace(0,-9, num=20)
        }
    }
}

# -----------------------------
# 5. Eğitim ve Sonuçlar
# -----------------------------
results = []

for name, mp in models.items():
    print(f"\n🔹 {name} modeli deneniyor...")
    grid = GridSearchCV(mp["model"], mp["params"], cv=5, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ En iyi parametreler: {grid.best_params_}")
    print(f"✅ Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "Accuracy": acc
    })

# -----------------------------
# 6. Sonuç Karşılaştırma Tablosu
# -----------------------------
results_df = pd.DataFrame(results)
print("\n📊 Tüm modellerin karşılaştırması:\n")
print(results_df.sort_values(by="Accuracy", ascending=False))