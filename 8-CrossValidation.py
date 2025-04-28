import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score 
from sklearn.preprocessing import StandardScaler 


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
import library as GetDataFrame
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# veri setini yükle
excelPath = "Dry_Bean_ClassLabelNumeric_Final.xlsx"
dataFrame=GetDataFrame.GetExcelDataFrame(excelPath)

# Özellikler (X) ve etiketler (y)
X = dataFrame.drop("Class", axis=1)
y = dataFrame["Class"]

# scaler oluşturuyoruz
scaler = StandardScaler()
scaledObject = scaler.fit_transform(X)

# 6 bileşenle PCA oluşturuyoruz
pca = PCA(6)
PCAObject = pca.fit_transform(scaledObject)

# 3 bileşen için LDA oluşturuyoruz
lda = LDA(n_components=3)
LDAObject = lda.fit_transform(scaledObject, y)

print("Veriler işlendi.\n")

#modelleri oluşturuyoruz
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.01, 0.1, 1, 10]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth": [3, 5, 10],
            "criterion": ["gini", "entropy"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10]
        }
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(eval_metric="mlogloss"),
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1]
        }
    },
    "NativeBayes": {
        "model": GaussianNB(),
        "params": {} 
    }
}

print("Modeller hazırlandı.\n")

# Cross validation için gereken tanımlar
def run_nested_cv(X, y, model, param_grid, outer_splits=5, inner_splits=3):
    outerCrossValidation = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42) 
    innerCrossValidation = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)

    acc_scores, prec_scores, recall_scores, f1_scores = [], [], [], []

    for trainId, testId in outerCrossValidation.split(X, y):
        XTrain, XTest = X[trainId], X[testId]
        YTrain, YTest = y[trainId], y[testId]

        clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=innerCrossValidation, scoring='accuracy', n_jobs=-1)
        clf.fit(XTrain, YTrain)

        idealModel = clf.best_estimator_
        yPredict = idealModel.predict(XTest)

        acc_scores.append(accuracy_score(YTest, yPredict))
        prec_scores.append(precision_score(YTest, yPredict, average='macro', zero_division=0))
        recall_scores.append(recall_score(YTest, yPredict, average='macro', zero_division=0))
        f1_scores.append(f1_score(YTest, yPredict, average='macro', zero_division=0))

    return {
        "accuracy": (np.mean(acc_scores), np.std(acc_scores)),
        "precision": (np.mean(prec_scores), np.std(prec_scores)),
        "recall": (np.mean(recall_scores), np.std(recall_scores)),
        "f1": (np.mean(f1_scores), np.std(f1_scores)),
    }

print("Cross validation fonksiyonu.\n")

# Dictionary içine atıyoruz
dataDictionary = {
    "Pure Data": scaledObject,
    "PCA Data": PCAObject,
    "LDA Data": LDAObject
}

results = {}

for versionName, XData in dataDictionary.items():
    print(f"\n==== {versionName} ====")
    results[versionName] = {}
    
    for modelName, modelInfo in models.items():
        print(f"\nModel: {modelName}")
        metrics = run_nested_cv(XData, y.values, modelInfo["model"], modelInfo["params"])
        
        results[versionName][modelName] = metrics
        
        # Metriklerin yazdırılması
        for metricName, (mean_val, std_val) in metrics.items():
            print(f"{metricName.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")