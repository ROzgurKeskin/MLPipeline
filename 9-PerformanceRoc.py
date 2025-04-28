import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_curve, auc
import library as GetDataFrame

# Exceli okuyalım
excelPath = "Dry_Bean_ClassLabelNumeric_Final.xlsx"
dataFrame=GetDataFrame.GetExcelDataFrame(excelPath)

X = dataFrame.drop("Class", axis=1)
y = dataFrame["Class"]

# Etiketleri sayısallaştır (binarize)

yBinarize = label_binarize(y, classes=np.unique(y))
n_classes = yBinarize.shape[1]

# Datayı ölçeklendirelim
scaledObject = StandardScaler().fit_transform(X)
trainX, testX, trainYBin, testYBin = train_test_split(scaledObject, yBinarize, test_size=0.3, random_state=42, stratify=yBinarize)

# Modelleri tanımlayalım
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "NaiveBayes": GaussianNB(),
    "XGBoost": xgb.XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=42),
}

# Her model için ROC eğrisi çizimi
for model_name, model in models.items():
    # Modeli eğit
    if model_name in ["LogisticRegression", "DecisionTree", "RandomForest","NaiveBayes"]:
        # Bu modeller için y'nin 1D olması gerekiyor
        model.fit(trainX, np.argmax(trainYBin, axis=1))
        scoreY = model.predict_proba(testX)
    else:
        # Diğer modeller için y'nin binarize edilmiş hali kullanılabilir
        model.fit(trainX, trainYBin)
        scoreY = model.predict_proba(testX)

    # ROC eğrisi ve AUC hesaplama
    predictA = dict()
    predictB = dict()
    aucRoc = dict()

    for i in range(n_classes):
        predictA[i], predictB[i], _ = roc_curve(testYBin[:, i], scoreY[:, i])
        aucRoc[i] = auc(predictA[i], predictB[i])

    # ROC eğrisi çizimi
    plt.figure(figsize=(10, 7))
    classLabels = np.unique(y)
    for i in range(n_classes):
        plt.plot(
            predictA[i],
            predictB[i],
            label=f"Class {classLabels[i]} (AUC = {aucRoc[i]:.2f})",
        )
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.title(f"ROC Çizimi - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"roc_drawing_{model_name.lower()}.pdf", format="pdf")
    print(f"ROC çizimi kaydedildi: roc_drawing_{model_name.lower()}.pdf")
    plt.show()