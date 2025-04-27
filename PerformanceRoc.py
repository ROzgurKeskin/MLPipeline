import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

# Modeli eğitelim
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(trainX, trainYBin)

# olasılıkları alalım
scoreY = clf.predict_proba(testX)

predictA = dict()
predictB = dict()
aucRoc = dict()

for i in range(n_classes):
    predictA[i], predictB[i], _ = roc_curve(testYBin[:, i], scoreY[i][:, 1])
    aucRoc[i] = auc(predictA[i], predictB[i])

# Çizimlere başlayalım
plt.figure(figsize=(10, 7))
classLabels = np.unique(y)
for i in range(n_classes):
    plt.plot(predictA[i], predictB[i], label=f"Class {classLabels[i]} (AUC = {aucRoc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.title("ROC Çizimi - Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_drawing_random_forest.pdf", format='pdf')
print("Roc çizimi kaydedildi")
plt.show()