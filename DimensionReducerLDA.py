import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns
import library as GetDataFrame

# Veri setini yükle
excelPath = "Dry_Bean_ClassLabelNumeric_Final.xlsx"
dataFrame = GetDataFrame.GetExcelDataFrame(excelPath)

# veriyi Özellikler (X) ve Hedef olarak (y) olarak ayıralım
X = dataFrame.drop(columns=['Class'], axis=1)
y = dataFrame['Class']

# LDA nesnesini oluştur ve 3 bileşen seç
LDAObject = LDA(n_components=3)
XLDA = LDAObject.fit_transform(X, y)

# LDA bileşenlerini DataFrame'e ekle
datasetLDA = pd.DataFrame(data=XLDA, columns=['LDA1', 'LDA2', 'LDA3'])
datasetLDA['Class'] = y.values

# LDA bileşenlerini görselleştir (ilk iki bileşen)
plt.figure(figsize=(12, 8))
sns.scatterplot(data=datasetLDA, x='LDA1', y='LDA2', hue='Class', palette='tab10', s=50, alpha=0.7)
plt.title('LDA ile 2D Görselleştirme')
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# Yeni veri setini kaydet
datasetLDA.to_excel("Dry_Bean_LDA_Reduced.xlsx", index=False)
print("LDA ile boyut indirgeme tamamlandı. Yeni veri seti 'Dry_Bean_LDA_Reduced.xlsx' olarak kaydedildi.")