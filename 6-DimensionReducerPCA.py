import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import library as GetDataFrame

#Bu yöntem, verideki özellikler için yeni bileşenler oluşturur. 
#Boyut indirgeme, özellikle yüksek boyutlu veri setlerinde 
#işlem maliyetini azaltmak ve görselleştirme için faydalıdır.


# PCA uygulamak için final veri setini yükle
excelPath = "Dry_Bean_ClassLabelNumeric_Final.xlsx"
dataFrame=GetDataFrame.GetExcelDataFrame(excelPath)

# veriyi Özellikler (X) ve Hedef olarak (y) olarak ayıralım
X = dataFrame.drop(columns=['Class'], axis=1)
y = dataFrame['Class']

# PCA nesnesini oluşturalım
pca = PCA()
PCAObject = pca.fit_transform(X)

# Açıklanan attributları ve varyans oranlarını göster
currentVariant = pca.explained_variance_ratio_
print("\nPCA bileşenlerinin varyantlar:", currentVariant)

# Toplam açıklanan varyansı göster
cumulativeVariant = currentVariant.cumsum()
print("\nToplam varyans oranları:", cumulativeVariant)

# Açıklanan varyans oranlarını grafik olarak göster
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(currentVariant) + 1), cumulativeVariant, marker='o', linestyle='--')
plt.title('PCA Mevcut Özellikler Oranı')
plt.xlabel('Bileşenler')
plt.ylabel('PCA Tüm Özellikler Oranı')
plt.grid()
plt.show()

# PCA nesnesini 2 bileşen ile oluştur
PCA2 = PCA(n_components=2)
PCAOBJECT2 = PCA2.fit_transform(X)

# PCA ile elde edilen bileşenleri DataFrame'e ekle
dataFramePCA = pd.DataFrame(data=PCAOBJECT2, columns=['PCA1', 'PCA2'])
dataFramePCA['Class'] = y.values

# Scatter plot ile PCA bileşenlerini görselleştir
plt.figure(figsize=(12, 8))
sns.scatterplot(data=dataFramePCA, x='PCA1', y='PCA2', hue='Class', palette='tab10', s=50, alpha=0.7)
plt.title('PCA ile 2D Görselleştirme')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# Yeni veri setini kaydet
dataFramePCA.to_excel("Dry_Bean_PCA_Reduced.xlsx", index=False)
print("PCA ile boyut indirgeme tamamlandı. Yeni veri seti 'Dry_Bean_PCA_Reduced.xlsx' olarak kaydedildi.")
