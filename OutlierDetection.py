import pandas as pd
import library as GetDataFrame
from scipy.stats import zscore
import numpy as np

excelPath = "Dry_Bean_WithFilledValues.xlsx"
dataFrame=GetDataFrame.GetExcelDataFrame(excelPath)

# sayısal verileri al (label hariç)
numberColumns = dataFrame.select_dtypes(include=['float64', 'int64']).columns

# Z-score yöntemi ile tespit edilecek
threshold = 3  # Z-score eşik değeri genellikle 3 olarak alınır
for col in numberColumns:
    # 1. çeyrek ve 3. çeyrek verilerini al
    z_scores = zscore(dataFrame[col])
    outlierIndexes = np.abs(z_scores) > threshold
    medianValue = dataFrame[col].median()

    # OQR ile üst sınır hesapla
    Q1 = dataFrame[col].quantile(0.25) 
    Q3 = dataFrame[col].quantile(0.75) 
    IQR = Q3 - Q1
    upperLimit = Q3 + 1.5 * IQR # Üst sınır
    lowerLimit=  Q1 - 1.5 * IQR # Alt sınır

    # Alt sınırdan küçük değerleri alt sınırla değiştir
    dataFrame.loc[dataFrame[col] < lowerLimit, col] = lowerLimit
    
    # Üst sınırdan büyük değerleri üst sınırla değiştir
    dataFrame.loc[dataFrame[col] > upperLimit, col] = upperLimit

    #dataFrame.loc[outlierIndexes, col] = upperLimit
    print(f"{col} sütununda bulunan aykırı değer sayısı: {outlierIndexes.sum()}")
    
# yeni veri setini kaydet
dataFrame.to_excel("Dry_Bean_HandleOutlierDetection.xlsx", index=False)
    
