import pandas as pd
from sklearn.preprocessing import StandardScaler
import library as GetDataFrame

#Özelliklerin farklı ölçek birimlerine sahip olması nedeniyle bazılarının baskın olmasına sebep olabilir.
# Bu nedenle özelliklerin ölçeklendirilmesi gerekmektedir.
# Ölçeklendirme işlemi, verilerin belirli bir aralığa (genellikle 0 ile 1 arasında) dönüştürülmesini ve aynı ölçeğe gelmesini sağlar.

#son veriyi alalım
excelPath = "Dry_Bean_HandleOutlierDetection.xlsx"
dataFrame=GetDataFrame.GetExcelDataFrame(excelPath)

# sayısal veriler
numberColumns = dataFrame.select_dtypes(include=['float64', 'int64']).columns

# label sınıf etiketi kolonunun adını al
TypeColumn = 'Class' # fasulyenin tipi
# numerik kolonlardan silmek istediğimiz ölçeklemeye dahil etmeyeceğimiz bir kolon olsaydı silecektik.
# Örn: tür bilgisi sayısal bir ifade olsaydı

# ölçekleyici
scaler = StandardScaler()

# sayısal verileri ölçeklendir
dataFrame[numberColumns] = scaler.fit_transform(dataFrame[numberColumns])

# ölçeklendirilmiş veriyi kaydet
dataFrame.to_excel("Dry_Bean_Dataset_ScaledFeature.xlsx", index=False)

print("Veri ölçeklendirme işlemi tamamlandı. Yeni veri 'Dry_Bean_Dataset_ScaledFeature.xlsx' içerisine kaydedildi.")
