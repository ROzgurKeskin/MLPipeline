import pandas as pd
import library as GetDataFrame

#Eksik dataların olduğu dosyayı db olarak yükleyelim
excelPath = "Dry_Bean_WithMissingData.xlsx"
dataFrame=GetDataFrame.GetExcelDataFrame(excelPath)

#Eksikleri bulalım
missingDataset=dataFrame.isnull().sum()
missingDataset.sort_values(ascending=False)
print(dataFrame.isnull().sum())

totalRows = len(dataFrame)
deletedDataColumn = missingDataset[missingDataset > (totalRows * 0.35)].index
medianDataColumn = missingDataset[((missingDataset > totalRows * 0.05) & (missingDataset < (totalRows * 0.35)))].index

print (f"Yüzde 35 eksik kolonlar: {deletedDataColumn}")
print (f"Yüzde 5 eksik kolonlar: {medianDataColumn}")

#Yüzde 5 eksik kolonları ortalama ile doldur
for col in medianDataColumn:
    medianValue = dataFrame[col].median()
    dataFrame[col] = dataFrame[col].fillna(medianValue)

#yuzde 35 eksik kolonları sil
dataFrame.drop(deletedDataColumn, axis=1, inplace=True)

# eksik verileri birkez daha kontrol et
print(dataFrame.isnull().sum())

# son veri setini kaydet
dataFrame.to_excel("Dry_Bean_WithFilledValues.xlsx", index=False)

print("Eksik veriler dolduruldu ve yeni veri seti kaydedildi: Dry_Bean_WithFilledValues.xlsx")