import pandas as pd
import numpy as np
import library as GetDataFrame

excelPath = "Dry_Bean_Dataset.xlsx"
dataFrame=GetDataFrame.GetExcelDataFrame(excelPath)

# Yuzde 5 eksik veriler
MissingDataColumn5 = ('MinorAxisLength', 'AspectRation')

#Yuzde 35 eksik veriler
MissingDataColumn35 = ('MajorAxisLength')

#Yuzde 5 data oluştur
for col in MissingDataColumn5:
    # rastgele 5% eksik veri ekle
    dataFrame.loc[dataFrame.sample(frac=0.05).index, MissingDataColumn5] = np.nan

#Yuzde 35 data oluştur
for col in MissingDataColumn35:
    dataFrame.loc[dataFrame.sample(frac=0.35).index, MissingDataColumn35] = np.nan

#Eksik verilerin kontrolü
print(dataFrame.isnull().sum())

# Oluşan yeni dataseti kaydedelim
dataFrame.to_excel("Dry_Bean_WithMissingData.xlsx", index=False)




