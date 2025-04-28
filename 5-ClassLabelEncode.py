import pandas as pd
from sklearn.preprocessing import LabelEncoder
import library as GetDataFrame

#Class kolonunun değerlerini sayısal değerlere dönüştürelim

excelPath = "Dry_Bean_Dataset_ScaledFeature.xlsx"
dataFrame=GetDataFrame.GetExcelDataFrame(excelPath)

# Sayısallaştırma öncesi durum
print("Mevcut Türler:")
print(dataFrame['Class'].value_counts())
print("\n")

#Sıralama önemli olmasaydı OneHotEncoding ile de yapılabilirdi.
labelEncoder = LabelEncoder()

# class kolonunu sayısal değerlere dönüştür
dataFrame['Class'] = labelEncoder.fit_transform(dataFrame['Class'])


# işlem sonrası durum   
print("Yeni kategoriler:")
print(dataFrame['Class'].value_counts())
print("\n")


# label encoded veriyi kaydet
dataFrame.to_excel("Dry_Bean_ClassLabelNumeric_Final.xlsx", index=False)

print("türlerin sayısal veriye dönüştürülmesi tamamlandı.")