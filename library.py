import pandas as pd

def GetExcelDataFrame(excelPath):
    """
    Excel dosyasını pandas dataframe olarak yükler.
    :param excelPath: Excel dosyasının yolu
    :return: DataFrame
    """
    dataSet = pd.read_excel(excelPath)
    dataFrame = pd.DataFrame(dataSet)
    return dataFrame

