import numpy as np
from sklearn import preprocessing


class FeatureExtractor():
    def __init__(self):
        self.le = preprocessing.LabelEncoder()
        self.enc = preprocessing.OneHotEncoder()

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()
        for i in range(90,171):
            col = "attr" + str(i)
            X_df_new = X_df_new.drop([col], axis=1)
        X_df_new = self.compute_mean(X_df_new, 18, 58)
        return X_df_new


    def compute_mean(self, data, begin, end):
        name = "mean"+str(begin)+"and"+str(end)
        sum_data = 0
        for index in range(begin,end+1):
            sum_data += data[data.columns[index]]
        data[name] = sum_data/(end-begin)
        return data