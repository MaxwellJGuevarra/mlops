import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def impute_median(df):
    bedrooms_median = df['total_bedrooms'].median()
    df['total_bedrooms'] = df['total_bedrooms'].fillna(bedrooms_median)
    return df


if __name__=="__main__":

    df = pd.read_csv('data/1553768847-housing.csv')
    df = impute_median(df)
    
    y = df['median_house_value']
    X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    X_train['y'] = y_train
    X_test['y'] = y_test

    X_train.to_csv('data/preprocessed_training_house.csv')
    X_test.to_csv('data/preprocessed_test_house.csv')
