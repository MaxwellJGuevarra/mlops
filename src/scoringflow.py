from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

class ScoringFlow(FlowSpec):

    model_name = Parameter("model_name", default="lab_6_best_model")

    @step
    def start(self):
        # Preprocess data
        df = pd.read_csv('~/mlops/data/1553768847-housing.csv')
        df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

        # Use hold-out data
        y = df['median_house_value']
        X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
        _, self.X_holdout, _, self.y_holdout = train_test_split(X, y, test_size=0.2, random_state=99)

        self.next(self.score)

    @step
    def score(self):
        # Load latest model from MLflow
        model = mlflow.sklearn.load_model(f"models:/{self.model_name}/latest")

        predictions = model.predict(self.X_holdout)
        mse = mean_squared_error(self.y_holdout, predictions)
        rmse = np.sqrt(mse)
        print(f"Scoring RMSE: {rmse:.2f}")
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow complete.")

if __name__ == '__main__':
    ScoringFlow()
