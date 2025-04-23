from metaflow import FlowSpec, step, Parameter, kubernetes, resources, retry, timeout, catch, conda_base
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split

@conda_base(libraries={'numpy': '1.26.4', 'pandas': '2.2.2', 'scikit-learn':'1.4.2', 'mlflow': '2.12.1', 'fsspec': '2024.3.1', 'gcsfs': '2024.3.1'}, python='3.10')
class ScoringFlow(FlowSpec):

    model_name = Parameter("model_name", default="lab_7_best_model")

    @kubernetes(memory=4096, cpu="1")
    @resources(memory=4096, cpu=1)
    @retry(times=3)
    @timeout(seconds=300)
    @catch(var="start_error")
    @step
    def start(self):
        # Preprocess data
        df = pd.read_csv("gs://mxguevarra-data/1553768847-housing.csv")
        df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

        # Use hold-out data
        y = df['median_house_value']
        X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
        _, self.X_holdout, _, self.y_holdout = train_test_split(X, y, test_size=0.2, random_state=99)

        self.next(self.score)

    @kubernetes(memory=4096, cpu="1")
    @resources(memory=4096, cpu=1)
    @retry(times=2)
    @catch(var="score_error")
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
