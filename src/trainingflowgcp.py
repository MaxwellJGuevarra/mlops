from metaflow import FlowSpec, step, Parameter, kubernetes, resources, retry, timeout, catch, conda_base
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
import os

@conda_base(libraries={'numpy': '1.26.4', 'pandas': '2.2.2', 'scikit-learn':'1.4.2', 'mlflow': '2.12.1', 'fsspec': '2024.3.1', 'gcsfs': '2024.3.1'}, python='3.10')
class TrainingFlow(FlowSpec):

    seed = Parameter("seed", default=23)
    n_estimators = Parameter("n_estimators", default=50)
    max_features = Parameter("max_features", default=3)

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

        y = df['median_house_value']
        X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=0.8, random_state=self.seed
        )

        self.next(self.train_model)

    @kubernetes(memory=4096, cpu="2")
    @resources(memory=4096, cpu=2)
    @retry(times=2)
    @catch(var="train_error")
    @step
    def train_model(self):

        # Set local MLflow tracking URI
        mlflow.set_tracking_uri("https://service-1-73941950916.us-west2.run.app")
        mlflow.set_experiment("lab-7")

        with mlflow.start_run() as run:
            mlflow.set_tags({
                "model_type": "random_forest",
                "framework": "sklearn",
                "source": "metaflow"
            })

            # Train the model
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                random_state=self.seed
            )
            rf.fit(self.X_train, self.y_train)

            # Evaluate and log metrics
            preds = rf.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, preds))

            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("max_features", self.max_features)
            mlflow.log_metric("rmse", rmse)

            # Log model artifact
            artifact_subdir = "better_models"
            mlflow.sklearn.log_model(rf, artifact_path=artifact_subdir)

            # Save run and artifact info to pass to next step
            self.run_id = run.info.run_id
            self.artifact_subdir = artifact_subdir
            self.model_uri = f"runs:/{self.run_id}/{artifact_subdir}"
            self.rmse = rmse

        self.next(self.register_model)

    @retry(times=2)
    @catch(var="register_error")
    @step
    def register_model(self):
        mlflow.set_tracking_uri("https://service-1-73941950916.us-west2.run.app")

        # Register model from URI
        result = mlflow.register_model(
            model_uri=self.model_uri,
            name="lab_7_best_model"
        )

        print(f"Model registered: {result.name}; version: {result.version}")
        self.next(self.end)

    @step
    def end(self):
        print("Training flow complete.")

if __name__ == "__main__":
    TrainingFlow()
