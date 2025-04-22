from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import numpy as np
import os

class TrainingFlow(FlowSpec):

    seed = Parameter("seed", default=23)
    n_estimators = Parameter("n_estimators", default=50)
    max_features = Parameter("max_features", default=3)

    @step
    def start(self):
        # Preprocess data
        df = pd.read_csv('~/mlops/data/1553768847-housing.csv')
        df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

        y = df['median_house_value']
        X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=0.8, random_state=self.seed
        )

        self.next(self.train_model)

    @step
    def train_model(self):

        # Set local MLflow tracking URI
        mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
        mlflow.set_experiment("lab-6")

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

    @step
    def register_model(self):
        mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

        # Register model from URI
        result = mlflow.register_model(
            model_uri=self.model_uri,
            name="lab_6_best_model"
        )

        print(f"Model registered: {result.name}; version: {result.version}")
        self.next(self.end)

    @step
    def end(self):
        print("Training flow complete.")

if __name__ == "__main__":
    TrainingFlow()
