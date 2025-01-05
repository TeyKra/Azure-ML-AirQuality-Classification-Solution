import mltable
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    Model,
    Environment,
    CodeConfiguration,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment
)
import joblib
import optuna
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    auc,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
from warnings import filterwarnings
from itertools import cycle
import time
import random
import string
import requests
import os

# Suppress warnings for cleaner output
filterwarnings('ignore')


def create_conda_yaml():
    """
    Creates a Conda YAML file for the custom environment with specified dependencies.
    
    Returns:
        str: Path to the created Conda YAML file.
    """
    conda_content = """
    name: custom-conda-env
    channels:
    - defaults
    - conda-forge
    dependencies:
    - python=3.8
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - seaborn
    - plotly
    - optuna
    - joblib
    - pip
    - pip:
        - azureml-mlflow
        - azureml-inference-server-http
        - azureml-defaults
        - mltable
        - azure-ai-ml
    """
    conda_file_path = "conda.yaml"
    with open(conda_file_path, "w") as f:
        f.write(conda_content)
    print(f"conda.yaml file created: {conda_file_path}")
    return conda_file_path


def register_environment(ml_client, conda_file_path):
    """
    Registers a custom environment with Azure ML using the provided Conda YAML file.
    
    Args:
        ml_client (MLClient): The Azure ML client.
        conda_file_path (str): Path to the Conda YAML file.
    
    Returns:
        Environment: The registered Azure ML environment.
    """
    custom_env = Environment(
        name="custom-mlflow-env",
        description="Custom environment with MLflow and scikit-learn",
        conda_file=conda_file_path,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        version="5.0"
    )
    custom_env = ml_client.environments.create_or_update(custom_env)
    print(f"Custom environment '{custom_env.name}' with version '{custom_env.version}' created successfully.")
    return custom_env


def load_data(ml_client):
    """
    Loads the Air Quality dataset from Azure ML data assets.
    
    Args:
        ml_client (MLClient): The Azure ML client.
    
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    data_asset = ml_client.data.get("Air_Quality_Dataset", version="1")
    tbl = mltable.load(f'azureml:/{data_asset.id}')
    df = tbl.to_pandas_dataframe()
    return df


def preprocess_data(df):
    """
    Preprocesses the dataset by encoding the target variable.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target variables.
    """
    label_encoder = LabelEncoder()
    df['Air Quality'] = label_encoder.fit_transform(df['Air Quality'])
    print("Classes:", label_encoder.classes_)
    X = df.drop(columns='Air Quality')
    y = df['Air Quality']
    return X, y


def split_data(X, y):
    """
    Splits the dataset into training and testing sets.
    
    Args:
        X (pd.DataFrame): Feature variables.
        y (pd.Series): Target variable.
    
    Returns:
        Tuple: Training and testing splits for features and target.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial, X_train, y_train):
    """
    Objective function for Optuna hyperparameter tuning.
    
    Args:
        trial (optuna.trial.Trial): The trial object.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        float: Mean cross-validated accuracy score.
    """
    model_name = trial.suggest_categorical(
        'model', 
        ['Logistic Regression', 'K-Nearest Neighbors', 'SVC', 'Random Forest', 'Gradient Boosting']
    )

    if model_name == 'Logistic Regression':
        C = trial.suggest_float('C', 0.01, 10.0)
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    elif model_name == 'K-Nearest Neighbors':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_name == 'SVC':
        C = trial.suggest_float('C', 0.01, 10.0)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    elif model_name == 'Random Forest':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_name == 'Gradient Boosting':
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)

    # Perform cross-validation and return the mean accuracy
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score


def hyperparameter_tuning(X_train, y_train):
    """
    Performs hyperparameter tuning using Optuna to find the best model parameters.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
    print(f"Best model parameters: {study.best_params}")
    return study.best_params


def train_and_evaluate_model(best_params, X_train, y_train, X_test, y_test):
    """
    Trains the final model with the best hyperparameters and evaluates its performance.
    
    Args:
        best_params (dict): Best hyperparameters.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
    
    Returns:
        sklearn.base.BaseEstimator: The trained final model.
    """
    # Mapping of model names to their classes
    model_mapping = {
        'Logistic Regression': LogisticRegression,
        'K-Nearest Neighbors': KNeighborsClassifier,
        'SVC': SVC,
        'Random Forest': RandomForestClassifier,
        'Gradient Boosting': GradientBoostingClassifier
    }

    ModelClass = model_mapping[best_params['model']]
    # Extract model parameters excluding the 'model' key
    model_args = {k: v for k, v in best_params.items() if k != 'model'}

    # Binarize labels for ROC curves
    y_train_binarized = label_binarize(y_train, classes=np.unique(y_train))
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_train_binarized.shape[1]

    with mlflow.start_run():
        # Initialize and train the final model
        final_model = ModelClass(**model_args, random_state=42)
        final_model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = final_model.predict(X_train)
        y_test_pred = final_model.predict(X_test)
        y_train_proba = final_model.predict_proba(X_train)
        y_test_proba = final_model.predict_proba(X_test)

        # Calculate training metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1_score = f1_score(y_train, y_train_pred, average='weighted')
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        train_log_loss = log_loss(y_train, y_train_proba)
        train_auc = roc_auc_score(y_train_binarized, y_train_proba, average='weighted', multi_class='ovr')

        # Calculate testing metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1_score = f1_score(y_test, y_test_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_log_loss = log_loss(y_test, y_test_proba)
        test_auc = roc_auc_score(y_test_binarized, y_test_proba, average='weighted', multi_class='ovr')

        # Display classification reports
        train_class_report = classification_report(
            y_train, 
            y_train_pred, 
            target_names=[f"Class {c}" for c in np.unique(y_train)]
        )
        test_class_report = classification_report(
            y_test, 
            y_test_pred, 
            target_names=[f"Class {c}" for c in np.unique(y_test)]
        )

        print("Training Metrics by Class:")
        print(train_class_report)
        print("Testing Metrics by Class:")
        print(test_class_report)

        # Plot ROC curves for training data
        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_train_binarized[:, i], y_train_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Train ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve_train.png")
        mlflow.log_artifact("roc_curve_train.png")
        plt.close()

        # Plot ROC curves for testing data
        plt.figure()
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_test_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve_test.png")
        mlflow.log_artifact("roc_curve_test.png")
        plt.close()

        # Plot confusion matrix for testing data
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix for Final Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig("confusion_matrix_test.png")
        mlflow.log_artifact("confusion_matrix_test.png")
        plt.close()

        # Log global metrics to MLflow
        # Training metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_f1_score", train_f1_score)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_log_loss", train_log_loss)
        mlflow.log_metric("train_auc", train_auc)

        # Testing metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_score", test_f1_score)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_log_loss", test_log_loss)
        mlflow.log_metric("test_auc", test_auc)

        # Log the trained model manually
        mlflow.sklearn.log_model(final_model, "model")

    return final_model

def save_model(final_model, ml_client, best_params):
    """
    Saves the trained model using MLflow and registers it with Azure ML.
    
    Args:
        final_model (sklearn.base.BaseEstimator): The trained model.
        ml_client (MLClient): The Azure ML client.
        best_params (dict): Best hyperparameters.
    
    Returns:
        Model: The registered Azure ML model.
    """
    mlflow_model_dir = "outputs/mlflow_model"
    if not os.path.exists(mlflow_model_dir):
        mlflow.sklearn.save_model(
            sk_model=final_model,
            path=mlflow_model_dir
        )
        print(f"Model saved at {mlflow_model_dir}")

    # Create a model name by formatting the best model's name
    model_name = best_params['model'].replace(" ", "_").replace("-", "_").lower()
    registered_model = Model(
        name=model_name,
        path=mlflow_model_dir,
        description="Best model from Optuna (MLflow format)",
        type="mlflow_model"
    )
    registered_model = ml_client.models.create_or_update(registered_model)
    print(f"Model registered with name={registered_model.name}, version={registered_model.version}")

    return registered_model


def enable_application_insights(ml_client, endpoint_name, deployment_name):
    """
    Enables Application Insights for the specified deployment.
    
    Args:
        ml_client (MLClient): The Azure ML client.
        endpoint_name (str): Name of the online endpoint.
        deployment_name (str): Name of the deployment.
    """
    deployment = ml_client.online_deployments.get(endpoint_name=endpoint_name, name=deployment_name)
    deployment.app_insights_enabled = True
    updated_deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Application Insights enabled for deployment '{deployment_name}'.")


def create_endpoint(ml_client, registered_model):
    """
    Creates an online endpoint and deploys the registered model to it.
    
    Args:
        ml_client (MLClient): The Azure ML client.
        registered_model (Model): The registered Azure ML model.
    
    Returns:
        str: Name of the created endpoint.
    """
    # Generate a random suffix for the endpoint name to ensure uniqueness
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    endpoint_name = f"air-quality-endpoint-{random_suffix}"
    print("Endpoint name:", endpoint_name)

    # Create the managed online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key"
    )
    ml_client.begin_create_or_update(endpoint).result()

    # Configure the deployment with the registered model and environment
    code_config = CodeConfiguration(
        code=".",
        scoring_script="score.py"
    )

    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=registered_model,
        environment="custom-mlflow-env:1",
        code_configuration=code_config,
        instance_type="Standard_F2s_v2",
        instance_count=1
    )

    # Deploy the model
    ml_client.begin_create_or_update(deployment).result()
    print(f"Deployment created: {endpoint_name}")

    # Route 100% traffic to the "blue" deployment
    endpoint.traffic = {"blue": 100}
    endpoint_updated = ml_client.begin_create_or_update(endpoint).result()
    print(f"Traffic updated: {endpoint_updated.traffic}")

    # Enable Application Insights for monitoring
    enable_application_insights(ml_client, endpoint_name, "blue")

    return endpoint_name


def main():
    """
    Main function to orchestrate the machine learning workflow:
    1. Initialize MLClient
    2. Create and register environment
    3. Load and preprocess data
    4. Perform hyperparameter tuning
    5. Train and evaluate the final model
    6. Save and register the model
    7. Deploy the model to an online endpoint
    """
    # Initialize MLClient with default Azure credentials
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    # Step 1: Create Conda environment
    conda_file_path = create_conda_yaml()
    custom_env = register_environment(ml_client, conda_file_path)

    # Step 2: Load and preprocess data
    df = load_data(ml_client)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Hyperparameter tuning with Optuna
    best_params = hyperparameter_tuning(X_train, y_train)

    # Step 4: Train and evaluate the final model
    final_model = train_and_evaluate_model(best_params, X_train, y_train, X_test, y_test)

    # Step 5: Save and register the trained model
    registered_model = save_model(final_model, ml_client, best_params)

    # Step 6: Deploy the model to an online endpoint
    endpoint_name = create_endpoint(ml_client, registered_model)
    print(f"Deployment completed at endpoint: {endpoint_name}")


if __name__ == "__main__":
    main()
