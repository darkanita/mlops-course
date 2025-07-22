import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import config

def load_data():
    """Cargar datos de entrenamiento y prueba."""
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    return X_train, X_test, y_train, y_test

def train_model(model_type: str = "random_forest", **model_params):
    """Entrenar un modelo con seguimiento MLFlow."""
    
    # Configurar URI de seguimiento MLFlow y experimento
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        # Registrar información de la ejecución
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("developer", "estudiante")
        mlflow.set_tag("purpose", "entrenamiento")
        
        # Cargar datos
        X_train, X_test, y_train, y_test = load_data()
        
        # Registrar información de datos
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", X_train.shape[1])
        mlflow.log_param("classes", len(np.unique(y_train)))
        
        # Crear modelo
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**model_params)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")
        
        # Registrar parámetros del modelo
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calcular métricas
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        # Registrar métricas
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        
        # Crear y guardar gráficos
        create_confusion_matrix_plot(y_test, y_pred_test)
        create_feature_importance_plot(model, X_train.columns)
        
        # Registrar modelo
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"iris-{model_type}"
        )
        
        # Imprimir resultados
        print(f"Run ID: {run.info.run_id}")
        print(f"Modelo: {model_type}")
        print(f"Precisión Test: {test_accuracy:.4f}")
        print(f"MLFlow UI: {config.MLFLOW_TRACKING_URI}")
        
        return model, run.info.run_id

def create_confusion_matrix_plot(y_true, y_pred):
    """Crear y guardar gráfico de matriz de confusión."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png", "plots")
    plt.close()

def create_feature_importance_plot(model, feature_names):
    """Crear y guardar gráfico de importancia de características."""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Importancia de Características')
        plt.xlabel('Importancia')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png", "plots")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="random_forest", choices=["random_forest", "logistic_regression"])
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    
    args = parser.parse_args()
    
    model_params = {
        "random_state": args.random_state
    }
    
    if args.model_type == "random_forest":
        model_params.update({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth
        })
    
    model, run_id = train_model(args.model_type, **model_params)
    print(f"¡Entrenamiento completado! Verificar MLFlow UI: {config.MLFLOW_TRACKING_URI}")