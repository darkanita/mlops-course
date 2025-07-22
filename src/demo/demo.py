import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Configurar experimento
mlflow.set_experiment("demo-live")

with mlflow.start_run():
    # Cargar datos
    X, y = load_iris(return_X_y=True)

    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # Evaluar modelo
    accuracy = model.score(X, y)

    # Registrar en MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
