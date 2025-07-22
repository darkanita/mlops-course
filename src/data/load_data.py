import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from typing import Tuple
import os

def load_iris_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Cargar y retornar dataset iris como objetos pandas."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y

def load_wine_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Cargar y retornar dataset wine como objetos pandas."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name='target')
    return X, y

def get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Dividir datos en conjuntos de entrenamiento y prueba."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def save_data(X_train, X_test, y_train, y_test, data_dir: str = "data"):
    """Guardar divisiones de entrenamiento/prueba en archivos CSV."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Guardar datos de entrenamiento
    train_data = X_train.copy()
    train_data['target'] = y_train
    train_data.to_csv(f"{data_dir}/train.csv", index=False)
    
    # Guardar datos de prueba
    test_data = X_test.copy()
    test_data['target'] = y_test
    test_data.to_csv(f"{data_dir}/test.csv", index=False)
    
    print(f"Datos guardados en {data_dir}/")
    print(f"Forma train: {train_data.shape}")
    print(f"Forma test: {test_data.shape}")

if __name__ == "__main__":
    # Cargar datos
    X, y = load_iris_data()
    
    # Dividir datos
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    # Guardar datos
    save_data(X_train, X_test, y_train, y_test)