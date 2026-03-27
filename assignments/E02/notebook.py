# notebook.py - Extraído automaticamente de E02_Ensemble_Learning.ipynb

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data(seed=42):
    """
    Carrega o dataset Fashion MNIST, converte os rótulos e separa em treino/teste.

    Args:
        seed (int): Semente para garantir a reprodutibilidade da divisão.

    Returns:
        X_train, X_test, y_train, y_test: Arrays contendo os dados e rótulos.
    """
    print("Carregando Fashion MNIST... Isso pode levar alguns segundos.")

    X, y = fetch_openml('Fashion-MNIST', version=1, as_frame=False, return_X_y=True, parser='auto')

    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    print(f"Dados carregados com sucesso!")
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

    return X_train, X_test, y_train, y_test


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def train_random_forest(X_train, y_train, seed=42):
    """
    Treina um classificador Random Forest.
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

    print("Treinando Random Forest...")
    rf_model.fit(X_train, y_train)

    return rf_model

def train_adaboost(X_train, y_train, seed=42):
    """
    Treina um classificador AdaBoost utilizando Decision Trees como base.
    """
    ada_model = AdaBoostClassifier(n_estimators=50, random_state=seed)

    print("Treinando AdaBoost...")
    ada_model.fit(X_train, y_train)

    return ada_model


from sklearn.metrics import accuracy_score

def evaluate(model, X_test, y_test):
    """
    Realiza predições em um conjunto de teste e retorna a acurácia.

    Args:
        model: O modelo treinado (Random Forest, AdaBoost, etc.).
        X_test: Dados de teste.
        y_test: Rótulos reais de teste.

    Returns:
        float: O valor da acurácia (0.0 a 1.0).
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    model_name = model.__class__.__name__
    print(f"Acurácia [{model_name}]: {acc:.4f}")

    return acc


def run_pipeline(model_type="rf", seed=42):
    """
    Orquestra o carregamento, pré-processamento, treinamento e avaliação.

    Args:
        model_type (str): "rf" para Random Forest ou "ab" para AdaBoost.
        seed (int): Semente de aleatoriedade para garantir reprodutibilidade.

    Returns:
        float: Acurácia final obtida no conjunto de teste.
    """
    X_train, X_test, y_train, y_test = load_data(seed=seed)

    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0

    if model_type == "rf":
        model = train_random_forest(X_train_norm, y_train, seed=seed)
    elif model_type == "ab":
        model = train_adaboost(X_train_norm, y_train, seed=seed)
    else:
        raise ValueError("model_type deve ser 'rf' (Random Forest) ou 'ab' (AdaBoost)")

    accuracy = evaluate(model, X_test_norm, y_test)

    return accuracy