# com_bert/src/data_preparation.py

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

# Inicializar o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath):
    """
    Carrega os dados do arquivo CSV.

    Args:
        filepath (str): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame carregado.
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dados carregados de {filepath}. Quantidade de linhas: {len(df)}")
        logger.info(f"Tipos das colunas:\n{df.dtypes}")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


def clean_data(df, text_column, label_column):
    """
    Limpa os dados removendo linhas com valores nulos e assegurando rótulos corretos.

    Args:
        df (pd.DataFrame): DataFrame a ser limpo.
        text_column (str): Nome da coluna contendo os textos.
        label_column (str): Nome da coluna contendo os rótulos.

    Returns:
        pd.DataFrame: DataFrame limpo.
    """
    original_length = len(df)
    df = df.dropna(subset=[text_column, label_column])
    logger.info(f"Linhas após remover valores nulos: {len(df)}")

    # Converter a coluna de rótulo para numérico, forçando erros a NaN
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce")

    # Remover linhas onde a conversão resultou em NaN
    df = df.dropna(subset=[label_column])
    logger.info(f"Linhas após converter '{label_column}' para numérico: {len(df)}")

    # Logar valores únicos antes do filtro
    unique_values = df[label_column].unique()
    logger.info(f"Valores únicos na coluna '{label_column}': {unique_values}")

    # Garantir que os rótulos sejam apenas 0 ou 1
    df = df[df[label_column].isin([0, 1])]
    logger.info(f"Linhas após filtrar rótulos 0 e 1: {len(df)}")

    cleaned_length = len(df)
    logger.info(f"Dados limpos. Antes: {original_length}, Depois: {cleaned_length}")
    return df


def split_data(df, text_column, label_column, test_size=0.2, random_state=42):
    """
    Divide os dados em conjuntos de treinamento e validação.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados.
        text_column (str): Nome da coluna com os textos.
        label_column (str): Nome da coluna com os rótulos.
        test_size (float, optional): Proporção do conjunto de teste. Defaults to 0.2.
        random_state (int, optional): Semente para reprodutibilidade. Defaults to 42.

    Returns:
        tuple: X_train, X_val, y_train, y_val
    """
    original_length = len(df)
    X = df[text_column]
    y = df[label_column]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"Dados divididos. Treinamento: {len(X_train)}, Validação: {len(X_val)}"
    )
    return X_train, X_val, y_train, y_val
