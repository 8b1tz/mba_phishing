# com_bert/scripts/train_bert.py

import csv  # Import necessário para parâmetros de leitura do CSV
import logging
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.bert_model import preprocess_bert_data, save_bert_model, train_bert_model
from transformers import BertTokenizer

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Função principal para treinar o modelo BERT utilizando dados do CSV.
    """
    print("Iniciando o treinamento do modelo BERT com dados CSV.")

    # Diretório atual do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Diretório do script: {script_dir}")

    # Caminho para o arquivo CSV
    csv_path = os.path.join(
        script_dir, "data", "unified_phishing_with_realistic_urls.csv"
    )
    if not os.path.isfile(csv_path):
        logger.error(f"Arquivo CSV não encontrado: {csv_path}")
        print(f"Arquivo CSV não encontrado: {csv_path}")
        sys.exit(1)
    else:
        logger.info(f"Arquivo CSV encontrado: {csv_path}")
        print(f"Arquivo CSV encontrado: {csv_path}")

    # Carregar os dados do CSV com parâmetros adequados
    try:
        # Primeiro, tente ler com cabeçalho
        df = pd.read_csv(
            csv_path,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            engine="python",
            on_bad_lines="warn",
        )
        # Verificar se as colunas corretas foram lidas
        if set(["is_phishing", "text", "url"]).issubset(df.columns):
            logger.info("Cabeçalho do CSV identificado corretamente.")
        else:
            # Se as colunas não estiverem corretas, tente ler sem cabeçalho
            logger.warning(
                "Cabeçalho do CSV não identificado corretamente. Tentando ler sem cabeçalho."
            )
            df = pd.read_csv(
                csv_path,
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                engine="python",
                header=None,  # Indica que o CSV não tem cabeçalho
                names=["is_phishing", "text", "url"],  # Nomes das colunas
                on_bad_lines="warn",
            )
            logger.info("CSV lido sem cabeçalho e colunas nomeadas manualmente.")
    except Exception as e:
        logger.error(f"Erro ao ler o arquivo CSV: {e}")
        print(f"Erro ao ler o arquivo CSV: {e}")
        sys.exit(1)

    logger.info(f"Dados carregados com sucesso. Número de registros: {len(df)}")
    print(f"Dados carregados com sucesso. Número de registros: {len(df)}")

    # Verificar as colunas necessárias
    required_columns = {"is_phishing", "text", "url"}
    if not required_columns.issubset(df.columns):
        logger.error(f"O CSV deve conter as colunas: {required_columns}")
        print(f"O CSV deve conter as colunas: {required_columns}")
        sys.exit(1)

    # Converter a coluna 'is_phishing' para inteiro, se necessário
    if df["is_phishing"].dtype != int:
        try:
            # Verificar se há entradas não numéricas na coluna
            non_numeric = (
                df["is_phishing"].apply(lambda x: not isinstance(x, (int, float))).sum()
            )
            if non_numeric > 0:
                logger.warning(
                    f"A coluna 'is_phishing' contém {non_numeric} entradas não numéricas. Tentando converter."
                )
                # Tentar converter, forçando entradas inválidas a NaN
                df["is_phishing"] = pd.to_numeric(df["is_phishing"], errors="coerce")
                # Remover linhas onde a conversão falhou (NaN)
                df = df.dropna(subset=["is_phishing"])
                # Converter para inteiro
                df["is_phishing"] = df["is_phishing"].astype(int)
                # Verificar novamente
                non_numeric_after = df["is_phishing"].isna().sum()
                if non_numeric_after > 0:
                    logger.error(
                        f"A coluna 'is_phishing' contém {non_numeric_after} entradas não numéricas após tentativa de conversão."
                    )
                    print(
                        f"A coluna 'is_phishing' contém {non_numeric_after} entradas não numéricas após tentativa de conversão."
                    )
                    sys.exit(1)
            else:
                df["is_phishing"] = df["is_phishing"].astype(int)
            logger.info("Coluna 'is_phishing' convertida para inteiro.")
        except Exception as e:
            logger.error(f"Erro ao converter a coluna 'is_phishing' para inteiro: {e}")
            print(f"Erro ao converter a coluna 'is_phishing' para inteiro: {e}")
            sys.exit(1)

    # Verificar valores únicos na coluna 'is_phishing'
    unique_values = df["is_phishing"].unique()
    print("\nValores Únicos na Coluna 'is_phishing':")
    print(unique_values)
    logger.info(f"Valores Únicos na Coluna 'is_phishing': {unique_values}")

    # Verificar a distribuição das classes
    class_counts = df["is_phishing"].value_counts()
    print("\nDistribuição das Classes Antes do Balanceamento:")
    print(class_counts)
    logger.info(f"Distribuição das Classes Antes do Balanceamento:\n{class_counts}")

    # Filtrar apenas linhas onde 'is_phishing' é 0 ou 1
    df = df[df["is_phishing"].isin([0, 1])]
    class_counts = df["is_phishing"].value_counts()
    print("\nDistribuição das Classes Após Filtragem (após remover valores inválidos):")
    print(class_counts)
    logger.info(
        f"Distribuição das Classes Após Filtragem (após remover valores inválidos):\n{class_counts}"
    )

    # Identificar classes com pelo menos 2 exemplos
    classes_to_keep = class_counts[class_counts >= 2].index
    df_filtered = df[df["is_phishing"].isin(classes_to_keep)]

    # Verificar a nova distribuição
    new_class_counts = df_filtered["is_phishing"].value_counts()
    print("\nDistribuição das Classes Após Filtragem (classes com >= 2 exemplos):")
    print(new_class_counts)
    logger.info(f"Distribuição das Classes Após Filtragem:\n{new_class_counts}")

    # Verificar se há desbalanceamento
    if len(new_class_counts.unique()) > 1:
        # Identificar a classe minoritária
        minority_class = new_class_counts.idxmin()
        minority_count = new_class_counts.min()

        # Aplicar Under-sampling na classe majoritária
        df_majority = df_filtered[df_filtered["is_phishing"] != minority_class]
        df_minority = df_filtered[df_filtered["is_phishing"] == minority_class]

        # Verificar se a classe minoritária tem pelo menos 2 exemplos
        if minority_count < 2:
            logger.error(
                "A classe minoritária tem menos de 2 exemplos após a filtragem."
            )
            print("A classe minoritária tem menos de 2 exemplos após a filtragem.")
            sys.exit(1)

        # Downsample da classe majoritária
        try:
            df_majority_downsampled = resample(
                df_majority,
                replace=False,  # sem reposição
                n_samples=minority_count,  # Ajustado para evitar erro
                random_state=42,
            )
            logger.info(
                f"Classe majoritária subamostrada para {minority_count} instâncias."
            )
            print(f"Classe majoritária subamostrada para {minority_count} instâncias.")
        except ValueError as ve:
            logger.error(f"Erro ao subamostrar a classe majoritária: {ve}")
            print(f"Erro ao subamostrar a classe majoritária: {ve}")
            sys.exit(1)

        # Combinar as classes
        df_balanced = pd.concat([df_majority_downsampled, df_minority])

        # Verificar a distribuição final
        balanced_class_counts = df_balanced["is_phishing"].value_counts()
        print("\nDistribuição das Classes Após Under-sampling:")
        print(balanced_class_counts)
        logger.info(
            f"Distribuição das Classes Após Under-sampling:\n{balanced_class_counts}"
        )
    else:
        df_balanced = df_filtered.copy()
        balanced_class_counts = new_class_counts
        print("\nNenhum balanceamento adicional necessário.")
        logger.info("Nenhum balanceamento adicional necessário.")

    # Preparar os dados
    texts = df_balanced["text"].tolist()
    labels = df_balanced["is_phishing"].tolist()

    # Dividir os dados em treinamento e validação
    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        logger.info(
            f"Dados divididos em treinamento ({len(train_texts)} registros) e validação ({len(val_texts)} registros)."
        )
        print(
            f"Dados divididos em treinamento ({len(train_texts)} registros) e validação ({len(val_texts)} registros)."
        )
    except ValueError as ve:
        logger.error(f"Erro na divisão dos dados: {ve}")
        print(f"Erro na divisão dos dados: {ve}")
        sys.exit(1)

    # Inicializar o tokenizer
    tokenizer = BertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    # Pré-processar os dados
    train_dataset = preprocess_bert_data(train_texts, train_labels, tokenizer)
    val_dataset = preprocess_bert_data(val_texts, val_labels, tokenizer)

    # Treinar o modelo
    model = train_bert_model(
        train_dataset, val_dataset, epochs=6, batch_size=32, lr=3e-5, patience=2
    )

    # Salvar o modelo treinado
    output_dir = os.path.join(script_dir, "..", "models", "bert_model")
    save_bert_model(model, tokenizer, output_dir)

    print("Treinamento concluído e modelo salvo com sucesso.")


if __name__ == "__main__":
    main()
