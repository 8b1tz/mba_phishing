# com_bert/scripts/classify_message_bert.py

import logging
import os
import sys

import torch
from src.bert_model import load_bert_model

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_bert(message, model, tokenizer, device, max_length=128):
    """
    Classifica uma mensagem como phishing ou legítima usando o modelo BERT.

    Args:
        message (str): Mensagem a ser classificada.
        model (BertForSequenceClassification): Modelo BERT treinado.
        tokenizer (BertTokenizer): Tokenizer do BERT.
        device (torch.device): Dispositivo para inferência (CPU ou GPU).
        max_length (int, optional): Comprimento máximo dos tokens. Defaults to 128.

    Returns:
        str: 'Phishing' ou 'Legítima'
    """
    try:
        inputs = tokenizer(
            [message],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        return "Phishing" if predicted_class == 1 else "Legítima"
    except Exception as e:
        logger.error(f"Erro ao classificar a mensagem: {e}")
        return "Erro"


def load_model(model_dir="models/bert_model"):
    """
    Carrega o modelo BERT e o tokenizer.

    Args:
        model_dir (str, optional): Diretório onde o modelo está salvo. Defaults to 'models/bert_model'.

    Returns:
        tuple: (model, tokenizer, device)
    """
    if not os.path.isdir(model_dir):
        logger.error(f"Diretório do modelo não encontrado: {model_dir}")
        raise FileNotFoundError(f"Diretório do modelo não encontrado: {model_dir}")

    model, tokenizer = load_bert_model(model_dir)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return model, tokenizer, device


def main():
    """
    Função principal para classificar uma mensagem fornecida pelo usuário.
    """
    print("Classificação de Mensagens com Modelo BERT")
    print("------------------------------------------")

    # Carregar o modelo e o tokenizer
    try:
        model, tokenizer, device = load_model()
        print("Modelo e tokenizer carregados com sucesso.")
    except Exception as e:
        logger.error(f"Falha ao carregar o modelo: {e}")
        print(f"Falha ao carregar o modelo: {e}")
        sys.exit(1)

    while True:
        message = input(
            "Digite a mensagem a ser classificada (ou 'sair' para encerrar): "
        )
        if message.lower() == "sair":
            print("Encerrando a classificação.")
            break
        classification = classify_bert(message, model, tokenizer, device)
        print(f"Classificação: {classification}\n")


if __name__ == "__main__":
    main()
