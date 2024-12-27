# com_bert/src/bert_model.py

import logging
import os
import re

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTDataset(Dataset):
    """
    Dataset personalizado para o modelo BERT.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Limpeza básica do texto
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


def load_bert_model(model_dir):
    """
    Carrega o modelo BERT e o tokenizer a partir do diretório especificado.

    Args:
        model_dir (str): Caminho para o diretório onde o modelo e o tokenizer estão salvos.

    Returns:
        tuple: (model, tokenizer)
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Usando o dispositivo: {device}")

    try:
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        logger.info(f"Modelo e tokenizer carregados de {model_dir}.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {e}")
        raise


def save_bert_model(model, tokenizer, output_dir):
    """
    Salva o modelo BERT e o tokenizer.

    Args:
        model (BertForSequenceClassification): Modelo treinado.
        tokenizer (BertTokenizer): Tokenizer do BERT.
        output_dir (str): Diretório de saída para salvar o modelo.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Modelo e tokenizer salvos em {output_dir}.")


def preprocess_bert_data(texts, labels, tokenizer, max_length=128):
    """
    Pré-processa os dados para serem utilizados pelo BERT.

    Args:
        texts (list): Lista de textos.
        labels (list): Lista de rótulos.
        tokenizer (BertTokenizer): Tokenizer do BERT.
        max_length (int, optional): Comprimento máximo dos tokens. Defaults to 128.

    Returns:
        BERTDataset: Dataset preparado para o BERT.
    """
    return BERTDataset(texts, labels, tokenizer, max_length)


def train_bert_model(
    train_dataset, val_dataset, epochs=4, batch_size=16, lr=2e-5, patience=2
):
    """
    Treina o modelo BERT com pesos de classe e early stopping.

    Args:
        train_dataset (BERTDataset): Dataset de treinamento.
        val_dataset (BERTDataset): Dataset de validação.
        epochs (int, optional): Número de épocas. Defaults to 4.
        batch_size (int, optional): Tamanho do batch. Defaults to 16.
        lr (float, optional): Taxa de aprendizado. Defaults to 2e-5.
        patience (int, optional): Número de épocas para esperar antes de parar. Defaults to 2.

    Returns:
        BertForSequenceClassification: Modelo treinado.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Usando o dispositivo: {device}")

    # Carregar modelo pré-treinado
    model = BertForSequenceClassification.from_pretrained(
        "distilbert-base-multilingual-cased", num_labels=2
    )
    model.to(device)

    # Criar DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Calcular pesos de classe para lidar com classes desbalanceadas
    labels = [item["labels"].item() for item in train_dataset]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    weights = class_weights / class_weights.sum() * 2  # Normalizar
    weights = weights.to(device)
    logger.info(f"Pesos das classes: {weights}")

    # Definir a função de perda com pesos de classe
    loss_fn = CrossEntropyLoss(weight=weights)

    # Definir o otimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Scheduler para reduzir a taxa de aprendizado
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Implementar Early Stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Treinamento com progress bar
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Treinando")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss de Treinamento: {avg_train_loss}")

        # Validação com progress bar
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validando")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                accuracy = correct / total
                progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy)

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct / total
        logger.info(
            f"Epoch {epoch+1}/{epochs} - Loss de Validação: {avg_val_loss} - Acurácia: {accuracy:.4f}"
        )

        # Verificar Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Salvar o melhor modelo até agora
            output_dir = os.path.join(os.getcwd(), "models", "bert_model")
            save_bert_model(model, train_dataset.tokenizer, output_dir)
            logger.info("Melhor modelo salvo.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping ativado após {patience} épocas sem melhoria."
                )
                break

        # Atualizar scheduler
        scheduler.step()

    return model
