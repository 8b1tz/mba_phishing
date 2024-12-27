# com_bert/scripts/test_messages_bert.py

import logging
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
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


def main():
    print("Iniciando o teste do modelo BERT.")

    # Diretório atual do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Diretório do script: {script_dir}")

    # Caminho para o modelo treinado
    model_dir = os.path.join(script_dir, "..", "models", "bert_model")
    logger.info(f"Caminho do modelo: {model_dir}")

    # Verificar se o diretório do modelo existe
    if not os.path.isdir(model_dir):
        logger.error(f"Diretório do modelo não encontrado: {model_dir}")
        sys.exit(1)
    else:
        logger.info("Diretório do modelo encontrado.")

    # Carregar o modelo e o tokenizer
    try:
        model, tokenizer = load_bert_model(model_dir)
        logger.info("Modelo e tokenizer carregados com sucesso.")
        print("Modelo e tokenizer carregados com sucesso.")
    except Exception as e:
        logger.error(f"Falha ao carregar o modelo: {e}")
        print(f"Falha ao carregar o modelo: {e}")
        sys.exit(1)

    # Definir o dispositivo
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Dispositivo usado para inferência: {device}")
    print(f"Dispositivo usado para inferência: {device}")

    # Lista de mensagens para teste com rótulos esperados
    test_cases = [
        # Phishing (20 mensagens)
        {
            "message": "PAGAMENTO PENDENTE: Faça sua confirmação via PIX e evite multas!",
            "expected": "Phishing",
        },
        {
            "message": "Parabéns! Você ganhou um prêmio. Clique no link para resgatar.",
            "expected": "Phishing",
        },
        {
            "message": "Seu cartão foi bloqueado por motivos de segurança. Clique aqui para reativá-lo.",
            "expected": "Phishing",
        },
        {
            "message": "Parabéns! Você ganhou um prêmio exclusivo. Responda para reivindicar.",
            "expected": "Phishing",
        },
        {
            "message": "Verificação de Conta Bancária Urgente: Clique aqui para atualizar suas informações.",
            "expected": "Phishing",
        },
        {
            "message": "Atualização Obrigatória do Microsoft Office: Proteja seu sistema agora.",
            "expected": "Phishing",
        },
        {
            "message": "Saldo do PayPal Bloqueado: Acesse o link para desbloquear sua conta.",
            "expected": "Phishing",
        },
        {
            "message": "Notificação Urgente: Verifique Seu E-mail para evitar bloqueio.",
            "expected": "Phishing",
        },
        {
            "message": "Presente Exclusivo da Apple! Clique aqui para receber seu voucher.",
            "expected": "Phishing",
        },
        {
            "message": "Dispositivo iCloud Sobrecarregado: Liberar espaço agora.",
            "expected": "Phishing",
        },
        {
            "message": "Confirme Sua Conta do Google para continuar recebendo e-mails.",
            "expected": "Phishing",
        },
        {
            "message": "Cartão de Crédito Expirado: Atualize suas informações aqui.",
            "expected": "Phishing",
        },
        {
            "message": "Certificado SSL Expirado: Renove seu certificado imediatamente.",
            "expected": "Phishing",
        },
        {
            "message": "Sua Conta foi Bloqueada: Clique aqui para reativá-la.",
            "expected": "Phishing",
        },
        {
            "message": "Atualização de Segurança: Atualize sua senha agora.",
            "expected": "Phishing",
        },
        {
            "message": "Oferta Limitada: Ganhe um smartphone grátis clicando aqui.",
            "expected": "Phishing",
        },
        {
            "message": "Alerta de Fraude: Proteja sua conta bancária imediatamente.",
            "expected": "Phishing",
        },
        {
            "message": "Recompensa Exclusiva: Ganhe R$1000 grátis agora.",
            "expected": "Phishing",
        },
        {
            "message": "Confirmação de Pagamento: Verifique seu pagamento clicando aqui.",
            "expected": "Phishing",
        },
        # Legítimas (10 mensagens)
        {"message": "Lembrete: Relatório Mensal", "expected": "Legítima"},
        {
            "message": "Confirmação de Inscrição no Curso de Python",
            "expected": "Legítima",
        },
        {"message": "Detalhes da Reunião com o Cliente", "expected": "Legítima"},
        {"message": "Obrigado por Atualizar Suas Informações", "expected": "Legítima"},
        {"message": "Manutenção do Sistema de TI", "expected": "Legítima"},
        {
            "message": "Convite para Workshop de Desenvolvimento Profissional",
            "expected": "Legítima",
        },
        {"message": "Acesso ao Sistema Restabelecido", "expected": "Legítima"},
        {"message": "Novas Vagas no Departamento de Marketing", "expected": "Legítima"},
        {"message": "Lembrete: Conferência de Tecnologia", "expected": "Legítima"},
        {"message": "Parabéns pelo Novo Cargo!", "expected": "Legítima"},
        {
            "message": "Reunião marcada para amanhã às 10h. Não se esqueça de preparar os documentos.",
            "expected": "Legítima",
        },
    ]

    correct = 0
    incorrect = 0

    y_true = []
    y_pred = []

    print(
        f"{'Teste':<6} {'Resultado Esperado':<15} {'Classificação BERT':<20} {'Status':<10}"
    )
    print("-" * 60)

    # Iterar sobre as mensagens e classificar
    for idx, test in enumerate(test_cases, 1):
        message = test["message"]
        expected = test["expected"]

        classification = classify_bert(message, model, tokenizer, device)

        y_true.append(expected.lower())
        y_pred.append(classification.lower())

        # Verificar se a classificação está correta
        if classification.lower() == expected.lower():
            correct += 1
            status = "Correto"
        else:
            incorrect += 1
            status = "Incorreto"

        print(f"{idx:<6} {expected:<15} {classification:<20} {status:<10}\n")

    # Resumo dos resultados
    total_tests = len(test_cases)
    print("Resumo dos Resultados:")
    print(f"Corretos: {correct} / {total_tests}")
    print(f"Incorretos: {incorrect} / {total_tests}")

    # Relatório de Classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=["Legítima", "Phishing"]))

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred, labels=["legítima", "phishing"])
    print("Matriz de Confusão:")
    print(cm)

    # Plotar a Matriz de Confusão
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legítima", "Phishing"],
        yticklabels=["Legítima", "Phishing"],
    )
    plt.xlabel("Predição")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão")
    plt.show()

    print("Script concluído.")


if __name__ == "__main__":
    main()
