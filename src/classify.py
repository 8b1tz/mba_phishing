# com_bert/src/classify.py

import logging
import pickle

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_resources(
    model_path="models/phishing_spam_model_best.h5",
    vectorizer_path="models/tfidf_vectorizer_phishing_spam.pkl",
):
    """
    Carrega o modelo e o vetor TF-IDF do modelo anterior.

    Args:
        model_path (str): Caminho para o modelo treinado.
        vectorizer_path (str): Caminho para o vetor TF-IDF.

    Returns:
        tuple: (modelo, vetor TF-IDF)
    """
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        logger.info(f"Vetor TF-IDF carregado de {vectorizer_path}.")
    except Exception as e:
        logger.error(f"Erro ao carregar o vetor TF-IDF: {e}")
        raise

    try:
        # Supondo que o modelo anterior seja um classificador do Scikit-learn
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Modelo carregado de {model_path}.")
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {e}")
        raise

    return model, vectorizer


def classify(model, vectorizer, message, threshold=0.5):
    """
    Classifica uma mensagem como phishing ou legítima usando o modelo anterior.

    Args:
        model (sklearn.base.BaseEstimator): Modelo treinado.
        vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): Vetor TF-IDF.
        message (str): Mensagem a ser classificada.
        threshold (float, optional): Limite para classificação binária. Defaults to 0.5.

    Returns:
        str: 'phishing' ou 'legítimo'
    """
    try:
        X = vectorizer.transform([message])
        proba = model.predict_proba(X)[0][1]
        return "phishing" if proba >= threshold else "legítimo"
    except Exception as e:
        logger.error(f"Erro ao classificar a mensagem: {e}")
        return "Erro"
