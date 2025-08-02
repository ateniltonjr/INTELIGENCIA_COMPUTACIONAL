import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import io
import PIL.Image

# Oculta mensagens de log do TensorFlow (aviso de compilador etc)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Função para extrair features de um único arquivo
def extrair_features(df):
    features = []
    colunas = ['accX', 'accY', 'accZ', 'giroX', 'giroY', 'giroZ']
    for col in colunas:
        features.append(df[col].mean())
        features.append(df[col].std())
        features.append(df[col].max())
        features.append(df[col].min())
    return features

# Função para carregar o arquivo CSV e classificá-lo
def classificar_novo_arquivo(caminho_csv):
    print("🔄 Carregando modelo e dados de normalização...")
    try:
        model = load_model("modelo_mpu_nn.h5")
        scaler = joblib.load("scaler_nn.pkl")
    except FileNotFoundError:
        print("❌ Arquivo de modelo 'modelo_mpu_nn.h5' não encontrado. Treine o modelo antes de classificar.")
        return

    print(f"📥 Lendo arquivo: {caminho_csv}")
    colunas = ['amostra', 'accX', 'accY', 'accZ', 'giroX', 'giroY', 'giroZ']
    df = pd.read_csv(caminho_csv, header=None, names=colunas)

    print("🔎 Extraindo características do sinal...")
    feats = np.array(extrair_features(df)).reshape(1, -1)
    feats_scaled = scaler.transform(feats)

    print("🧠 Realizando predição...")
    pred = model.predict(feats_scaled, verbose=0)
    classe = np.argmax(pred, axis=1)[0]
    descricao = "Motor em funcionamento normal (todas as fases)" if classe == 0 else "Falta de fase detectada no motor"

    print(f"\nResultado da Classificação: {descricao}")


print("__________________________________________________")
classificar_novo_arquivo("banco_de_dados/teste_motor_faltando_fase.csv")
