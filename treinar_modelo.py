# Importa bibliotecas necessárias
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib  # Para salvar o scaler

# Função para extrair características (features) dos dados brutos
def extrair_features(df):
    """
    Extrai features estatísticas (média, desvio padrão, máx e min) para cada eixo do sensor.
    """
    features = []
    colunas = ['accX', 'accY', 'accZ', 'giroX', 'giroY', 'giroZ']
    for col in colunas:
        features.append(df[col].mean())
        features.append(df[col].std())
        features.append(df[col].max())
        features.append(df[col].min())
    return features

# Função para carregar os arquivos .csv de uma pasta
def carregar_dados_pasta(caminho, label):
    """
    Lê todos os arquivos .csv de uma pasta, extrai as features e associa o rótulo (label).
    """
    dados = []
    rotulos = []
    for arquivo in os.listdir(caminho):
        if arquivo.endswith(".csv"):
            colunas = ['amostra', 'accX', 'accY', 'accZ', 'giroX', 'giroY', 'giroZ']
            df = pd.read_csv(os.path.join(caminho, arquivo), header=None, names=colunas)
            feats = extrair_features(df)
            dados.append(feats)
            rotulos.append(label)
    return dados, rotulos

# Caminhos das pastas com os dados das duas classes
caminho_fase = "banco_de_dados/fase"
caminho_falta = "banco_de_dados/faltando_fase"

# Carrega os dados com e sem falha
X_fase, y_fase = carregar_dados_pasta(caminho_fase, 0)    # 0 = motor normal
X_falta, y_falta = carregar_dados_pasta(caminho_falta, 1) # 1 = falta de fase




# Junta dados e rótulos em arrays, mantendo duplicatas
X = np.array(X_fase + X_falta)
y = np.array(y_fase + y_falta)

# Embaralha os dados para evitar que duplicatas fiquem juntas
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)

# Verifica duplicatas e alerta sobre impacto nas métricas
import pandas as pd
df_check = pd.DataFrame(X)
df_check['label'] = y
num_total = len(df_check)
num_duplicadas = num_total - len(df_check.drop_duplicates())
if num_duplicadas > 0:
    print(f"⚠️ Atenção: Existem {num_duplicadas} amostras duplicadas nos dados ({num_duplicadas/num_total:.2%}). Isso pode causar métricas irreais como acurácia 1.0.")

# Alerta se o número de arquivos for muito pequeno
if len(X) < 30:
    print(f"⚠️ Poucas amostras ({len(X)}). Métricas podem ser irreais. Use mais arquivos para resultados confiáveis.")

# Normaliza as features (zero média e variância 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divide os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Converte os rótulos para formato one-hot
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Define a arquitetura da rede neural
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 2 saídas para 2 classes

# Compila o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treina o modelo
history = model.fit(X_train, y_train_cat, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Avalia o modelo no conjunto de teste
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Acurácia no teste: {acc:.4f}")

# Mostra o relatório de classificação com métricas reais
y_pred_cat = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_cat, axis=1)
print(classification_report(y_test, y_pred, digits=4))

# Plota os gráficos de loss e accuracy
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss por época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.tight_layout()
plt.show()

# Salva o modelo, o normalizador e o histórico
model.save("modelo_mpu_nn.h5")
joblib.dump(scaler, "scaler_nn.pkl")
import pickle
with open("history_nn.pkl", "wb") as f:
    pickle.dump(history.history, f)
