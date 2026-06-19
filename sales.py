"""
PROJETO: PREVISÃO DE VENDAS - REDE NEURONAL LSTM
CURSO: INTELIGÊNCIA ARTIFICIAL (MASTERD)
AUTOR: Nelson Botão
BASE DE DADOS: store_sales.csv importada do Kaggle https://www.kaggle.com/datasets/abhishekjaiswal4896/store-sales-dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==============================================================================
# 1. RECOLHA E PRÉ-PROCESSAMENTO DE DADOS
# ==============================================================================

# Deteta automaticamente a pasta onde o seu script 'sales.py' está guardado
pasta_do_script = os.path.dirname(os.path.abspath(__file__))

# Tenta encontrar o ficheiro correto na pasta
nome_ficheiro = "store_sales.csv"  # Se o nome mudar no seu PC, altere apenas aqui!
caminho_completo = os.path.join(pasta_do_script, nome_ficheiro)

# Carrega os dados com o caminho correto garantido
df_total = pd.read_csv(caminho_completo, parse_dates=["date"])

# Filtrar apenas os dados de uma única loja para isolar o problema específico
df = df_total[df_total["store"] == 1].copy()
df.set_index("date", inplace=True)
df = df.sort_index()

# Limpeza e tratamento de valores ausentes por interpolação
df.fillna(method="ffill", inplace=True)

# Seleção da variável alvo (Vendas)
dados_vendas = df[["sales"]].values


# 2. NORMALIZAÇÃO DOS DADOS (Necessário para a estabilidade da LSTM)

scaler = MinMaxScaler(feature_range=(0, 1))
dados_escalados = scaler.fit_transform(dados_vendas)

# 3. SELEÇÃO E TREINO DO MODELO

# Criação da estrutura de janela temporal (Lookback de 30 dias para prever o dia seguinte)
X, Y = [], []
janela_temporal = 30

for i in range(janela_temporal, len(dados_escalados)):
    X.append(dados_escalados[i-janela_temporal:i, 0])
    Y.append(dados_escalados[i, 0])

X, Y = np.array(X), np.array(Y)

# Redimensionar os dados para o formato tridimensional exigido pela LSTM [amostras, timesteps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Divisão cronológica
divisao = int(len(X) * 0.8)
X_treino, X_teste = X[:divisao], X[divisao:]
Y_treino, Y_teste = Y[:divisao], Y[divisao:]

# Construção da Arquitetura da Rede Neuronal LSTM
modelo = Sequential()
# Camada LSTM com 50 neurónios para capturar padrões de dependência temporal
modelo.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
# Camada de Dropout para evitar o sobreajustamento (Overfitting)
modelo.add(Dropout(0.1))
# Camada de saída densa para prever o valor contínuo de vendas
modelo.add(Dense(units=1))

# Compilação do modelo com o otimizador Adam e função de perda MSE
modelo.compile(optimizer="adam", loss="mean_squared_error")

# Execução do Treino do Modelo
historico = modelo.fit(X_treino, Y_treino, epochs=15, batch_size=32, validation_split=0.1, verbose=1)

# Guardar o modelo treinado para uso posterior em ambiente de produção
modelo.save("modelo_vendas_lstm.h5")

# 4. AVALIAÇÃO DO MODELO 

# Realizar as previsões no conjunto de teste independente
previsoes = modelo.predict(X_teste)

# Inverter a normalização para voltar à escala real de vendas
previsoes_reais = scaler.inverse_transform(previsoes)
Y_teste_real = scaler.inverse_transform(Y_teste.reshape(-1, 1))

# Cálculo da Métrica de Erro: MAPE
mape = np.mean(np.abs((Y_teste_real - previsoes_reais) / Y_teste_real)) * 100
print(f"\n---> ERRO DO MODELO (MAPE): {mape:.2f}% <---\n")

# Gerar o gráfico
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(Y_teste_real):], Y_teste_real, label="Vendas Reais (Histórico)", color="blue", alpha=0.7)
plt.plot(df.index[-len(Y_teste_real):], previsoes_reais, label="Previsão da Rede LSTM", color="orange", linestyle="--")

# --- NOVO: Caixa de texto com o Erro Percentual dentro do gráfico ---
texto_erro = f"Erro do Modelo (MAPE): {mape:.2f}%"
plt.gca().text(0.02, 0.95, texto_erro, transform=plt.gca().transAxes,
            fontsize=12, fontweight='bold', color='darkred',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

# Configurações visuais do gráfico
plt.title("XPTO - Comparativo de Vendas Reais vs Previsões LSTM", fontsize=14, fontweight='bold')
plt.xlabel("Linha Temporal (Datas)", fontsize=12)
plt.ylabel("Volume de Vendas", fontsize=12)
plt.legend(loc="upper right", fontsize=11)
plt.grid(True, linestyle=":")

# Guardar e mostrar o resultado
plt.savefig("resultado_previsao_vendas.png", dpi=300)
plt.show()