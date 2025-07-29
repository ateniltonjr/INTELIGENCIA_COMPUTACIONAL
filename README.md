# Classificação de Vibrações em Motores com Falta de Fase Usando Redes Neurais

Este projeto utiliza uma **Rede Neural Artificial (RNA)** para classificar o funcionamento de motores elétricos com base nos dados de vibração coletados por um sensor **MPU6050** (acelerômetro + giroscópio).

---

## 🔍 Técnica de Inteligência Computacional Utilizada

A técnica aplicada é uma rede neural multicamadas (MLP - Multilayer Perceptron), treinada com características estatísticas extraídas dos sinais de aceleração e rotação:

- Média
- Desvio padrão
- Valor máximo
- Valor mínimo

```
🧠 Técnica Utilizada: Rede Neural Multicamadas (MLP)
A análise é baseada em uma rede neural multicamadas (MLP – Multilayer Perceptron), que é um tipo de modelo de inteligência computacional supervisionado, capaz de aprender padrões complexos a partir de dados numéricos. Ela é composta por:

Uma camada de entrada, que recebe os dados;

Uma ou mais camadas ocultas, onde ocorrem os cálculos com funções de ativação (como ReLU);

Uma camada de saída, que fornece a predição da classe: motor com todas as fases (classe 0) ou com fase faltando (classe 1).

📊 Entrada da Rede: Extração de Características
Como os sinais brutos do acelerômetro e giroscópio contêm 10.000 amostras por eixo, seria inviável alimentar toda essa quantidade de dados diretamente na rede. Para isso, são extraídas características estatísticas resumidas dos sinais de cada eixo (6 no total: accX, accY, accZ, giroX, giroY, giroZ):

Para cada eixo, são calculadas:
Média: valor médio do sinal, que representa seu nível geral.

Desvio padrão: mede a variação ou dispersão dos dados.

Valor máximo: identifica picos positivos nas vibrações ou rotações.

Valor mínimo: identifica picos negativos.

🔢 Como há 6 eixos e 4 medidas por eixo, isso gera 24 características (6 × 4) por arquivo de dados.

🧪 Treinamento da Rede
Os arquivos CSV são lidos e processados para extrair essas 24 características.

As características são normalizadas (padronizadas) com StandardScaler, para que todos os valores fiquem na mesma escala.

Os dados são separados em treinamento (80%) e teste (20%).

A rede é treinada com os dados rotulados:

0 para motor operando normalmente.

1 para motor com falta de fase.

🧾 Predição
Quando um novo arquivo é fornecido:

As mesmas 24 características são extraídas.

Aplicada a mesma normalização.

O modelo realiza a classificação, prevendo se o motor está em condição normal ou com defeito (falta de fase).

✅ Vantagens da Abordagem
Simples e eficiente para detectar anormalidades baseadas em padrões vibracionais.

A extração de estatísticas reduz o volume de dados e preserva informações essenciais.

A rede MLP consegue aprender relações não lineares entre os dados de vibração e o estado do motor.

````
````

📁 modelo_mpu_nn.h5
O que é: Arquivo que armazena o modelo de rede neural treinado.

Conteúdo:

A arquitetura da rede (camadas, neurônios, ativações).

Os pesos aprendidos durante o treinamento.

As configurações de compilação (otimizador, função de perda, etc.).

Para que serve:

Permite carregar o modelo pronto para fazer novas predições em outros arquivos de sinais, sem precisar treinar tudo de novo.

É usado na função load_model("modelo_mpu_nn.h5").

📁 scaler_nn.pkl
O que é: Arquivo que armazena o normalizador dos dados, criado com StandardScaler da biblioteca sklearn.

Conteúdo:

Os valores de média e desvio padrão usados para normalizar os dados de entrada no treinamento.

Para que serve:

Garante que os novos dados usados na predição passem pela mesma normalização feita durante o treinamento, o que é essencial para manter a consistência e a precisão da rede neural.

É carregado com joblib.load("scaler_nn.pkl").

````

Essas features são extraídas de cada um dos seguintes eixos:  
`accX`, `accY`, `accZ`, `giroX`, `giroY`, `giroZ`.

---

## 📁 Estrutura Esperada dos Dados

A pasta `dataset` deve conter duas subpastas com os arquivos `.csv` de cada classe:

dataset/
├── fase/ # Arquivos CSV com motor funcionando normalmente
├── faltando_fase/ # Arquivos CSV com falta de fase

markdown
Copiar
Editar

Cada arquivo `.csv` deve conter os dados em colunas **sem cabeçalho** com o seguinte formato:

amostra, accX, accY, accZ, giroX, giroY, giroZ

yaml
Copiar
Editar

---

## 🧠 Treinamento

O script `treinamento_mpu.py`:

- Lê todos os arquivos da pasta `dataset`
- Extrai características estatísticas
- Normaliza os dados
- Treina uma rede neural para classificar as medições
- Salva o modelo treinado (`modelo_mpu_nn.h5`) e o scaler (`scaler_nn.pkl`)

Para executar o treinamento:

```bash
python treinamento_mpu.py
📊 Classificação
O script classificacao_mpu.py carrega um novo arquivo .csv com medições e faz a classificação automática.

Exemplo de uso dentro do script:

python
Copiar
Editar
classificar_novo_arquivo("dataset/velo_acc1.csv")
✅ Requisitos
Certifique-se de instalar as dependências necessárias:

bash
Copiar
Editar
pip install pandas numpy scikit-learn tensorflow joblib
📌 Observações
A rede neural foi treinada com um conjunto limitado de medições (10 com todas as fases e 10 com uma fase faltando).

Os dados são normalizados antes do treinamento e da predição.

A saída da classificação indica se o motor está operando normalmente ou com falta de fase.
````
👨‍🔧 Autor

Projeto desenvolvido por:

Atenilton Santos de Souza Júnior para análise de vibrações em motores utilizando sensores MPU e técnicas de inteligência computacional.
