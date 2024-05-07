import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.metrics import mean_squared_error

# Carregando o conjunto de dados
dados = pd.read_excel('./imoveis.xlsx', engine='openpyxl')

# Definindo as variáveis independentes e dependentes
variaveis_independentes = ['area', 'dorm1', 'dorm2', 'banh1', 'banh2', 'vaga1', 'vaga2',
                          'renda', 'meiosal', 'dezsal', 'escolar', 'mortinf', 'txcresc',
                          'mortext', 'denspop', 'pop', 'domicil', 'txurb']
variavel_dependente = 'preco'

# Transformando as variáveis `renda`, `meiosal`, `dezsal`, `pop` e `domicil` em logaritmos
dados['renda'] = dados['renda'].apply(lambda x: np.log(x))
dados['meiosal'] = dados['meiosal'].apply(lambda x: np.log(x))
dados['dezsal'] = dados['dezsal'].apply(lambda x: np.log(x))
dados['pop'] = dados['pop'].apply(lambda x: np.log(x))
dados['domicil'] = dados['domicil'].apply(lambda x: np.log(x))

# Ajustando o modelo de regressão linear múltipla
modelo = LinearRegression()
modelo.fit(dados[variaveis_independentes], dados[variavel_dependente])

# Analisando os coeficientes de regressão
print(modelo.coef_)

r2 = r2_score(dados[variavel_dependente], modelo.predict(dados[variaveis_independentes]))
n = len(dados)
p = len(variaveis_independentes) + 1

# Avaliando o modelo com R² ajustado
r2_ajustado = 1 - (1 - r2) * (n - p) / (n - 1)
print(f"R² ajustado: {r2_ajustado:.3f}")

# Avaliando o modelo com teste F
# Calcula MSE
mse = mean_squared_error(dados[variavel_dependente], modelo.predict(dados[variaveis_independentes]))

f_statistic = modelo.n_features_in_ / mse * (modelo.score(dados[variaveis_independentes], dados[variavel_dependente])) # Calculate F-statistic

# Calcula p-value usando F-distribution CDF
df_residual = len(dados) - len(variaveis_independentes) - 1

p_value = 1 - stats.f.cdf(f_statistic, len(dados) - len(variaveis_independentes), df_residual)
print(f"Teste F: {f_statistic:.3f}, p-valor: {p_value:.3f}")

# Validando o modelo com validação cruzada
scores = cross_val_score(modelo, dados[variaveis_independentes], dados[variavel_dependente], cv=5, scoring='r2')
print(f"R² médio na validação cruzada: {scores.mean():.3f}")

# Inicializar variáveis para seleção stepwise
selected_features = []  # List to store selected features
best_aic = np.inf  # Initialize best AIC with a high value
remaining_features = list(variaveis_independentes)  # List of remaining features

# Seleção em loop
while remaining_features:
    # Avaliar cada característica remanescente para inclusão
    for feature in remaining_features:
        # Criar um modelo temporário com a variável candidata
        temp_model = LinearRegression()
        temp_model.fit(dados[selected_features + [feature]], dados[variavel_dependente])

        # Calcula a AIC para o modelo temporario
        n = len(dados)
        k = len(selected_features) + 1  # Including the added feature
        aic = (n * np.mean(mean_squared_error(dados[variavel_dependente],
                                              temp_model.predict(dados[selected_features + [feature]]))) + 2 * k)

        # Atualiza o melhor modelo se a AIC melhorar
        if aic < best_aic:
            best_aic = aic
            best_feature = feature

    # Atualiza as features selecionadas e restantes
    selected_features.append(best_feature)
    remaining_features.remove(best_feature)

    # Printa atualizacao
    print(f"Feature added: {best_feature}")

# Ajuste e avaliação do modelo final
final_model = LinearRegression()
final_model.fit(dados[selected_features], dados[variavel_dependente])

# Analisa e printa os resultados
print("\nModelo do Coeficiente Final:")
print(final_model.coef_)
