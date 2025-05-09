## Importacao da tabela

import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)
dados.head()

## Organizacao e limpeza de dados

a_renomear = {
    'expected_hours' : 'horas_esperadas',
    'price' : 'preco',
    'unfinished' : 'nao_finalizado'
}
dados = dados.rename(columns = a_renomear)
dados.head()

troca = {
    0 : 1,
    1 : 0
}
dados['finalizado'] = dados.nao_finalizado.map(troca)
dados.head()

## Grafico relacionando Horas esperadas x preco

import seaborn as sns
sns.scatterplot(x="horas_esperadas", y="preco", data=dados)

## Organizando grafico com cores

sns.scatterplot(x="horas_esperadas", y="preco", hue="finalizado", data=dados)

##separando graficos

sns.relplot(x="horas_esperadas", y="preco", hue='finalizado', col='finalizado', data=dados)

## Inicio do treinamento 

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x,y,
                                                        random_state=SEED, test_size = 0.25,
                                                        stratify = y)

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes)*100

print("a acuracia	foi %.2f%%" % acuracia)


print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

## criando precisoes teste com valores base (baseline)

import numpy as np
previsoes_de_base= np.ones(540)
acuracia=accuracy_score(teste_y, previsoes_de_base) *100
print('A acuracia de baseline foi %.2f%%' % acuracia)

## Gerando grafico novamente apóes teste para entender o baseline

sns.scatterplot(x="horas_esperadas", y="preco", hue=teste_y, data=dados)

## Segregando os dados para treino do baseline e comparacao com SVC
x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()
print(x_min, x_max, y_min, y_max)

import numpy as np

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]
pontos

z = modelo.predict(pontos)
z = z.reshape(xx.shape)
z

## comeco da criacao de Boundery line
import matplotlib.pyplot as plt

plt.contourf(xx, yy, z, alpha = 0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)

## treino após entendimento das decisoes previas e Boundery Line

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x,y,
                                                        random_state=SEED, test_size = 0.25,
                                                        stratify = y)

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes)*100

print("a acuracia	foi %.2f%%" % acuracia)


print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

##grafico para checagem de alteracoes

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

z = modelo.predict(pontos)
z = z.reshape(xx.shape)
z

import matplotlib.pyplot as plt

plt.contourf(xx, yy, z, alpha = 0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)

## novo treino utlizando raw data 

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 5
np.random.seed(SEED)
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x,y,
                                                        random_state=SEED, test_size = 0.25,
                                                        stratify = y)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes)*100
print("a acuracia	foi %.2f%%" % acuracia)

## alterando eixos sobrepondo valores para que seja possivel a implementacao de curva de decisao

treino_x

data_x = teste_x[:, 0]
data_y = teste_x[: ,1]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

z = modelo.predict(pontos)
z = z.reshape(xx.shape)

import matplotlib.pyplot as plt

plt.contourf(xx, yy, z, alpha = 0.3)
plt.scatter(data_x, data_y, c=teste_y, s=1)

