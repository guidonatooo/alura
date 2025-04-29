## Importacao dados para treinamento da ML

import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

pd.read_csv(uri)
dados = pd.read_csv(uri)
dados.head()
## Change of names - Mudanca de nomes ##
mapa = {
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"
}
dados = dados.rename(columns = mapa)

## Separacao dados

x= dados[["principal","como_funciona","contato"]]
y= dados[["comprou"]]

x.head()

## shape dos dados
dados.shape

## Escolha e divisão de treino

treino_x = x[:75]
treino_x.shape

## Treino
treino_x = x[:75]
treino_y = y[:75]

teste_x = x[75:]
teste_y = y[75:]

## Primeiro treino completo
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes)*100

print("a acuracia	foi %.2f%%" % acuracia)
