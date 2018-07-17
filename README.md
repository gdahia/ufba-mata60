# Trabalho de Mineração de Dados para disciplina Banco de Dados - MATA60
Este repositório contém todos os códigos do trabalho final de Gabriel Dahia, Gabriel Lecomte e Pedro Vidal para a disciplina MATA60 - Banco de Dados da Universidade Federal da Bahia.

Para este trabalho, utilizamos a base [LattesDoctoralDataset](https://github.com/thiagomagela/LattesDoctoralDataset). A partir dela, analisamos e utilizamos técnicas de mineração de dados para prever as contribuições entre os doutores na base.

## Configuração
Os códigos desse repositório requerem Python3.

É recomendado a utilização de um ambiente virtual Python (`virtualenv`) para instalação e execução dos códigos. Para instalar o `virtualenv` para o Python3, no Ubuntu, utilize:
```
sudo apt install python3-venv
```
Em seguida, para configurar o ambiente virtual para executar os códigos, faça:
```
python3 -m venv env               # Cria o ambiente virtual
source env/bin/activate           # Ativa o ambiente virtual
pip3 install -r requirements.txt  # Instala as dependencias
```
e os pode rodar os códigos como explicado abaixo.

Para deixar o ambiente virtual:
```
deactivate                        # Desativa o ambiente virtual
```

## Execução
Depois de baixar o LattesDoctoralDataset, descompacte-o em uma pasta, dentro deste repositório, chamada `data`.

Em seguida, faça `python3 -m preprocess --seed 0`. As opções desse código são:
```
usage: preprocess.py [-h] [--data_path DATA_PATH] [--chunk_sz CHUNK_SZ]
                     [--plot_path PLOT_PATH] [--results_path RESULTS_PATH]
                     [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to dataset.
  --chunk_sz CHUNK_SZ   Chunk size to read large collaborations file.
  --plot_path PLOT_PATH
                        Path to save mutual information plot.
  --results_path RESULTS_PATH
                        Path to save resulting csvs.
  --seed SEED           random seed
```

Depois da etapa de pré-processamento, é necessário realizar a preparação e divisão dos dados em conjunto de treino e teste. Isso é feito com `python3 -m create_dataset --seed 0`. Para esse código, as opções são:
```
usage: create_dataset.py [-h] [--data_path DATA_PATH] [--save_path SAVE_PATH]
                         [--test_size TEST_SIZE] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        path to dataset
  --save_path SAVE_PATH
                        path to save resulting dataset
  --test_size TEST_SIZE
                        share of dataset for testing
  --seed SEED           random seed
```

Para treinar o modelo, o comando é `python3 -m train --seed 0 --data_path data/train.pkl --binary`. Suas opções são:
```
usage: train.py [-h] --data_path DATA_PATH [--save_path SAVE_PATH]
                [--seed SEED] [--n_trees N_TREES] [--binary]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        path to training dataset
  --save_path SAVE_PATH
                        path to save resulting model
  --seed SEED           random seed
  --n_trees N_TREES     number of tree estimators in random forest classifier
  --binary              use this flag to perform binary classification
```

Por último, para validar o modelo, faça `python3 -m validate --binary --data_path data/test.pkl --model_path data/model.pkl`. Esse código admite as opções:
```
usage: validate.py [-h] --data_path DATA_PATH --model_path MODEL_PATH
                   [--binary]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        path to test set
  --model_path MODEL_PATH
                        path to trained classifier model
  --binary              use this flag to perform binary classification
```

O resultado esperado é a acurácia do método, que, quando rodado com os parâmetros acima, deve ser de `0.8689972453355295`.

## Validação
O conjunto de dados acima é dividido da seguinte maneira: são sorteados 60% de todos os pares de currículos para que sejam usados no treino; o restante é usado no teste. A divisão é realizada de maneira balanceada, garantindo que dos pares presentes no treino e no teste, metade sejam de colaborações e metade não o sejam.

O resultado reportado acima é para o caso em que classificamos apenas se, dado o par de currículos pré-processados, há colaboração ou não entre eles. Chamamos essa situação de `binary`, ou classificação binária. Apresentamos também os resultados onde tentamos estimar, dados dois currículos, quantas colaborações já existem entre seus pesquisadores.

Calculamos a acurácia como:

<div align='center'>
  <img src='accuracy.png' width='240px'>
</div>

A acurácia para `binary` é, usando `seed` 0, `~86.90%`. Para o outro caso, também usando `seed` 0, é de `0.7830062760912163` ou aproximadamente `78.30%`.

Ressaltamos que esses resultados, são, na verdade, um limite inferior da performance do nosso métodos. Caso nosso algoritmo classifique que entre dois pesquisadores possa haver uma colaboração mas ela ainda não exista, não há garantias que isso, no futuro, não vá ocorrer.
