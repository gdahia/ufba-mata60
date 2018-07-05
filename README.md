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

