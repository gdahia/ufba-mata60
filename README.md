# Trabalho de Mineração de Dados para disciplina Banco de Dados - MATA60
Para executar a etapa de pré-processamento, as dependências são:

* python 3.5.2;
* numpy 1.14;
* sklearn 0.19.1;
* pandas 0.23;
* matplotlib 2.1.2 (opcional, necessário apenas para plotar gráfico de informação mútua).

Outras versões devem funcionar, mas não há garantias quanto a isso.

Para pré-processar a base, é necessário descompactar os arquivos do LattesDoctoralDataset em uma única pasta.
O comando, então, é `python3 preprocess.py`:
```
usage: preprocess.py [-h] [--data_path DATA_PATH] [--chunk_sz CHUNK_SZ]
                     [--plot_path PLOT_PATH] [--results_path RESULTS_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to dataset.
  --chunk_sz CHUNK_SZ   Chunk size to read large collaborations file.
  --plot_path PLOT_PATH
                        Path to save mutual information plot.
  --results_path RESULTS_PATH
                        Path to save resulting csvs.
```

Se os arquivos da base foram descompactados em uma pasta chamada `data`, o comando seria `python3 preprocess.py --data_path data`
