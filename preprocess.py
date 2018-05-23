import os
import pandas as pd
import numpy as np


def drop_if_missing(data, spared_columns):
  for column in data.columns:
    if column.lower() not in spared_columns:
      data = data[~pd.isna(data[column])]
  return data


def preprocess(collab, work, edu, advs, prods, langs):
  # drop rows with no collaborations
  data = collab[collab['Colaboracoes'] != 0]

  # drop work rows with missing values,
  # except UF and country because
  # both can be deducted
  work = drop_if_missing(work, ('uf', 'pais'))

  # join work data to running data
  data = data.join(work, how='inner')

  # drop edu rows with missing values
  # except start and beginning dates
  edu = drop_if_missing(edu, ('inicio', 'fim'))

  # join education data to running data
  data = data.join(edu, how='inner')

  return data


def main():
  # load collaborations graph in chunks
  print('Loading collaborations graph...')
  n_collab = np.zeros(265188, dtype=np.int32)
  collab_path = os.path.join(FLAGS.data_path, 'Colaboracoes.csv')
  for chunk in pd.read_csv(
      collab_path,
      sep=';',
      chunksize=FLAGS.chunk_sz,
      dtype=np.int32,
      usecols=[0, 1, 3]):
    # retrieve edge's endpoints and weight
    u, v, w = chunk
    u = chunk[u]
    v = chunk[v]
    w = chunk[w]

    # update degrees
    n_collab[u - 1] += w
    n_collab[v - 1] += w

  # convert to pandas dataframe
  collab = pd.DataFrame(
      n_collab,
      index=np.arange(1, len(n_collab) + 1),
      columns=['Colaboracoes'])

  print('Loaded.')

  # load affiliation and area of interest
  print('Loading professional information data...')
  work_path = os.path.join(FLAGS.data_path, 'Atuacao_Profissional.csv')
  work = pd.read_csv(work_path, sep=';', index_col='Identificador')
  print('Loaded.')

  # load education data discarding unknown columns
  print('Loading education data...')
  edu_path = os.path.join(FLAGS.data_path, 'Formacao_Academica.csv')
  edu = pd.read_csv(
      edu_path, sep=';', index_col='Identificador', usecols=range(21))
  print('Loaded.')

  # load number of advisees
  print('Loading advising data...')
  adv_path = os.path.join(FLAGS.data_path, 'Orientacoes.csv')
  adv = pd.read_csv(adv_path, index_col='Identificador', sep=';')
  print('Loaded.')

  # load number of scientific productions
  # discarding last update
  print('Loading scientific production data...')
  prods_path = os.path.join(FLAGS.data_path, 'Producao_Cientifica.csv')
  prods = pd.read_csv(
      prods_path,
      index_col='Identificador',
      sep=';',
      usecols=lambda x: 'Ultima' not in x)
  print('Loaded.')

  # load language proficiency
  print('Loading language proficiency data...')
  langs_path = os.path.join(FLAGS.data_path, 'Proficiencia.csv')
  langs = pd.read_csv(langs_path, index_col='Identificador', sep=';')
  print('Loaded.')

  print(preprocess(collab, work, edu, adv, prods, langs))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_path', default='.', type=str, help='Path to dataset path.')
  parser.add_argument(
      '--chunk_sz',
      default=1123456,
      type=int,
      help='Chunk size to read large collaborations file.')
  FLAGS, _ = parser.parse_known_args()

  main()
