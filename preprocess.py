import pandas as pd
import numpy as np


def drop_if_missing(data, spared_columns):
  for column in data.columns:
    if column.lower() not in spared_columns:
      data = data[~pd.isna(data[column])]
  return data


def preprocess(collab, work, edu):
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

  print(data)


def main():
  # load collaborations graph in chunks
  print('Loading collaborations graph...')
  n_collab = np.zeros(265188, dtype=np.int32)
  for chunk in pd.read_csv(
      FLAGS.collab_data_path,
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

  print('Loaded.')

  # convert to pandas dataframe
  collab = pd.DataFrame(
      n_collab,
      index=np.arange(1, len(n_collab) + 1),
      columns=['Colaboracoes'])

  # load affiliation and area of interest
  print('Loading professional information data...')
  work = pd.read_csv(FLAGS.work_data_path, sep=';', index_col='Identificador')
  print('Loaded.')

  # load education data
  print('Loading education data...')
  edu = pd.read_csv(
      FLAGS.edu_data_path,
      sep=';',
      index_col='Identificador',
      usecols=range(21))

  preprocess(collab, work, edu)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--collab_data_path',
      default='Colaboracoes.csv',
      type=str,
      help='Path to collaborations data csv.')
  parser.add_argument(
      '--work_data_path',
      default='Atuacao_Profissional.csv',
      type=str,
      help='Path to working data csv.')
  parser.add_argument(
      '--edu_data_path',
      default='Formacao_Academica.csv',
      type=str,
      help='Path to educational data csv.')
  parser.add_argument(
      '--chunk_sz',
      default=1123456,
      type=int,
      help='Chunk size to read large collaborations file.')
  FLAGS, _ = parser.parse_known_args()

  main()
