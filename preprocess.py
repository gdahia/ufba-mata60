import pandas as pd
import numpy as np

import utils


def compute_n_collab(collab):
  n_collab = np.zeros(np.max(collab[:, :2]), dtype=np.int32)
  for v, u, w in collab:
    n_collab[v - 1] += w
    n_collab[u - 1] += w

  return n_collab


def remove_if_missing(data, spared_columns):
  for column in data.columns:
    if column.lower() not in spared_columns:
      data = data[~pd.isna(data[column])]
  return data


def preprocess(collab, work, cv):
  # compute number of collaborations per subject
  print('Computing number of collaborations...')
  n_collab = compute_n_collab(collab)
  print('Computed.')

  # convert to pandas dataframe
  data = pd.DataFrame(
      n_collab,
      index=np.arange(1, len(n_collab) + 1),
      columns=['Colaboracoes'])

  # drop rows with no collaborations
  data = data[data['Colaboracoes'] != 0]

  # drop rows with missing values,
  # except UF and country because
  # both can be deducted
  work = remove_if_missing(work, ('uf', 'pais'))

  # join work data to running data
  data = data.join(work, how='inner')

  print(data)


def main():
  # load collaborations graph
  print('Loading collaborations graph...')
  collab = utils.read_lattes_csv(open(FLAGS.collab_data_path, 'r'))
  collab = np.array(collab)
  collab = collab[:, [0, 1, 3]].astype(np.int32)
  print('Loaded.')

  # load affiliation and area of interest
  print('Loading professional information data...')
  work = pd.read_csv(
      FLAGS.work_data_path, delimiter=';', index_col='Identificador')
  print('Loaded.')

  preprocess(collab, work)


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
  FLAGS, _ = parser.parse_known_args()

  main()
