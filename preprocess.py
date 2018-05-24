import os
import pandas as pd
import numpy as np
import utils

import utils


def drop_if_missing(data, spared_columns):
  for column in data.columns:
    if column.lower() not in spared_columns:
      data = data[~pd.isna(data[column])]
  return data


def padronizeString(str, isName):
  str = str.replace('.', ' ')
  str = str.replace('*', ' ')
  str = str.replace('/', ' ')
  str = str.replace('-', ' ')
  str = str.replace('0', ' ')
  str = str.replace('1', ' ')
  str = str.replace('2', ' ')
  str = str.replace('3', ' ')
  str = str.replace('4', ' ')
  str = str.replace('5', ' ')
  str = str.replace('6', ' ')
  str = str.replace('7', ' ')
  str = str.replace('8', ' ')
  str = str.replace('9', ' ')
  if isName:
    flag = True
    aux = ''
    for chr in str:
      if flag:
        aux += chr
        flag = False
      if chr == ' ':
        flag = True
    str = aux
  str = str.replace(' ', '')
  str = str.upper()
  
  return str


def preprocess(collab, work, edu, advs, prods):
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
  
  # join advisees data to running data
  data = data.join(advs, how='inner')

  # remove rows with no scientific production
  prods = prods[(prods != 0).any(axis=1)]

  # join scientific production to running data
  data = data.join(prods, how='inner')

  # compute collaborations probabilities
  collab = data['Colaboracoes']
  total = len(collab)
  collab_prob = [np.sum(collab == x) / total for x in np.unique(collab)]

  # compute mutual information between features
  for column in data.columns:
    if column != 'Colaboracoes':
      mi = utils.mutual_information(
          collab, data[column], X_marginal=collab_prob)
      print('I({}; {}) = {}'.format('Colaboracoes', column, mi))

      # discard zero information features
      if np.isclose(mi, 0):
        data = data.drop(columns=column)

  siglas = set()

  siglas.add('USP')
  siglas.add('UFPE')
  siglas.add('FGV')
  siglas.add('UFBA')
  siglas.add('UFRJ')
  siglas.add('UFPR')
  
  success = 0
  for local in data['local']:
    # turn row into list
    local = local.replace('-', ',')
    local = list(local.split(','))
    
    # unpack
    first_field, *_ = local

    if first_field.upper() == 'UNIVERSIDADE FEDERAL DE PERNAMBUCO ':
      siglas.add('UFPE')
      success += 1

    elif first_field.upper() == 'ESCOLA DE ENGENHARIA MAUA':
      siglas.add('IMT')
      success += 1

    elif first_field == 'Centro Universitario Barao de Maua ':
      siglas.add('CBM')
      success += 1

    elif first_field == 'USP ':
      siglas.add('USP')
      success += 1
    
    elif len(local) == 2:
      # unpack
      first_field, second_field = local

      # process strings
      first_field = padronizeString(first_field, True)
      
      second_field = padronizeString(second_field, False)

      if first_field in siglas:
        second_field = first_field

      siglas.add(second_field)
      success += 1

    elif len(local) == 3:
      # unpack
      first_field, second_field, first_complement = local
      
      second_field = padronizeString(second_field, False)

      if (second_field  == 'HUMANAS'):
        second_field = 'FMV'
      
      first_field = padronizeString(first_field, True)

      if first_field in siglas:
        second_field = first_field
      
      first_field = padronizeString(first_field + first_complement, True)
      
      if first_field in siglas:
        second_field = first_field

      first_complement = padronizeString(first_complement, False)

      if first_complement in siglas:
        second_field = first_complement

      if (utils.lcs(second_field, 'CIENCIAS') == 8):
        second_field = 'UCV'

      siglas.add(second_field)
      success += 1

    elif len(local) > 3:
      first_field, first_complement, second_field, second_complement, *_ = local
      second_field = padronizeString(second_field, False)

      first_field = padronizeString(first_field, True)

      if first_field in siglas:
        second_field = first_field

      first_field = padronizeString(first_field + first_complement, True)
      
      if first_field in siglas:
        second_field = first_field

      first_complement = padronizeString(first_complement, False)
      
      second_complement = padronizeString(second_complement, False)

      if first_complement in siglas:
        second_field = first_complement

      if second_complement in siglas:
        second_field = second_complement

      if (second_field == '6'):
        second_field = 'ENSAPB'

      if (second_field  == 'GRADUACAO'):
        second_field = 'FAESPE'
      
      siglas.add(second_field)
      success += 1
    else:
      if local[0] in siglas:
        success += 1

  print(len(siglas))
  print(success, len(data['local']) - success)

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
  advs_path = os.path.join(FLAGS.data_path, 'Orientacoes.csv')
  advs = pd.read_csv(advs_path, index_col='Identificador', sep=';')
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

  print(preprocess(collab, work, edu, advs, prods))


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
