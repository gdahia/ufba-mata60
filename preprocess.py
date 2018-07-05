import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

import utils


def drop_if_missing(data):
  for column in data.columns:
    data = data[~pd.isna(data[column])]
  return data


def cluster_text(data, columns, n_clusters, stop_words=[]):
  # retrieve all text
  all_text = []
  for column in sorted(columns):
    all_text += list(data[column])

  # get tf-idf matrix
  vectorizer = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True)
  tfidf = vectorizer.fit_transform(all_text)

  # perform LSA
  lsa = TruncatedSVD(n_components=100)
  X = lsa.fit_transform(tfidf)

  # cluster with K-means
  km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=3 * n_clusters)
  clustered = km.fit_predict(X)
  clustered = np.reshape(clustered, (-1, len(data)))

  # replace previous text with cluster index
  col_ind = 0
  for column in sorted(columns):
    data[column] = clustered[col_ind]
    col_ind += 1

  return data


def preprocess(collab, work, edu, advs, prods, stop_words=[]):
  # drop rows with no collaborations
  data = collab[collab['Colaboracoes'] != 0]

  # drop work rows with missing
  # vals and join to running data
  work = drop_if_missing(work)
  data = data.join(work, how='inner')

  # coerce numerical types in edu and
  # drop rows with missing values,
  # except post-doc and specialization,
  # which can be NaN
  for col in edu.columns:
    if col in ('inicio', 'inicio.1', 'inicio.2', 'fim', 'fim.1', 'fim.2'):
      edu[col] = pd.to_numeric(edu[col], errors='coerce')
  for column in edu.columns:
    if column != 'pos-doutorado' and column != 'especializacao':
      edu = edu[~pd.isna(edu[column])]

  # join to running data
  data = data.join(edu, how='inner')

  # join advisees data to running data
  data = data.join(advs, how='inner')

  # remove rows with no scientific
  # production and join to running data
  prods = prods[(prods != 0).any(axis=1)]
  data = data.join(prods, how='inner')

  # since there is high variability in how users
  # specify places and courses in their CVs, we
  # cluster them  with LSA + K-Means

  # cluster places
  places = [col for col in data.columns
            if 'local' in col] + ['Instituicao Atual']
  data = cluster_text(
      data, columns=places, n_clusters=3000, stop_words=stop_words)

  # cluster higher education
  courses = [
      'doutorado', 'graduacao', 'especializacao', 'mestrado', 'pos-doutorado'
  ]
  data = cluster_text(
      data, columns=courses, n_clusters=500, stop_words=stop_words)

  # compute collaborations probabilities
  collab = data['Colaboracoes']
  total = len(collab)
  collab_prob = [np.sum(collab == x) / total for x in np.unique(collab)]

  # compute mutual information between features
  # and discard those that are independent from
  # collaborations
  all_cols = []
  mis = []
  for column in sorted(data.columns):
    if column != 'Colaboracoes':
      # compute mutual information
      mi = utils.mutual_information(
          collab, data[column], X_marginal=collab_prob)

      all_cols.append(column)
      mis.append(mi)

      # discard independent features
      if np.isclose(mi, 0):
        data = data.drop(columns=column)

  return data, mis, all_cols


def main():
  # use provided random seed for derandomization
  np.random.seed(FLAGS.seed)

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
  # discarding last update date
  print('Loading scientific production data...')
  prods_path = os.path.join(FLAGS.data_path, 'Producao_Cientifica.csv')
  prods = pd.read_csv(
      prods_path,
      index_col='Identificador',
      sep=';',
      usecols=lambda x: 'Ultima' not in x)
  print('Loaded.')

  # load higher education institutions
  print('Loading portuguese stop words...')
  stop_words_path = os.path.join(FLAGS.data_path, 'stop_words.txt')
  stop_words = [w.strip() for w in open(stop_words_path, 'r')]
  print('Loaded.')

  print('Processing data...')
  data, mis, all_cols = preprocess(collab, work, edu, advs, prods, stop_words)
  print('Processed.')

  # plot
  if FLAGS.plot_path is not None:
    import matplotlib.pyplot as plt
    plot = plt.figure()
    plt.title('Informacao mutua com "Colaboracoes"')
    ind = np.arange(len(all_cols))
    plt.bar(ind, mis)
    plt.xticks(ind, all_cols, fontsize=7, rotation='vertical')
    plot.savefig(FLAGS.plot_path, bbox_inches='tight')

  # print mutual informations
  for mi, col in zip(mis, all_cols):
    print('I(Colaboracoes; {}) = {}'.format(col, mi))

  # filter collaborations graph with
  # with only remaining indices
  rem_inds = set(data.index.values)
  edges = []
  n_collab = np.zeros(265188, dtype=np.int32)
  print('Filtering collaborations graph...')
  for chunk in pd.read_csv(
      collab_path,
      sep=';',
      chunksize=FLAGS.chunk_sz,
      dtype=np.int32,
      usecols=[0, 1, 3]):
    for _, (u, v, w) in chunk.iterrows():
      # only keep remaining vertices
      if u in rem_inds and v in rem_inds:
        # add edge to filtered graph
        edges.append((u, v, w))

        # update filtered degrees
        n_collab[u - 1] += w
        n_collab[v - 1] += w

  collab = pd.DataFrame(edges)
  print('Done.')

  # drop collaborations from preprocessed
  data = data.drop(labels='Colaboracoes', axis=1)

  # save processed data
  print('Saving results...')
  data_path = os.path.join(FLAGS.results_path, 'preprocessed.csv')
  collab_path = os.path.join(FLAGS.results_path, 'collaborations.csv')
  data.to_csv(data_path, sep=';')
  collab.to_csv(collab_path, sep=';')
  print('Saved.')


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_path', default='data', type=str, help='Path to dataset.')
  parser.add_argument(
      '--chunk_sz',
      default=1123456,
      type=int,
      help='Chunk size to read large collaborations file.')
  parser.add_argument(
      '--plot_path',
      default=None,
      type=str,
      help='Path to save mutual information plot.')
  parser.add_argument(
      '--results_path',
      default='data',
      type=str,
      help='Path to save resulting csvs.')
  parser.add_argument('--seed', type=int, help='random seed')
  FLAGS, _ = parser.parse_known_args()

  main()
