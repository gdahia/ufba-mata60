import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def stats():
  # compute resources required for statistics
  # online, while loading collaborations graph
  collab_path = os.path.join(FLAGS.data_path, 'Colaboracoes.csv')
  n_collab = np.zeros(265187, dtype=np.int32)
  edges = []
  for chunk in pd.read_csv(
      collab_path,
      sep=';',
      chunksize=FLAGS.chunk_size,
      dtype=np.int32,
      usecols=[0, 1, 3]):
    for _, (u, v, w) in chunk.iterrows():
      n_collab[u - 1] += w
      n_collab[v - 1] += w
      edges.append(w)

  # compute degree statistics
  mean_degree = np.mean(n_collab)
  var_degree = np.var(n_collab)
  print('E[degree] = {}'.format(mean_degree))
  print('Var[degree] = {}'.format(var_degree))
  print()

  # plot degree histogram
  plot_path = os.path.join(FLAGS.results_path, 'degree_histogram.pdf')
  plot = plt.figure()
  plt.title('Histograma de colaboracoes por curriculo')
  plt.hist(n_collab, bins=len(np.unique(n_collab)) // 10, log=True)
  plot.savefig(plot_path, bbox_inches='tight')

  # compute active degree statistics
  active_collabs = n_collab[np.nonzero(n_collab)]
  mean_active_degree = np.mean(active_collabs)
  var_active_degree = np.var(active_collabs)
  print('#active collaborators = {}'.format(len(active_collabs)))
  print('E[degree|active] = {}'.format(mean_active_degree))
  print('Var[degree|active] = {}'.format(var_active_degree))
  print()

  # plot active degree histogram
  plot_path = os.path.join(FLAGS.results_path, 'active_degree_histogram.pdf')
  plot = plt.figure()
  plt.title('Histograma de colaboracoes por curriculo ativo')
  plt.hist(active_collabs, bins=len(np.unique(active_collabs)) // 10, log=True)
  plot.savefig(plot_path, bbox_inches='tight')

  # compute weight statistics
  print('E[w] = {}'.format(np.mean(edges)))
  print('Var[w] = {}'.format(np.var(edges)))

  # plot weight histogram
  plot_path = os.path.join(FLAGS.results_path, 'weight_histogram.pdf')
  plot = plt.figure()
  plt.title('Histograma da produtividade por colaboracao')
  plt.hist(edges, bins=len(np.unique(edges)) // 10, log=True)
  plot.savefig(plot_path, bbox_inches='tight')


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_path', default='.', type=str, help='Path to dataset.')
  parser.add_argument(
      '--chunk_size',
      default=112345,
      type=int,
      help='Chunk size to read large collaborations file.')
  parser.add_argument(
      '--results_path',
      default='.',
      type=str,
      help='Path to save resulting plots.')
  FLAGS, _ = parser.parse_known_args()

  stats()
