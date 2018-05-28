import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def main():
  # load uf data from csv
  work_path = os.path.join(FLAGS.data_path, 'Atuacao_Profissional.csv')
  work = pd.read_csv(work_path, sep=';', index_col='Identificador')
  uf = work['UF']

  # make region dataframe
  ufs_by_region = {
      'nordeste': {'AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'},
      'norte': {'AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'},
      'sul': {'PR', 'RS', 'SC'},
      'centroeste': {'GO', 'MT', 'MS', 'DF'},
      'sudeste': {'ES', 'MG', 'SP', 'RJ'},
  }
  regions = [region for region in ufs_by_region] + ['nao-especificado']
  regions_by_uf = {
      uf: region
      for region, ufs in ufs_by_region.items() for uf in ufs
  }
  uf_to_region = lambda uf: regions_by_uf[uf] if uf in regions_by_uf else 'nao-especificado'
  region_df = uf.map(uf_to_region)

  # compute resources required for statistics
  # online, while loading collaborations graph
  collab_path = os.path.join(FLAGS.data_path, 'Colaboracoes.csv')
  edges_by_region = {region: [] for region in regions}
  inter_edges = {
      region1: {region2: []
                for region2 in regions if region1 < region2}
      for region1 in regions
  }
  for chunk in pd.read_csv(
      collab_path,
      sep=';',
      chunksize=FLAGS.chunk_size,
      dtype=np.int32,
      usecols=[0, 1, 3]):
    for _, (u, v, w) in chunk.iterrows():
      # add edges to regions
      u_region = region_df.loc[u]
      v_region = region_df.loc[v]
      edges_by_region[u_region].append(w)
      edges_by_region[v_region].append(w)

      # add edges to interregions
      if u_region < v_region:
        inter_edges[u_region][v_region].append(w)
      elif v_region < u_region:
        inter_edges[v_region][u_region].append(w)

  # print results
  for i, region in enumerate(sorted(regions)):
    print('{} = {} ({})'.format(region,
                                np.mean(edges_by_region[region]),
                                np.std(edges_by_region[region])))
    for region1 in sorted(regions)[i + 1:]:
      print('{} x {} = {} ({})'.format(region, region1,
                                       np.mean(inter_edges[region][region1]),
                                       np.std(inter_edges[region][region1])))

  # plot region results
  labels, edges = zip(*edges_by_region.items())
  plot_path = os.path.join(FLAGS.results_path, 'all.pdf')
  plot = plt.figure()
  plt.title('Colaboracoes por regiao')
  plt.boxplot(edges)
  plt.xticks(np.arange(1, len(labels) + 1), labels, rotation='vertical')
  plot.savefig(plot_path, bbox_inches='tight')

  # plot inter results
  for region in sorted(regions)[:-1]:
    labels, edges = zip(*inter_edges[region].items())
    plot_path = os.path.join(FLAGS.results_path, '{}.pdf'.format(region))
    plot = plt.figure()
    plt.title('Colaboracoes com "{}"'.format(region))
    plt.boxplot(edges)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation='vertical')
    plot.savefig(plot_path, bbox_inches='tight')


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_path', default='.', type=str, help='Path to dataset path.')
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

  main()
