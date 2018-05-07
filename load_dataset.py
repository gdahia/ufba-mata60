from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import matplotlib.pyplot as plt
from collections import defaultdict


def read_lattes_csv(f, remove_header=True):
  data = list(csv.reader(f, delimiter=';'))
  if remove_header:
    data = data[1:]
  return data


def main(prod_data_path, act_data_path):
  # load data from csvs
  prod_data = read_lattes_csv(open(prod_data_path, 'r'))
  act_data = read_lattes_csv(open(act_data_path, 'r'))

  # get 'unidade da federacao' (uf) for each id
  subject_ufs = dict()
  uf_count = defaultdict(int)
  for row in act_data:
    # retrieve subject id
    subject_id = int(row[0])

    # retrieve subject uf
    subject_uf = row[1]

    # make id-uf association
    subject_ufs[subject_id] = subject_uf

    # update uf count
    uf_count[subject_uf] += 1

  # get journal and conference publications by id
  journals = defaultdict(int)
  journals_sqr = defaultdict(int)
  for row in prod_data:
    # retrieve subject id
    subject_id = int(row[0])

    # retrieve journal pubs by subject
    subject_journals = int(row[3])

    # retrieve conf pubs by subject
    subject_confs = int(row[2])

    # retrieve subject uf
    subject_uf = subject_ufs[subject_id]

    # update journals and confs by uf
    journals[subject_uf] += subject_journals
    confs[subject_uf] += subject_confs

  # compute region publication share, number of
  # researchers and average publication stats
  regions = {
      'nordeste': {'AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'},
      'norte': {'AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'},
      'sul': {'PR', 'RS', 'SC'},
      'centroeste': {'GO', 'MT', 'MS', 'DF'},
      'sudeste': {'ES', 'MG', 'SP', 'RJ'}
  }
  confs_by_region = {
      'nordeste': 0,
      'norte': 0,
      'sul': 0,
      'centroeste': 0,
      'sudeste': 0,
  }
  journals_by_region = {
      'nordeste': 0,
      'norte': 0,
      'sul': 0,
      'centroeste': 0,
      'sudeste': 0,
  }
  ids_by_region = {
      'nordeste': 0,
      'norte': 0,
      'sul': 0,
      'centroeste': 0,
      'sudeste': 0,
  }
  total_confs = 0
  total_journals = 0
  total_ids = 0
  for region in regions:
    for uf in regions[region]:
      # update region statistics
      confs_by_region[region] += confs[uf]
      journals_by_region[region] += journals[uf]
      ids_by_region[region] += uf_count[uf]

      # update total statistics
      total_confs += confs[uf]
      total_journals += journals[uf]
      total_ids += uf_count[uf]

  # print results
  conf_share = []
  conf_means = []
  conf_stds = []
  journal_share = []
  journal_means = []
  journal_stds = []
  id_share = []
  region_labels = []

  total_conf_mean = total_confs / total_ids
  total_journal_mean = total_journals / total_ids

  print('total:')
  print('conference papers = {}'.format(total_confs))
  print('journal papers = {}'.format(total_journals))
  print('researchers = {}'.format(total_ids))
  print('conference / researcher = {}'.format(total_conf_mean))
  print('journals / researcher = {}'.format(total_journal_mean))
  print()
  for region in regions:
    conf_mean = confs_by_region[region] / ids_by_region[region]
    journal_mean = journals_by_region[region] / ids_by_region[region]

    conf_share.append(confs_by_region[region])
    conf_means.append(conf_mean)
    journal_share.append(journals_by_region[region])
    journal_means.append(journal_mean)
    id_share.append(ids_by_region[region])
    region_labels.append(region)

    print('region {}'.format(region))
    print('\tjournals = {}'.format(journals_by_region[region]))
    print('\tconferences = {}'.format(confs_by_region[region]))
    print('\tresearchers = {}'.format(ids_by_region[region]))
    print('\tconference / researcher = {}'.format(conf_mean))
    print('\tjournal / researcher = {}'.format(journal_mean))
    print('\tjournal share = {}'.format(journals_by_region[region] /
                                        total_journals))
    print('\tconference share = {}'.format(confs_by_region[region] /
                                           total_confs))
    print('\tresearcher share = {}'.format(ids_by_region[region] / total_ids))
    print()

  conf_share_plot = plt.figure()
  plt.title('Publicacoes em conferencias por regiao')
  plt.pie(conf_share, labels=region_labels, autopct='%1.1f%%')
  plt.show()
  conf_share_plot.savefig('conf_share.pdf')

  journal_share_plot = plt.figure()
  plt.title('Publicacoes em journals por regiao')
  plt.pie(journal_share, labels=region_labels, autopct='%1.1f%%')
  plt.show()
  journal_share_plot.savefig('journal_share.pdf')

  id_share_plot = plt.figure()
  plt.title('Curriculos por regiao')
  plt.pie(id_share, labels=region_labels, autopct='%1.1f%%')
  plt.show()
  id_share_plot.savefig('id_share.pdf')

  ind = list(range(len(conf_means)))
  conf_means_plot = plt.figure()
  plt.title('Media de publicacoes em conferencias por regiao')
  plt.bar(ind, conf_means)
  plt.xticks(ind, region_labels)
  plt.show()
  conf_means_plot.savefig('conf_means.pdf')

  journal_means_plot = plt.figure()
  plt.title('Media de publicacoes em journals por regiao')
  plt.bar(ind, journal_means)
  plt.xticks(ind, region_labels)
  plt.show()
  journal_means_plot.savefig('journal_means.pdf')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--prod_data',
      default='Producao_Cientifica.csv',
      type=str,
      help='Path to production data csv.')
  parser.add_argument(
      '--act_data',
      default='Atuacao_Profissional.csv',
      type=str,
      help='Path to actuation data csv.')
  flags, _ = parser.parse_known_args()

  main(flags.prod_data, flags.act_data)
