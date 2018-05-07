from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
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
  journals = defaultdict(list)
  confs = defaultdict(list)
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
    if subject_confs != 0 or subject_journals != 0:
      journals[subject_uf].append(subject_journals)
      confs[subject_uf].append(subject_confs)

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
      'nordeste': [],
      'norte': [],
      'sul': [],
      'centroeste': [],
      'sudeste': [],
  }
  journals_by_region = {
      'nordeste': [],
      'norte': [],
      'sul': [],
      'centroeste': [],
      'sudeste': [],
  }
  all_confs = []
  all_journals = []
  for region in regions:
    for uf in regions[region]:
      # update region statistics
      confs_by_region[region].extend(confs[uf])
      journals_by_region[region].extend(journals[uf])

      # update total statistics
      all_confs.extend(confs[uf])
      all_journals.extend(journals[uf])

  # print results
  conf_share = []
  conf_means = []
  journal_share = []
  journal_means = []
  id_share = []
  region_labels = []

  total_confs = sum(all_confs)
  total_journals = sum(all_journals)
  total_ids = len(all_confs)
  total_conf_mean = np.mean(all_confs)
  total_journal_mean = np.mean(all_journals)

  print('total:')
  print('conference papers = {}'.format(total_confs))
  print('journal papers = {}'.format(total_journals))
  print('researchers = {}'.format(total_ids))
  print('conference / researcher = {}'.format(total_conf_mean))
  print('journals / researcher = {}'.format(total_journal_mean))
  print()

  for region in sorted(regions):
    conf_mean = np.mean(confs_by_region[region])
    journal_mean = np.mean(journals_by_region[region])

    conf_share.append(sum(confs_by_region[region]))
    conf_means.append(conf_mean)
    journal_share.append(sum(journals_by_region[region]))
    journal_means.append(journal_mean)
    id_share.append(len(confs_by_region[region]))
    region_labels.append(region)

    print('region {}'.format(region))
    print('\tjournals = {}'.format(journal_share[-1]))
    print('\tconferences = {}'.format(conf_share[-1]))
    print('\tresearchers = {}'.format(id_share[-1]))
    print('\tconference / researcher = {}'.format(conf_mean))
    print('\tjournal / researcher = {}'.format(journal_mean))
    print('\tjournal share = {}'.format(journal_share[-1] / total_journals))
    print('\tconference share = {}'.format(conf_share[-1] / total_confs))
    print('\tresearcher share = {}'.format(id_share[-1] / total_ids))
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

  journals_boxplot = plt.figure()
  plt.title('Boxplot de publicacoes em journals por regiao')
  plt.boxplot([journals_by_region[region] for region in region_labels])
  plt.xticks(np.array(ind) + 1, region_labels)
  plt.show()
  journals_boxplot.savefig('journals_boxplot.pdf')

  confs_boxplot = plt.figure()
  plt.title('Boxplot de publicacoes em conferencias por regiao')
  plt.boxplot([confs_by_region[region] for region in region_labels])
  plt.xticks(np.array(ind) + 1, region_labels)
  plt.show()
  confs_boxplot.savefig('confs_boxplot.pdf')


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
