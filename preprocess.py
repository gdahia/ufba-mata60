import pandas as pd
import numpy as np

import utils


def main(contr_data_path):
  # load contributions
  contr = utils.read_lattes_csv(open(contr_data_path, 'r'))

  # compute contributions' degrees
  degree = np.zeros(265187, dtype=np.int32)
  for v, u, _, w in contr:
    v = int(v)
    u = int(u)
    w = int(w)
    degree[v - 1] += w
    degree[u - 1] += w
  print(len(degree[degree != 0]))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--contr_data_path',
      default='Colaboracoes.csv',
      type=str,
      help='Path to contributions data csv.')
  flags, _ = parser.parse_known_args()

  main(flags.contr_data_path)
