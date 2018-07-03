import numpy as np
import pandas as pd

FLAGS = None


def train(cvs, collabs, binary=True):
  # generate examples from valid edge
  instances = []
  labels = []
  for _, (u, v, w) in collabs.iterrows():
    # generate instances
    u = cvs.loc[[u]]
    v = cvs.loc[[v]]
    instance1 = np.squeeze(np.concatenate([u, v], axis=1))
    instance2 = np.squeeze(np.concatenate([v, u], axis=1))
    instances.append(instance1)
    instances.append(instance2)

    # generate labels
    labels.append(w)
    labels.append(w)


def main():
  import os

  # derandomize, if 'FLAGS.seed' is not None
  np.random.seed(FLAGS.seed)

  # load cvs dataset
  cvs_path = os.path.join(FLAGS.data_path, 'preprocessed.csv')
  cvs = pd.read_csv(cvs_path, sep=';', index_col=0)

  # load collaborations
  collabs_path = os.path.join(FLAGS.data_path, 'collaborations.csv')
  collabs = pd.read_csv(collabs_path, sep=';', index_col=0)

  train(cvs, collabs)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--data_path', default='data', type=str, help='path to dataset')
  parser.add_argument(
      '--save_path',
      default='train',
      type=str,
      help='path to save resulting model')
  parser.add_argument('--seed', type=int, help='random seed')
  FLAGS = parser.parse_args()

  main()
