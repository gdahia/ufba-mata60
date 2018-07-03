import numpy as np
import pandas as pd

FLAGS = None


def train(cvs, collabs, binary=True):
  # generate positive examples
  instances = []
  labels = []
  edges = []
  for _, (u, v, w) in collabs.iterrows():
    # generate instances
    u_row = cvs.loc[[u]]
    v_row = cvs.loc[[v]]
    instance1 = np.squeeze(np.concatenate([u_row, v_row], axis=1))
    instance2 = np.squeeze(np.concatenate([v_row, u_row], axis=1))
    instances.append(instance1)
    instances.append(instance2)

    # generate corresponding labels
    labels.append(w)
    labels.append(w)

    # add edge to edges list
    edges.append((u, v))
    edges.append((v, u))

  # make queries fast to answer
  edges = set(edges)

  # generate negative examples
  n_neg_examples = len(instances) // 2
  for _ in range(n_neg_examples):
    # randomly choose non edge
    while True:
      # pick two random endpoints
      u, v = np.random.choice(len(cvs), 2, replace=False)

      # recover 'non_edge' endpoints' indices
      u_index = cvs.index.values[u]
      v_index = cvs.index.values[v]

      # check if it really is non edge
      if (u_index, v_index) not in edges:
        # generate negative instance
        u_row = cvs.loc[[u_index]]
        v_row = cvs.loc[[v_index]]
        instance1 = np.squeeze(np.concatenate([u_row, v_row], axis=1))
        instance2 = np.squeeze(np.concatenate([v_row, u_row], axis=1))
        instances.append(instance1)
        instances.append(instance2)

        # generate corresponding labels
        labels.append(0)
        labels.append(0)

        break


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
