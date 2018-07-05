import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FLAGS = None


def add_reverse_instances(instances, labels, n_features):
  rev_instances = []
  for i, instance in enumerate(instances):
    # split into subjects
    fst = instance[:n_features]
    snd = instance[n_features:]

    # add reverse instance
    instance = np.squeeze(np.concatenate([snd, fst]))
    rev_instances.append(instance)

    # generate corresponding label
    labels.append(labels[i])

  # unify reverse instances and original instances
  instances.extend(rev_instances)

  return instances, labels


def create_dataset(cvs, collabs, test_size):
  # number of features
  n_features = len(cvs.columns)

  # generate positive examples
  instances = []
  labels = []
  edges = set()
  for _, (u, v, w) in collabs.iterrows():
    # generate instance
    u_row = cvs.loc[[u]]
    v_row = cvs.loc[[v]]
    instance = np.squeeze(np.concatenate([u_row, v_row], axis=1))
    instances.append(instance)

    # generate corresponding label
    labels.append(w)

    # add edge and its reverse to edges list
    edges.add((u, v))
    edges.add((v, u))

  # generate negative examples
  n_neg_examples = len(instances) // 2
  for _ in range(n_neg_examples):
    # randomly choose non-edge
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
        instance = np.squeeze(np.concatenate([u_row, v_row], axis=1))
        instances.append(instance)

        # generate corresponding label
        labels.append(0)

        # add non-edge to edges
        edges.add((u_index, v_index))
        edges.add((v_index, u_index))

        break

  # convert instance to sklearn compatible format
  instances = np.asarray(instances)
  for index in range(np.shape(instances)[1]):
    if isinstance(instances[0, index], str):
      string_col = instances[:, index]
      _, instances[:, index] = np.unique(string_col, return_inverse=True)
  instances = list(instances)

  # generate binary labels for stratification
  binary_labels = np.array(labels)
  binary_labels = np.array(binary_labels != 0, dtype=np.uint8)

  # split dataset into train/test
  instances_train, instances_test, labels_train, labels_test = train_test_split(
      instances, labels, test_size=test_size, stratify=binary_labels)

  # add reverse instances to datasets
  instances_train, labels_train = add_reverse_instances(
      instances_train, labels_train, n_features)
  instances_test, labels_test = add_reverse_instances(instances_test,
                                                      labels_test, n_features)

  return (instances_train, labels_train), (instances_test, labels_test)


def main():
  import os
  import pickle

  # derandomize, if 'FLAGS.seed' is not None
  np.random.seed(FLAGS.seed)

  # load cvs dataset
  print('Loading preprocessed data...')
  cvs_path = os.path.join(FLAGS.data_path, 'preprocessed.csv')
  cvs = pd.read_csv(cvs_path, sep=';', index_col=0)

  # load collaborations
  collabs_path = os.path.join(FLAGS.data_path, 'collaborations.csv')
  collabs = pd.read_csv(collabs_path, sep=';', index_col=0)
  print('Done')

  print('Creating dataset...')
  train, test = create_dataset(cvs, collabs, FLAGS.test_size)
  print('Done')

  # save train dataset
  print('Saving training dataset...')
  train_path = os.path.join(FLAGS.save_path, 'train.pkl')
  with open(train_path, 'wb') as output:
    pickle.dump(train, output, -1)
  print('Done')

  # save test dataset
  print('Saving testing dataset...')
  test_path = os.path.join(FLAGS.save_path, 'test.pkl')
  with open(test_path, 'wb') as output:
    pickle.dump(test, output, -1)
  print('Done')


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--data_path', default='data', type=str, help='path to dataset')
  parser.add_argument(
      '--save_path',
      default='data',
      type=str,
      help='path to save resulting dataset')
  parser.add_argument(
      '--test_size',
      default=0.4,
      type=float,
      help='share of dataset for testing')
  parser.add_argument('--seed', type=int, help='random seed')
  FLAGS = parser.parse_args()

  main()
