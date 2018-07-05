import numpy as np
from sklearn.ensemble import RandomForestClassifier

FLAGS = None


def train(instances, labels, binary=True):
  # convert labels to binary, if required
  if binary:
    # convert labels to np array
    labels = np.array(labels)

    # squash non-zero labels to label 1
    labels = np.array(labels != 0, dtype=np.uint8)

  # create random forest classifier model
  model = RandomForestClassifier()
  model.fit(instances, labels)

  return model


def main():
  import os
  import pickle

  # use provided random seed for derandomization
  np.random.seed(FLAGS.seed)

  # load training dataset
  print('Loading training dataset...')
  instances = None
  labels = None
  with open(FLAGS.data_path, 'rb') as dataset:
    instances, labels = pickle.load(dataset)
  print('Done')

  # create classifier model
  print('Creating classifier model...')
  model = train(instances, labels, binary=FLAGS.binary)
  print('Done')

  # save file
  print('Saving model...')
  model_path = os.path.join(FLAGS.save_path, 'model.pkl')
  with open(model_path, 'wb') as output:
    pickle.dump(model, output, -1)
  print('Done')


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--data_path', required=True, type=str, help='path to training dataset')
  parser.add_argument(
      '--save_path',
      default='data',
      type=str,
      help='path to save resulting model')
  parser.add_argument('--seed', type=int, help='random seed')
  parser.add_argument(
      '--binary',
      action='store_true',
      help='use this flag to perform binary classification')
  FLAGS = parser.parse_args()

  main()
