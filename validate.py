import numpy as np

FLAGS = None


def validate(model, instances, labels, binary=True):
  # convert labels to binary, if required
  if binary:
    # convert labels to np array
    labels = np.array(labels)

    # squash non-zero labels to label 1
    labels = np.array(labels != 0, dtype=np.uint8)

  # convert instance to sklearn compatible format
  instances = np.asarray(instances)
  for index in range(np.shape(instances)[1]):
    if isinstance(instances[0, index], str):
      string_col = instances[:, index]
      _, instances[:, index] = np.unique(string_col, return_inverse=True)

  return model.score(instances, labels)


def main():
  import pickle

  # use provided random seed for derandomization
  np.random.seed(FLAGS.seed)

  # load test set
  print('Loading test set...')
  instances = None
  labels = None
  with open(FLAGS.data_path, 'rb') as dataset:
    instances, labels = pickle.load(dataset)
  print('Done')

  # load classifier model
  print('Loading classifier model...')
  model = None
  with open(FLAGS.model_path, 'rb') as model_file:
    model = pickle.load(model_file)
  print('Done')

  print(validate(model, instances, labels, FLAGS.binary))


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()

  parser.add_argument('--seed', type=int, help='random seed')
  parser.add_argument(
      '--data_path', required=True, type=str, help='path to test set')
  parser.add_argument(
      '--model_path',
      required=True,
      type=str,
      help='path to trained classifier model')
  parser.add_argument(
      '--binary',
      action='store_true',
      help='use this flag to perform binary classification')
  FLAGS = parser.parse_args()

  main()
