import csv


def read_lattes_csv(f, remove_header=True):
  data = list(csv.reader(f, delimiter=';'))
  if remove_header:
    data = data[1:]
  return data
