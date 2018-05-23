import csv
import numpy as np

def read_lattes_csv(f, remove_header=True):
  data = list(csv.reader(f, delimiter=';'))
  if remove_header:
    data = data[1:]
  return data

def lcs(a, b):
    size_a = len(a)
    size_b = len(b)
    dp = np.zeros((size_a + 1, size_b + 1), dtype = int)
    longest = 0
    for i in range(1, size_a + 1):
        for j in range(1, size_b + 1):
            if a[i - 1] == b [ j -1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
    return longest
