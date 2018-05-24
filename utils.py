import numpy as np
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

def mutual_information(X, Y, X_marginal=None, Y_marginal=None):
  # compute prob dist support
  X_support = np.unique(X)
  Y_support = np.unique(Y)

  # precompute marginals' log
  total = len(X)
  if X_marginal is None:
    X_marginal = [np.sum(X == x) / total for x in X_support]
  log_X_marginal = np.log2(X_marginal)

  if Y_marginal is None:
    Y_marginal = [np.sum(Y == y) / total for y in Y_support]
  log_Y_marginal = np.log2(Y_marginal)

  # compute mutual information
  mi = 0
  for i, y in enumerate(Y_support):
    for j, x in enumerate(X_support):
      # compute joint p(x, y)
      p_xy = np.sum((X == x) & (Y == y)) / total

      # if p(x, y) -> 0, make summand 0
      if not np.isclose(p_xy, 0):
        # retrieve marginals' log from table
        log_p_x = log_X_marginal[j]
        log_p_y = log_Y_marginal[i]

        # add term to mutual information
        mi += p_xy * (np.log2(p_xy) - log_p_x - log_p_y)

  return mi