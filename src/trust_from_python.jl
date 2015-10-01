# adapted from Kona's trust in util.py:
#   https://github.com/OptimalDesignLab/Kona/blob/master/src/kona/linalg/solvers/util.py
module Trust

export trust

  # Note ordering of arguments, different from matlab's trust.m
  function trust(H,g,delta)
