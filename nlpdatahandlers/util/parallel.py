'''
util.py -- utilities for data loading
'''

def parallel_run(f, params):
      '''
      performs multi-core map of the function `f`
      over the parameter space spanned by parms.

      `f` MUST take only one argument.
      '''
      from multiprocessing import Pool

      pool = Pool()
      ret = pool.map(f, params)
      pool.close()
      pool.join()
      return ret