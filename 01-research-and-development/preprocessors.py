import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin



class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
  # Temporal elapsed time transformer

  def __init__(self, variables, reference_variable):

    if not isinstance(variables, list):
      raise ValueError('variable should be list')

    self.variables = variables
    self.reference_variable = reference_variable

  def fit(self, X, y=None):
    #v we need this step to fit the sklearn pipeline
    return self

  def transform(self, X):

    # so that we do not over-write the original dataframe
    X = X.copy()

    for feature in self.variables:
      X[feature] = X[self.reference_variable] - X[feature]

    return X



# categorical missing value imputer
class Mapper(BaseEstimator, TransformerMixin):

  def __init__(self, variables, mappings):

    if not isinstance(variables, list):
      raise ValueError('variables should be a list')

    self.variables = variables
    self.mappings = mappings

  def fit(self, X, y=None):
    # we need the fit statement to accomodate the sklearn pipeline
    return self

  def transform(self, X):
    X = X.copy()
    for feature in self.variables:
      X[feature] = X[feature].mmap(self.mappings)

    return X