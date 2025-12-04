"""
    Dataset class for handling numerical features and binary labels.
    
    This module provides a Dataset class that encapsulates a dataset with numerical features
    and binary labels. It includes methods for data retrieval, splitting into training and testing sets,
    and reducing the dataset size.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    """
    Class representing the dataset with numerical features and binary labels.
    
    Attributes:
        features (np.ndarray): A 2D numpy array where each column represents a data point and each row represents a feature.
        labels (np.ndarray): A 1D numpy array containing binary labels (0 or 1) for each data point.
        size (int): The number of data points in the dataset.
        dim (int): The number of features for each data point.
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features: np.ndarray = features
        self.labels: np.ndarray = labels
        self.size: int = self.features.shape[1]
        self.dim: int = self.features.shape[0]
        
    def get_data(self) -> np.ndarray:
        """
        Returns the feature matrix of the dataset.

        :return: feature matrix
        """
        return self.features
    
    def get_labels(self) -> np.ndarray:
        """
        Returns the label vector of the dataset.

        :return: label vector
        """
        return self.labels
    
    def split(self, ratio: float) -> tuple['Dataset', 'Dataset']:
        """
        Splits the dataset into training and testing sets based on the given ratio

        :param ratio: ratio of the training set size to the total dataset size
        :return trainset: Dataset instance for the training set
        :return testset: Dataset instance for the test set
        """
        
        indices_train, indices_test = train_test_split(
            np.arange(self.size), train_size=ratio, random_state=0, stratify=self.labels
        )
        
        train_features = self.features[:, indices_train]
        train_labels = self.labels[indices_train]
        test_features = self.features[:, indices_test]
        test_labels = self.labels[indices_test]
        
        return Dataset(train_features, train_labels), Dataset(test_features, test_labels)
    
    def reduce(self, ratio: float, seed: int = 0) -> 'Dataset':
        """
        Reduces the dataset to a specified size.

        :param ratio: ratio of the reduced dataset size to the original dataset size
        :param seed: random seed (default 0)
        :return: reduced Dataset instance
        """
        reduced_indices, _ = train_test_split(
            np.arange(self.size), train_size=ratio, random_state=seed, stratify=self.labels
        )
        
        reduced_features = self.features[:, reduced_indices]
        reduced_labels = self.labels[reduced_indices]
        
        return Dataset(reduced_features, reduced_labels)

    def __getitem__(self, index) -> 'Dataset':
        """
        Allows indexing to retrieve a subset of the dataset.
        
        :param index: index or slice to retrieve
        :return: Dataset instance containing the subset
        """
        if isinstance(index, tuple):
            assert len(index) == 2, "Indexing with tuple requires two elements"
            return Dataset(self.features[index[0], index[1]], self.labels[index[1]])
        return Dataset(self.features[index], self.labels[index])
        
    @staticmethod
    def create(path: str) -> 'Dataset':
        """
        Factory method to create a Dataset instance from a CSV file

        :param path: path to the CSV file
        :return: Dataset instance
        """
        df = pd.read_csv(path, header=None).to_numpy()
        features, labels = df[:, :-1].T.astype(np.float32), df[:, -1].astype(np.int32)
        
        return Dataset(features, labels)