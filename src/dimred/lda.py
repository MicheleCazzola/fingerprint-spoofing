"""
    Linear Discriminant Analysis (LDA) implementation.
    
    This module provides a class for performing Linear Discriminant Analysis (LDA) for
    dimensionality reduction and classification tasks. It includes methods for fitting the LDA model,
    transforming data, and evaluating classification performance.
"""

import numpy as np
from src.dimred.pca import PCA
from numpy.linalg import linalg
from src.utils.utils import vcol, project

class LDA:
    """
    Class implementing Linear Discriminant Analysis (LDA) for dimensionality reduction and classification.
    
    Attributes:
        W (np.ndarray): LDA transformation matrix.
        pca (PCA, optional): PCA object for optional preprocessing.
    """
    
    def __init__(self):
        self.W = None
        self.pca: PCA | None = None
    
    def _joint_diag(self, SB, SW, m):
        """
        Computes LDA transformation matrix by using joint diagonalization, such as
        SB becomes diagonal and SW becomes the identity matrix

        :param SB: between-class covariance matrix
        :param SW: within-class covariance matrix
        :param m: LDA dimensions
        :return W: LDA transformation matrix
        """
        # Whitening transformation
        U1, s1, _ = linalg.svd(SW)
        P1 = U1 @ np.diag(1 / (s1 ** 0.5)) @ U1.T
        Sbt = P1 @ SB @ P1.T

        # Diagonalization
        s2, U2 = linalg.eigh(Sbt)
        P2 = U2[:, ::-1][:, 0:m]
        W = P1.T @ P2

        return W
    
    def _covariances(self, D, L):
        """
        Computes between-class covariance and within-class covariance matrices

        :param D: dataset
        :param L: labels
        :return SB: between-class covariance matrix
        :return SW: within-class covariance matrix
        """
        mu = D.mean(axis=1)
        SB = np.zeros((D.shape[0], D.shape[0]))
        SW = np.zeros((D.shape[0], D.shape[0]))
        for label in np.unique(L):
            D_label = D[:, L == label]
            mu_label = D_label.mean(axis=1)
            nc_label = D_label.shape[1]
            SB += nc_label * vcol(mu_label - mu) @ vcol(mu_label - mu).T
            SW += (D_label - vcol(mu_label)) @ (D_label - vcol(mu_label)).T

        return SB / D.shape[1], SW / D.shape[1]
    
    def _transform_matrix(self, SB, SW):
        """
        Computes LDA transformation matrix W, using joint diagonalization between
        covariance matrices SB and SW

        :param SB: between-class covariance matrix
        :param SW: within-class covariance matrix
        :return W: LDA transformation matrix
        """
        W = self._joint_diag(SB, SW, 1)

        return W
    
    def _prediction_function(self, data, threshold, mu0, mu1):
        """
        Assigns labels to data based on threshold and class means

        :param data: dataset
        :param threshold: threshold for classification
        :param mu0: mean of class 0
        :param mu1: mean of class 1
        :return predicted_labels: predicted labels
        """
        if mu1 > mu0:
            predicted_labels = (data > threshold).astype(int)
        else:
            predicted_labels = (data < threshold).astype(int)
        
        return predicted_labels
    
    def _error_rate(self, predicted_labels, true_labels):
        """
        Computes error rate between predicted labels and true labels

        :param predicted_labels: predicted labels
        :param true_labels: true labels
        :return: error rate
        """
        return np.mean(predicted_labels != true_labels)
    
    def fit(self, train_data, train_labels):
        """
        Estimates LDA transformation matrix from training data and labels
        
        :param train_data: training set
        :param train_labels: training labels
        :return W: LDA transformation matrix
        """
        
        SB, SW = self._covariances(train_data, train_labels)
        self.W = self._transform_matrix(SB, SW)
        return self.W
    
    def transform(self, data):
        """
        Projects data using estimated LDA transformation matrix
        
        :param data: data to project
        :return: projected data
        """
        if self.W is None:
            raise ValueError("LDA transformation matrix not estimated. Perform fit first.")
        return project(data, self.W)
    
    def fit_transform(self, train_data, train_labels):
        """
        Fits LDA transformation matrix and projects training data
        
        :param train_data: training set
        :param train_labels: training labels
        :return: projected training data
        """
        self.fit(train_data, train_labels)
        return self.transform(train_data)
    
    def classify(self, train_data, train_labels, test_data, test_labels, pca=None):
        """
        Classifies evaluation data using LDA projection and threshold computed from training data
        
        :param train_data: training set
        :param train_labels: training labels
        :param test_data: evaluation set
        :param test_labels: evaluation labels
        :param pca: PCA object for preprocessing (default None)
        :return predicted_labels: predicted labels for evaluation set
        :return error_rate: error rate of the classification
        :return threshold: classification threshold
        """
        
        if pca is not None:
            self.pca = pca
            
        if self.pca is not None:
            train_data = self.pca.fit_transform(train_data)
            test_data = self.pca.transform(test_data)
        
        self.fit(train_data, train_labels)
            
        projected_trainset = self.transform(train_data)
        projected_testset = self.transform(test_data)
        
        mu0 = projected_trainset[0, train_labels == 0].mean()
        mu1 = projected_trainset[0, train_labels == 1].mean()
        threshold = (mu0 + mu1) / 2
        
        predicted_labels = self._prediction_function(projected_testset, threshold, mu0, mu1)
        
        error_rate = self._error_rate(predicted_labels, test_labels)

        return predicted_labels, error_rate, threshold
    
    def classify_generalized_threshold(self, train_data, train_labels, test_data, test_labels):
        """
        Classifies evaluation data using LDA projection and threshold computed from training data
        
        :param train_data: training set
        :param train_labels: training labels
        :param test_data: evaluation set
        :param test_labels: evaluation labels
        :return thresholds: array of thresholds
        :return error_rates: array of error rates corresponding to thresholds
        :return red_thresholds: reduced array of thresholds within [-0.3, 0.3]
        :return red_error_rates: reduced array of error rates corresponding to reduced thresholds
        """
        
        self.fit(train_data, train_labels)
            
        projected_trainset = self.transform(train_data)
        projected_testset = self.transform(test_data)
        
        mu0 = projected_trainset[0, train_labels == 0].mean()
        mu1 = projected_trainset[0, train_labels == 1].mean()
        
        num_samples = int(1e5)
        er = np.zeros(num_samples)
        thresholds = np.linspace(projected_testset[0].min(), projected_testset[0].max(), num_samples)

        for c, t in enumerate(thresholds):
            predicted_labels = self._prediction_function(projected_testset, t, mu0, mu1)
            err_rate = self._error_rate(predicted_labels, test_labels)

            er[c] = err_rate

        mask = np.logical_and(thresholds >= -0.3, thresholds <= 0.3)
        red_th = thresholds[mask]
        red_er = er[mask]

        return (thresholds, er), (red_th, red_er)
    
    def classify_generalized_pca(self, train_data, train_labels, test_data, test_labels, pca_maxdim):
        """
        Computes the LDA classification with PCA preprocessing, using different dimensions

        :param train_data: training dataset
        :param train_labels: training labels
        :param test_data: validation dataset
        :param test_labels: validation labels
        :param pca_maxdim: maximum dimensionality for PCA
        :return dimensions: list of PCA dimensions used
        :return error_rates: error rates, depending on the dimensionality of the PCA
        """

        dimensions = list(range(pca_maxdim, 1, -1))
        error_rates = []
        for m in dimensions:
            pca = PCA(m)
            _, error_rate, _ = self.classify(train_data, train_labels, test_data, test_labels, pca)
            error_rates.append(error_rate)

        return dimensions, error_rates