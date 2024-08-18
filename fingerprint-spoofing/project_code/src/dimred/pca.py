from numpy.linalg import linalg

from utilities.utilities import vcol


class PCA:
    """
    Class to perform PCA on data.

    Number of components is to specify, otherwise it is forced to 2 for plot-friendly usage.
    """
    def __init__(self, n_components=2):
        """
        Constructor for PCA class

        :param n_components: number of components to use (default is 2)
        """
        self.n_components = n_components
        self.P = None

    def set_params(self, **kwargs):
        self.n_components = kwargs.get('n_components', self.n_components)

    def fit(self, D, **kwargs):
        """
        Computes PCA parameters with specified dimensions (default is 2)

        :param D: dataset on which compute parameters
        """
        self.set_params(**kwargs)

        mu = D.mean(axis=1)
        DC = D - vcol(mu)
        C = DC @ DC.T / DC.shape[1]

        s, U = linalg.eigh(C)
        self.P = U[:, ::-1][:, 0:self.n_components]

    def transform(self, D):
        """
        Transforms dataset using PCA, by projecting over PCA parameters

        :param D: dataset to transform
        :raise: ValueError if a previous fit has not been performed
        """
        if self.P is None:
            raise ValueError("Missing PCA parameters")
        return self.P.T @ D

    def fit_transform(self, D, **kwargs):
        """
        Performs fit and transform operations in sequence, on the same dataset

        :param D: dataset to transform
        :return: transformed dataset
        """
        self.fit(D, **kwargs)
        return self.transform(D)
