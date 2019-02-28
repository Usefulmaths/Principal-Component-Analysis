import numpy as np

class PCA(object):
    '''
    A class representing a dimensionality reduction method
    known as PCA.
    '''
    def __init__(self, number_of_components):
        self.number_of_components = number_of_components
    

    def fit(self, X):
        # Calculate the covariance between each feature
        covariance_matrix = np.cov(X.T)
        
        # Calculate the eigenvalues and eigenvectors
        values, vectors = np.linalg.eig(covariance_matrix)
        
        self.eigenvalues = values
        self.eigenvectors = vectors

        # Sort the eigenvectors in terms of highest eigenvalue
        sorted_indices = np.argsort(values)[::-1]
        sorted_vectors = vectors[sorted_indices]
        
        # Truncate eigenvector matrix to create projection 
        # matrix to the number_of_components.
        projection_matrix = sorted_vectors[:, :self.number_of_components] 

        # Calculate the principle components
        principle_components = np.dot(X, projection_matrix)
        
        return principle_components

    def variance_proportion(self):
        '''
        Calculates the proportion of information preserved
        through performing PCA with number_of_components
        principle components.
        '''

        proportion = sum(self.eigenvalues[:self.number_of_components]) / sum(self.eigenvalues)
        return proportion
