#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#------------------------------------------------------------------------------
# Class for a Gaussian distribution
#  
# @author: Onur Bagoren
# @email: onurbagoren3@gmail.com
#------------------------------------------------------------------------------

class GaussianDistribution:
    def __init__(self, mu, sigma) -> None:
        self.mu = mu
        self.sigma = sigma

        if type(mu) in [int, float]:
            self.dim = 1
        else:
            self.mu = np.array(mu).reshape(len(mu), 1)
            self.dim = self.mu.shape[0]

        if type(sigma) not in [int, float]:
            self.sigma = np.array(sigma)
            if np.linalg.det(self.sigma) <= 0:
                err_msg = 'The covariance matrix must be positive definite.'
                raise ValueError(err_msg)

        if self.dim != self.sigma.shape[0]:
            err_msg = 'The dimension of the mean vector and ' \
                        'covariance matrix must be the same.'
            raise ValueError(err_msg)
        
    def pdf(self, x):
        '''
        Compute the probability density function of the Gaussian distribution
        at x.

        Parameters
        ----------
        x : float or (n, d) numpy array
            The point(s) at which to evaluate the density function.

        Returns
        -------
        float or (n, d) numpy array
            The value(s) of the density function at x.
        '''
        print(x.shape)
        if self.dim == 1:
            delta = x - self.mu
            normal_constant = 1 / np.sqrt(2 * np.pi * self.sigma ** 2)
            exponent = -delta ** 2 / (2 * self.sigma ** 2)
        else:
            if type(x) != np.ndarray:
                x = np.array(x).reshape(self.dim, len(x))
            delta = x - self.mu
            normal_constant = (2*np.pi) ** (self.dim/2) * \
                    np.linalg.det(self.sigma) ** 0.5
            exponent = -0.5 * delta.T @ np.linalg.solve(self.sigma, delta)
        
        return normal_constant * np.exp(exponent)
    

    def plot_3d( self, grid_resolution=100 ):
        '''
        Plot the Gaussian distribution in 3D.

        Returns
        -------
        None
        '''
        if self.dim == 1:
            raise ValueError('Cannot plot a 1-dim Gaussian distribution in 3D.')
        if self.dim != 2:
            raise ValueError(f'Cannot plot a Gaussian \
                        distribution in {self.dim}D.')
        # Generate a grid of points
        x_min = self.mu[0, 0] - 3 * self.sigma[0, 0]
        x_max = self.mu[0, 0] + 3 * self.sigma[0, 0]
        y_min = self.mu[1, 0] - 3 * self.sigma[1, 1]
        y_max = self.mu[1, 0] + 3 * self.sigma[1, 1]
        x_range = np.linspace(x_min, x_max, grid_resolution)
        y_range = np.linspace(y_min, y_max, grid_resolution)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        evaluation_points = np.row_stack([x_grid.flatten(), y_grid.flatten()])

        # Evaluate the PDF on the grid
        pdf_grid = self.pdf(evaluation_points)
        pdf_values = np.reshape(
                                pdf_grid.diagonal(), 
                                (grid_resolution, grid_resolution))

        # Plot the PDF
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig.suptitle('Multivariate Gaussian Distribution')
        ax.plot_surface(x_grid, y_grid, pdf_values, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('pdf')
    
    def plot_2d(self, grid_resolution=100):
        ''' 
        Plot the Gaussian distribution in 2D.

        Returns
        -------
        None
        '''
        if self.dim != 1:
            raise ValueError('Cannot plot a >1-dim Gaussian distribution in 2D.')
        # Generate a grid of points
        x_min = self.mu - 3 * self.sigma
        x_max = self.mu + 3 * self.sigma
        x_range = np.linspace(x_min, x_max, grid_resolution)
        # Evaluate the PDF on the grid
        pdf_grid = self.pdf(x_range)
        # Plot the PDF
        fig = plt.figure()
        fig.suptitle('Multivariate Gaussian Distribution')
        plt.plot(x_range, pdf_grid)
        plt.xlabel('x')
        plt.ylabel('pdf')

    
    def plot(self, grid_resolution=100):
        '''
        Plot the Gaussian distribution in 2D.

        Returns
        -------
        None
        '''
        if self.dim == 1:
            self.plot_2d(grid_resolution)
        else:
            self.plot_3d(grid_resolution)
    

    def sample(self, n_samples):
        '''
        Sample from the Gaussian distribution.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        (n_samples, d) numpy array
            The samples.
        '''


if __name__ == "__main__":
    # Generate data
    mu_1 = [0.1, 0.1]
    sigma_1 = np.eye(2)

    # Create a Gaussian distribution
    gaussian1 = GaussianDistribution(mu_1, sigma_1)

    # Plot the PDF
    gaussian1.plot()

    # Generate data
    mu_3 = [0.3, 0.3]
    sigma_3 = np.eye(2)

    # Create a Gaussian distribution
    gaussian3 = GaussianDistribution(mu_3, sigma_3)

    # Plot the PDF
    gaussian3.plot()
    plt.show()