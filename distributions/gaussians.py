#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

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
            self.mu = np.array([mu])
            self.dim = self.mu.shape[1]

        if type(sigma) not in [int, float]:
            self.sigma = np.array(sigma)
            if np.linalg.det(self.sigma) <= 0:
                raise ValueError('The covariance matrix \
                            must be positive definite.')
        
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
        if self.dim == 1:
            delta = x - self.mu
            normal_constant = 1 / np.sqrt(2 * np.pi * self.sigma ** 2)
            exponent = -delta ** 2 / (2 * self.sigma ** 2)
        else:
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
            raise ValueError('Cannot plot a 1D Gaussian distribution in 3D.')
        if self.dim != 2:
            raise ValueError(f'Cannot plot a Gaussian \
                        distribution in {self.dim}D.')
        # Generate a grid of points
        x_min = self.mu[0, 0] - 3 * self.sigma[0, 0]
        x_max = self.mu[0, 0] + 3 * self.sigma[0, 0]
        y_min = self.mu[0, 1] - 3 * self.sigma[1, 1]
        y_max = self.mu[0, 1] + 3 * self.sigma[1, 1]
        x_range = np.linspace(x_min, x_max, grid_resolution)
        y_range = np.linspace(y_min, y_max, grid_resolution)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        evaluation_points = np.row_stack([x_grid.flatten(), y_grid.flatten()])

        # Evaluate the PDF on the grid
        pdf_grid = self.pdf(evaluation_points)
        pdf_values = pdf_grid.reshape(
                                pdf_grid.diagonal(), 
                                (grid_resolution, grid_resolution))

        # Plot the PDF
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('Multivariate Gaussian Distribution')
        ax.plot_surface(x_grid, y_grid, pdf_values, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('pdf')
        plt.show()

if __name__ == "__main__":
    # Generate data
    mu = [0, 3]
    sigma = [[1, 0.5], [0.5, 1]]

    gaussian1 = GaussianDistribution(mu, sigma)

    gaussian1.plot_3d()