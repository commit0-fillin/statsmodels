import numpy as np

class Pca:
    """
    A basic class for Principal Component Analysis (PCA).

    p is the number of dimensions, while N is the number of data points
    """
    _colors = ('r', 'g', 'b', 'c', 'y', 'm', 'k')

    def __init__(self, data, names=None):
        """
        p X N matrix input
        """
        A = np.array(data).T
        n, p = A.shape
        self.n, self.p = (n, p)
        if p > n:
            from warnings import warn
            warn('p > n - intentional?', RuntimeWarning)
        self.A = A
        self._origA = A.copy()
        self.__calc()
        self._colors = np.tile(self._colors, int((p - 1) / len(self._colors)) + 1)[:p]
        if names is not None and len(names) != p:
            raise ValueError('names must match data dimension')
        self.names = None if names is None else tuple([str(x) for x in names])

    def getCovarianceMatrix(self):
        """
        returns the covariance matrix for the dataset
        """
        return np.cov(self.A.T)

    def getEigensystem(self):
        """
        returns a tuple of (eigenvalues,eigenvectors) for the data set.
        """
        cov_matrix = self.getCovarianceMatrix()
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    def getEnergies(self):
        """
        "energies" are just normalized eigenvectors
        """
        eigenvalues, _ = self.getEigensystem()
        total_energy = np.sum(eigenvalues)
        return eigenvalues / total_energy

    def plot2d(self, ix=0, iy=1, clf=True):
        """
        Generates a 2-dimensional plot of the data set and principle components
        using matplotlib.

        ix specifies which p-dimension to put on the x-axis of the plot
        and iy specifies which to put on the y-axis (0-indexed)
        """
        import matplotlib.pyplot as plt

        if clf:
            plt.clf()

        # Plot data points
        plt.scatter(self.A[:, ix], self.A[:, iy], c='b', alpha=0.5)

        # Plot principal components
        _, eigenvectors = self.getEigensystem()
        for i in range(2):
            plt.quiver(0, 0, eigenvectors[ix, i], eigenvectors[iy, i],
                       angles='xy', scale_units='xy', scale=1, color=self._colors[i])

        plt.xlabel(f'PC{ix+1}')
        plt.ylabel(f'PC{iy+1}')
        plt.title('2D PCA Plot')
        plt.axis('equal')
        plt.show()

    def plot3d(self, ix=0, iy=1, iz=2, clf=True):
        """
        Generates a 3-dimensional plot of the data set and principle components
        using mayavi.

        ix, iy, and iz specify which of the input p-dimensions to place on each of
        the x,y,z axes, respectively (0-indexed).
        """
        from mayavi import mlab

        if clf:
            mlab.clf()

        # Plot data points
        mlab.points3d(self.A[:, ix], self.A[:, iy], self.A[:, iz], scale_factor=0.1)

        # Plot principal components
        _, eigenvectors = self.getEigensystem()
        for i in range(3):
            mlab.quiver3d(0, 0, 0, 
                          eigenvectors[ix, i], eigenvectors[iy, i], eigenvectors[iz, i], 
                          color=self._colors[i], scale_factor=1)

        mlab.xlabel(f'PC{ix+1}')
        mlab.ylabel(f'PC{iy+1}')
        mlab.zlabel(f'PC{iz+1}')
        mlab.title('3D PCA Plot')
        mlab.show()

    def sigclip(self, sigs):
        """
        clips out all data points that are more than a certain number
        of standard deviations from the mean.

        sigs can be either a single value or a length-p sequence that
        specifies the number of standard deviations along each of the
        p dimensions.
        """
        if np.isscalar(sigs):
            sigs = np.full(self.p, sigs)
        elif len(sigs) != self.p:
            raise ValueError("sigs must be a scalar or have length equal to the number of dimensions")

        mean = np.mean(self.A, axis=0)
        std = np.std(self.A, axis=0)
        
        mask = np.all(np.abs(self.A - mean) <= sigs * std, axis=1)
        self.A = self.A[mask]
        self.n = self.A.shape[0]

    def project(self, vals=None, enthresh=None, nPCs=None, cumen=None):
        """
        projects the normalized values onto the components

        enthresh, nPCs, and cumen determine how many PCs to use

        if vals is None, the normalized data vectors are the values to project.
        Otherwise, it should be convertable to a p x N array

        returns n,p(>threshold) dimension array
        """
        if vals is None:
            vals = self.A

        eigenvalues, eigenvectors = self.getEigensystem()
        energies = self.getEnergies()
        cumulative_energy = np.cumsum(energies)

        if enthresh is not None:
            n_components = np.sum(energies > enthresh)
        elif nPCs is not None:
            n_components = min(nPCs, self.p)
        elif cumen is not None:
            n_components = np.sum(cumulative_energy <= cumen) + 1
        else:
            n_components = self.p

        projected = np.dot(vals - np.mean(vals, axis=0), eigenvectors[:, :n_components])
        return projected

    def deproject(self, A, normed=True):
        """
        input is an n X q array, where q <= p

        output is p X n
        """
        _, eigenvectors = self.getEigensystem()
        q = A.shape[1]
        
        if normed:
            mean = np.mean(self._origA, axis=0)
            deprojected = np.dot(A, eigenvectors[:, :q].T) + mean
        else:
            deprojected = np.dot(A, eigenvectors[:, :q].T)
        
        return deprojected.T

    def subtractPC(self, pc, vals=None):
        """
        pc can be a scalar or any sequence of pc indecies

        if vals is None, the source data is self.A, else whatever is in vals
        (which must be p x m)
        """
        if vals is None:
            vals = self.A
        
        if np.isscalar(pc):
            pc = [pc]
        
        _, eigenvectors = self.getEigensystem()
        
        mean = np.mean(vals, axis=0)
        centered_vals = vals - mean
        
        for i in pc:
            component = np.outer(np.dot(centered_vals, eigenvectors[:, i]), eigenvectors[:, i])
            centered_vals -= component
        
        return centered_vals + mean
