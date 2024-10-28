from . import utils
from . import density_distribution as dd

class Surface_brightness:

    def __init__(self, data=0, shape = (2,2), extent = 10):
        self.data = data
        self.shape = shape
        self.extent = extent

    def deproject(self, bounds, initial_params, optimize_mask, sigma=1. ):
        return utils.deproject(self, bounds, initial_params, optimize_mask, sigma=1. )
    
    def generate(self,size,extent, e, p, q, rho0, s, a, b, i, phi, theta):
        den = dd.Density_distribution()
        den.generate(size,extent, e, p, q, rho0, s, a, b, i, phi, theta)
        data = utils.project(den)
        self.data = data
        self.shape = self.data.shape
        self.extent = extent