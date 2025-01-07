from . import utils
from . import density_distribution as dd

class Surface_brightness:

    def __init__(self, data=0, shape = (2,2), extent = 10):
        self.data = data
        self.shape = shape
        self.extent = extent
        self.x_coord = 0
        self.y_coord = 0

    def deproject(self,bounds, initial_params, optimize_mask,num_opt,num_fixed, sigma=1. ):
        true_image = self.data
        size = self.shape[0]
        extent = self.extent
        
        return utils.deproject(true_image, int(size), int(extent), bounds, initial_params, optimize_mask,num_opt,num_fixed, sigma )
    
    def generate(self,size,extent, e_k, e_d, p_a, p_b, p_c, p_d, q_a, q_b, q_c, q_d, rho0, s, a, b, i, phi, theta):
        den = dd.Density_distribution()
        den.generate(size,extent, e_k, e_d, p_a, p_b, p_c, p_d, q_a, q_b, q_c, q_d, rho0, s, a, b, i, phi, theta)
        data = utils.project(den)
        self.data = data
        self.shape = self.data.shape
        self.extent = extent
        self.x_coord = den.x_coord
        self.y_coord = den.y_coord