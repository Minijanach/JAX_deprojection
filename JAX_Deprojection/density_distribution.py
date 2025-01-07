from functools import partial

from jax import jit
from . import utils
from . import surface_brightness as sb

class Density_distribution:
    
    def __init__(self, data=0):
        self.data = data  
        #not rotated coordinates for img coord FIXME
        self.x_coord = 0
        self.y_coord = 0
   
    
    def generate(self,size,extent, e_k, e_d, p_a, p_b, p_c, p_d, q_a, q_b, q_c, q_d, rho0, s, a, b, i, phi, theta):
        x,y,z = utils.create_grid(size,extent)
        self.data = utils.rho(x, y, z, e_k, e_d, p_a, p_b, p_c, p_d, q_a, q_b, q_c, q_d, rho0, s, a, b, i, phi, theta)
        self.x_coord = x
        self.y_coord = y
     
    
    def project(self):
        data = utils.project(self)
        sb1 = sb.Surface_brightness(data)
        return sb1
    
