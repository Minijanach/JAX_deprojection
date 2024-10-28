from . import utils
from . import surface_brightness as sb

class Density_distribution:
    
    def __init__(self, data=0):
        self.data = data  
    
    def generate(self,size,extent, e, p, q, rho0, s, a, b, i, phi, theta):
        x,y,z = utils.create_grid(size,extent)
        self.data = utils.rho(x, y, z, e, p, q, rho0, s, a, b, i, phi, theta)
     
    
    def project(self):
        data = utils.project(self)
        sb1 = sb.Surface_brightness(data)
        return sb1
    
