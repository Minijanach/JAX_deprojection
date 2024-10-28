from . import density_distribution
from . import surface_brightness
import jax.numpy as jnp
from jax import grad , vmap
from jax.scipy.spatial.transform import Rotation as R
from jaxopt import ScipyBoundedMinimize

def deproject(self, bounds, initial_params, optimize_mask, sigma=1. ):
    """
    initial_params: Initial guess for parameters, including both known and unknown parameters. 
    Order :e, p, q, rho0, s, a, b, i, phi, theta
    bounds: List of tuples defining bounds for each unknown parameter.
    optimize_mask: A list of boolean values where True indicates a parameter should be optimized.
    sigma: Standard deviation for SSD calculation.
    """
    true_image = self.data
    size = self.shape[0]
    extent = self.extent

    params_history = [] # add to return also add flag if we want history

    # Split initial_guess and known_values based on the optimize_mask
    initial_optimize_guess = [param for param, opt in zip(initial_params, optimize_mask) if opt]
    fixed_params = [param for param, opt in zip(initial_params, optimize_mask) if not opt]

    
    def callback(xk):
        # Save the current optimized parameters at each step
        full_params = combine_params(xk, fixed_params, optimize_mask)
        params_history.append(full_params)

    optimizer = ScipyBoundedMinimize(fun=ssd_score_for_minimize, method='L-BFGS-B', callback=callback)

    # Run optimizer only on the parameters to be optimized
    result = optimizer.run(initial_optimize_guess, args=(size, extent, true_image, sigma, fixed_params, optimize_mask), bounds=bounds)

    # Combine optimized and fixed parameters for the final result
    final_params = combine_params(result, fixed_params, optimize_mask)

    predicted_density = density_distribution.Density_distribution()
    predicted_density.generate(size, extent, *final_params)

    return predicted_density


def combine_params(optimized_params, fixed_params, optimize_mask):
    """
    Helper function to combine optimized and fixed parameters based on the optimize_mask.
    """
    optimize_mask = jnp.asarray(optimize_mask)
    full_params = []
    opt_idx = 0
    fixed_idx = 0
    for opt in optimize_mask:
        if opt:
            full_params.append(optimized_params[opt_idx])
            opt_idx += 1
        else:
            full_params.append(fixed_params[fixed_idx])
            fixed_idx += 1
    return full_params


def ssd_score_for_minimize(optimized_params, args):
    """
    Args:
    optimized_params: Parameters that are being optimized.
    args: Tuple containing the size, extent, true_image, sigma, fixed_params, optimize_mask.
    """
    size, extent, true_image, sigma, fixed_params, optimize_mask = args

    # Combine optimized and fixed parameters
    full_params = combine_params(optimized_params, fixed_params, optimize_mask)

    # Unpack the full parameters
    e, p, q, rho0, s, a, b, i, phi, theta = full_params

    # Generate test density and project it
    test_density = density_distribution.Density_distribution.generate(size, extent, e, p, q, rho0, s, a, b, i, phi, theta)
    test_image = test_density.project()

    # Calculate SSD
    ssd_value = ssd_score(true_image, test_image, sigma)

    return ssd_value


def project(self):
    data = jnp.sum(self.data, axis = 2)
    return data

def ssd_score(image1, image2, sigma):
    
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Calculate SSD
    ssd_value = jnp.sum(((image1 - image2)/sigma)**2)
    
    return ssd_value

def create_grid(size,extent):
    # Create 3D coordinates
    x, y, z = jnp.meshgrid(jnp.linspace(-extent, extent, size),
                           jnp.linspace(-extent, extent, size),
                           jnp.linspace(-extent, extent, size), indexing = 'ij')
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    return x,y,z

def rotation(x,y,z,angle,axis):

    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = jnp.asarray(z)

    if(x.size!= y.size or x.size!= z.size or y.size!= z.size):
        raise ValueError("Input must have the same length!")

    if axis not in ['x', 'y', 'z']:
        raise ValueError("axis has to be either x , y or z!")
    
    rotation = R.from_euler(axis, angle, degrees=True)

    coordinates = jnp.array([x, y, z]).T
    
    rotated_coordinates = rotation.apply(coordinates)

    res = rotated_coordinates.T
    
    x_rot, y_rot, z_rot = res[0], res[1], res[2]

    return x_rot, y_rot, z_rot


def calculate_rho(x, y, z, e, p, q, rho0, s, a, b):
    r = ((jnp.abs(x)**(2 - e) + (jnp.abs(y) / p)**(2 - e) + (jnp.abs(z) / q)**(2 - e)))**(1 / (2 - e))
    den = rho0 / ((r / s)**a * (1 + (r / s))**(b - a))
    
    return den/(p*q)

def rho(x, y, z, e, p, q, rho0, s, a, b, i, phi, theta):
    # Rotate inputs to arrays
    x,y,z = rotation(x,y,z,i,'x')
    x,y,z = rotation(x,y,z,phi,'y')
    x,y,z = rotation(x,y,z,theta,'z')

    # Check if inputs are arrays or scalars
    if x.ndim == 0:  # Scalar input
        return calculate_rho(x, y, z, e, p, q, rho0, s, a, b)
    else:  # Array input
        # Check if inputs have the same length
        if x.size != y.size or x.size != z.size or y.size != z.size:
            raise ValueError("Input arrays must have the same length!")
        
        # Check if the length of inputs is reshapeable into a cube
        cube_len = int(round(x.size ** (1/3)))
        if cube_len ** 3 != x.size:
            raise ValueError("The length of the inputs must be reshapeable into a cube!")

        return calculate_rho(x, y, z, e, p, q, rho0, s, a, b).reshape((cube_len, cube_len, cube_len))
    

#------------------- Gradients -----------------------------



def _calculate_gradient_rho(x, y, z, e, p, q, rho0, s, a, b, i, phi, theta, arg_num):
    grad_rho = grad(rho, argnums=arg_num)
    return grad_rho(x, y, z, e, p, q, rho0, s, a, b, i, phi, theta)

def gradient_rho(x, y, z, e, p, q, rho0, s, a, b, i, phi, theta, arg_num):
    # Convert inputs to arrays
    x,y,z = rotation(x,y,z,i,'x')
    x,y,z = rotation(x,y,z,phi,'y')
    x,y,z = rotation(x,y,z,theta,'z')
    
    if not arg_num:
        raise ValueError("arg_num cannot be empty. Specify the arguments to differentiate with respect to.")
    
    # Check if inputs are arrays or scalars
    if x.ndim == 0:  # Scalar input
        return _calculate_gradient_rho(x, y, z, e, p, q, rho0, s, a, b, arg_num)
    else:  # Array input
        # Check if inputs have the same length
        if x.size != y.size or x.size != z.size or y.size != z.size:
            raise ValueError("Input arrays must have the same length!")
        
        # Define the function to compute the gradient
        grad_rho = grad(rho, argnums=arg_num)

        # Vectorize the gradient function
        vmap_grad_rho = vmap(lambda x, y, z: grad_rho(x, y, z, e, p, q, rho0, s, a, b), in_axes=(0, 0, 0))

        # Compute the gradients
        gradients = vmap_grad_rho(x, y, z)

        # Reshape and stack the gradients
        cube_len = int(round(x.size ** (1/3)))
        reshaped_gradients = [jnp.reshape(g, (cube_len, cube_len, cube_len)) for g in gradients]

        # Stack gradients along the last axis
        stacked_gradients = jnp.stack(reshaped_gradients, axis=-1)
        
        return stacked_gradients