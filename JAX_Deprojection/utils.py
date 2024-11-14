from functools import partial

import jax
from . import density_distribution
from . import surface_brightness
import jax.numpy as jnp
from jax import grad, jit , vmap
from jax.scipy.spatial.transform import Rotation as R
from jaxopt import ScipyBoundedMinimize
import numpy as np





#@partial(jit, static_argnames=['size','num_opt','num_fixed'])
def deproject(true_image, size,extent, bounds, initial_params, optimize_mask, num_opt, num_fixed, sigma=1.):
    """
    initial_params: Initial guess for parameters, including both known and unknown parameters. 
    Order :e, p, q, rho0, s, a, b, i, phi, theta
    bounds: List of tuples defining bounds for each unknown parameter.
    optimize_mask: A list of boolean values where True indicates a parameter should be optimized.
    sigma: Standard deviation for SSD calculation.
    """   

    

    # Split initial_guess and known_values based on the optimize_mask
    
    initial_params = jnp.asarray(initial_params)
    optimize_mask = jnp.asarray(optimize_mask)
    #num_opt = jnp.asarray(num_opt,int)
    #num_fixed = jnp.asarray(num_fixed,int)
    #bounds = jnp.asarray(bounds)
    true_image = jnp.asarray(true_image)

    #size_true = jnp.sum(optimize_mask)
    #size_true = jnp.asarray(size_true)
    #size_false = jnp.size(optimize_mask)-size_true
    #size_false = jnp.asarray(size_false)

    # set false values nan and then filter those
    arr_true = jnp.where(optimize_mask,initial_params,jnp.nan)
    #print("arr_true \n",arr_true)
    arr_false = jnp.where(~optimize_mask,initial_params,jnp.nan)
    #print("arr_false \n",arr_false)
    nonzero_jit = jax.jit(jnp.nonzero, static_argnames = "size")

    idx_true = nonzero_jit(~jnp.isnan(arr_true),size = num_opt)
    #print("idx_true \n",idx_true)
    idx_false = nonzero_jit(~jnp.isnan(arr_false),size = num_fixed)
    #print("idx_false \n",idx_false)

    initial_optimize_guess = arr_true[idx_true]
    fixed_params = arr_false[idx_false]


    params_history = [initial_optimize_guess[0]] # add to return also add flag if we want history

    x, y, z = create_grid(size,extent)

    
    def callback(xk):
        # Save the current optimized parameters at each step
        #full_params = combine_params(xk, fixed_params, optimize_mask)
        params_history.append(xk[0])

    #partial_ssd_score = partial(ssd_score_for_minimize,args =(size , extent, true_image, sigma, fixed_params, optimize_mask))

    #ssd_score_for_minimize_jit = jax.jit(partial_ssd_score)

    optimizer = ScipyBoundedMinimize(fun=ssd_score_for_minimize, method='L-BFGS-B', callback=callback)
    #print("Ini guess\n",initial_optimize_guess)
    #print("fixed params\n",fixed_params)
    # Run optimizer only on the parameters to be optimized
    result = optimizer.run(initial_optimize_guess, args=(x, y, z, true_image, sigma, fixed_params, optimize_mask), bounds=bounds)

    #print("res\n",result)

    # Combine optimized and fixed parameters for the final result
    final_params = combine_params(result[0], fixed_params, optimize_mask)

    #print("final params\n",final_params)

    predicted_density = density_distribution.Density_distribution()
    predicted_density.generate(size, extent, *final_params)

    final_score = ssd_score(true_image, predicted_density.project().data,1.)

    return predicted_density,params_history,final_score




def combine_params(optimized_params, fixed_params, optimize_mask):
    """
    Helper function to combine optimized and fixed parameters based on the optimize_mask.

    Parameters:
    optimized_params: jnp.ndarray - The parameters that are being optimized.
    fixed_params: jnp.ndarray - The parameters that are fixed.
    optimize_mask: jnp.ndarray - A boolean mask indicating which parameters to optimize.

    Returns:
    jnp.ndarray - A combined array of parameters.
    """
    optimize_mask = jnp.asarray(optimize_mask)
    optimized_params = jnp.asarray(optimized_params)
    fixed_params = jnp.asarray(fixed_params)


    # Generate indices for x and y where z is True and False, respectively
    optimized_params_indices = jnp.arange(len(optimized_params))    # Sequential indices for x
    fixed_params_indices = jnp.arange(len(fixed_params))    # Sequential indices for y

    # Create selection arrays based on cumulative sums to map x and y into positions of z
    optimized_params_selection = optimized_params_indices[jnp.cumsum(optimize_mask) - 1]
    fixed_params_selection = fixed_params_indices[jnp.cumsum(~optimize_mask) - 1]

    # Use jnp.where to build the final array based on the condition in z
    full_params = jnp.where(optimize_mask, optimized_params[optimized_params_selection], fixed_params[fixed_params_selection])

    return full_params

#@partial(jit, static_argnames=['args'])
def ssd_score_for_minimize(optimized_params, args):
    """
    Args:
    optimized_params: Parameters that are being optimized.
    args: Tuple containing the size, extent, true_image, sigma, fixed_params, optimize_mask.
    """
    x, y, z, true_image, sigma, fixed_params, optimize_mask = args

    # Combine optimized and fixed parameters
    full_params = combine_params(optimized_params, fixed_params, optimize_mask)

    # Unpack the full parameters
    e, p, q, rho0, s, a, b, i, phi, theta = full_params


    # Generate test density and project it
    data = rho(x, y, z, e, p, q, rho0, s, a, b, i, phi, theta)
    test_image = jnp.sum(data, axis = 2)

    # Calculate SSD
    ssd_value = ssd_score(true_image, test_image, sigma)

    return ssd_value

#def ssd_score_for_minimize_wrapper(optimized_params, args):
#    size = args[0]
#    other_args = args[1:]
#    return ssd_score_for_minimize(optimized_params, size , other_args)

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

#@partial(jit, static_argnames=['size','extent'])
def create_grid(size,extent):

   
    # Create 3D coordinates
    x, y, z = jnp.meshgrid(jnp.linspace(-extent, extent, jnp.astype(size,jnp.int32)),
                           jnp.linspace(-extent, extent, jnp.astype(size,jnp.int32)),
                           jnp.linspace(-extent, extent, jnp.astype(size,jnp.int32)), indexing = 'ij')
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