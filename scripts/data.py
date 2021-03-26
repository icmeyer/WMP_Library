import numpy as np
from numpy.random import multivariate_normal
from WMP_pole_only import WindowedMultipole

def pole_vec_to_complex(parameters):
    poles_res = parameters[0][0::2] + 1j*parameters[0][1::2]
    return poles_res

def pole_complex_to_vec(poles_res):
    n_poles = poles_res.shape[0]
    n_residues = poles_res.shape[1] - 1
    n_pars = n_poles*(n_residues+1)*2
    vec = np.zeros(n_pars)
    c_vec = poles_res.flatten()
    vec[0::2] = np.real(c_vec)
    vec[1::2] = np.imag(c_vec)
    return vec

# Building data for sampling problem
def get_single_pole_example_dict():
    # U-238 first resonance pole information
    E_min = 5
    E_max = 8
    Z_min = E_min**(1/2)
    Z_max = E_max**(1/2)
    n_windows = 1
    n_poles = 1
    fit_order = 0

    hdf_location = '/home/icmeyer/nuclear_data/WMP_Library/WMP_Library/092238.h5'
    u238 = WindowedMultipole.from_hdf5(hdf_location)
    wmpdata = u238.data[1]
    wmpdata = np.reshape(wmpdata, (n_poles, 4))
    windows = np.array([[1, 1]])
    broaden_poly = np.array([True])
    curvefit = np.zeros([n_windows, fit_order+1, 3])  

    info_dict = {'n_windows' : n_windows,
                 'windows' : windows,
                 'fit_order' : fit_order,
                 'broaden_poly' : broaden_poly,
                 'E_min' : E_min,
                 'E_max' : E_max,
                 'spacing' : (Z_max - Z_min)/n_windows,
                 'n_poles' : n_poles,
                 'wmpdata' : wmpdata,
                 'curvefit' : curvefit,
                 'fissionable' : 1,
                 'sqrtAWR' : 15.3624802
                }
        
    return info_dict


def create_wmp_object(info_dict, name):
    wmp_object = WindowedMultipole(name)
    wmp_object.spacing = info_dict['spacing']
    wmp_object.sqrtAWR = info_dict['sqrtAWR']
    wmp_object.E_min = info_dict['E_min']
    wmp_object.E_max = info_dict['E_max']
    wmp_object.data = info_dict['wmpdata']
    wmp_object.windows = info_dict['windows']
    wmp_object.broaden_poly = info_dict['broaden_poly']
    wmp_object.curvefit = info_dict['curvefit']
    return wmp_object

class pole_covariance:
    """
    Class for handling the pole covariance information, mostly needed to handle
    the sampling of complex values and formatting their output
    """
    def __init__(self, poles_res, percent_err=0, covariance=None):

        vec = pole_complex_to_vec(poles_res)
        if percent_err > 0:
            sd_vec = vec*percent_err*0.01
            var_vec = sd_vec*sd_vec
            cov = np.diag(var_vec)
        elif covariance is not None:
            cov = covariance
        else:
            raise ValueError('Must set covariance through percent error or\
                              inputting covariance')
        
        self.poles_res = poles_res
        self.mean = vec
        self.cov = cov

    def __call__(self):
        sample_parameters = multivariate_normal(self.mean, self.cov, size=1)
        return pole_vec_to_complex(sample_parameters)

    def sample(self, mean=False, cov=False):
        if mean==False:
            mean = self.mean
        if cov==False:
            cov = self.cov

        sample_parameters = multivariate_normal(mean, cov, size=1)
        return pole_vec_to_complex(sample_parameters)
        

if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    # Check fixed value
    one_pole_info = get_single_pole_example_dict()
    wmp_object = create_wmp_object(one_pole_info, 'single_pole')
    hdf_location = '/home/icmeyer/nuclear_data/WMP_Library/WMP_Library/092238.h5'
    u238 = WindowedMultipole.from_hdf5(hdf_location)

    E_min = 6.4
    E_max = 7.0
    E_fixed = [6.6]
    T_low = 0
    print('Custom')
    print(wmp_object(E_fixed, T_low, curve_flag=True))
    print('\nFull Eval')
    print(u238(E_fixed, T_low, curve_flag=True))
    
    # Plotting WMP data
    energies = np.linspace(E_min, E_max-0.0001*E_max, 1000)
    xs_array = wmp_object(energies, T_low, curve_flag=True)
    # Indexing is [total, absorption, fission]
    rxn = 2
    plt.plot(energies, xs_array[rxn], label='custom')
    plt.plot(energies, u238(energies, T_low, curve_flag=True)[rxn], label='hdf5')
    plt.legend()
    plt.show()
