import numpy as np
import  matplotlib.pyplot as plt
from scipy.integrate import quad

import data
import WMP_pole_only

def flat_flux_fun(E):
    return 1

def overE_flux_fun(E):
    return 1/E

def evaluate_mg_xs_numeric(e_bins, xs_fun, T, fission=False, flux_choice='flat'):
    n_bins = e_bins.shape[0]-1
    # xs_fun  should be of the form [abs, scat, fission] = xs_fun(E, T)
    if flux_choice=='flat':
        flux_fun = flat_flux_fun
    elif flux_choice=='overE':
        flux_fun = overE_flux_fun

    else:
        raise ValueError('Flux choice \'{:s}\' not in available '
                         'fluxes'.format(flux_choice))

    abs_fun = lambda x : flux_fun(x) * xs_fun(x, T)[0]
    scat_fun = lambda x : flux_fun(x) * xs_fun(x, T)[1]

    if fission:
        n_rxn = 3
        fis_fun = lambda x : xs_fun(x, T)[2]
        functions = [abs_fun, scat_fun, fis_fun]
    else:
        n_rxn = 2
        functions = [abs_fun, scat_fun]


    mg_xs_array = np.zeros([n_bins, n_rxn])
    mg_xs_err = np.zeros([n_bins, n_rxn])
    for i in range(n_bins):
        for j in range(n_rxn):
            quad_result = quad(functions[j], e_bins[i], e_bins[i+1])
            mg_xs_array[i, j] = quad_result[0]
            mg_xs_err[i, j] = quad_result[1]

    abs_mg = mg_xs_array[:, 0]
    scat_mg = mg_xs_array[:,1]

    return mg_xs_array

def flat_flux_definite_integral_E(poles_res, e_range, rxn):
    total = 0
    for ipole in range(poles_res.shape[0]):
        p = poles_res[ipole][0]
        r = poles_res[ipole][1 + rxn]
        # Using i-less numerator, convert residues
        r = 1j*r

        E0 = e_range[0]
        E1 = e_range[1]
        inner_val = (r/p)*(2*np.log((p-E1**(1/2))/(p-E0**(1/2)))+np.log(E0/E1))
        total += np.real(inner_val)
    return total

def overE_definite_integral_E(poles_res, e_range, rxn):
    total = 0
    for ipole in range(poles_res.shape[0]):
        p = poles_res[ipole][0]
        r = poles_res[ipole][1 + rxn]
        # Using i-less numerator, convert residues
        r = 1j*r

        E0 = e_range[0]
        E1 = e_range[1]
        poly_part = (p**2/E1 + 2*p/E1**(1/2) - p**2/E0 - 2*p/E0**(1/2))
        inner_val = (r/p**3)*(poly_part + 2*np.log((p-E1**(1/2))/(p-E0**(1/2)))+np.log(E0/E1))
        total += np.real(inner_val)
    return total

def xs_test(pole_vec,z,rxn):
    # Pole parts
    c = pole_vec[0]
    d = pole_vec[1]
    # Residue parts
    a = -pole_vec[2*rxn+3]
    b = -pole_vec[2*rxn+2]
    return (1/z**2)*(a*(z-c)+b*d)/((z-c)**2+d**2)

def xs_test_2(pole_vec,z,rxn):
    # Pole parts
    c = pole_vec[0]
    d = pole_vec[1]
    # Residue parts - applying change to i-less sum
    a = -pole_vec[2*rxn+3]
    b = pole_vec[2*rxn+2]
    return ((1/z**2)*(a+1j*b)/(z-(c+1j*d))).real


def evaluate_mg_xs_analytic(e_bins, poles_res, T, rxn, flux='flat'):
    if flux=='flat':
        integral_fun = flat_flux_definite_integral_E
    elif flux=='overE':
        integral_fun = overE_definite_integral_E

    n_rxn = poles_res.shape[1] - 1
    n_groups = len(e_bins) - 1
    array = np.zeros([n_groups, n_rxn])
    for i in range(n_groups):
        for j in range(n_rxn):
            array[i, j] = integral_fun(poles_res, e_bins[i:i+2], j)
    return array

if __name__ == '__main__':
    one_pole_info = data.get_single_pole_example_dict()
    poles_res = one_pole_info['wmpdata']
    wmp_object = data.create_wmp_object(one_pole_info, 'single_pole')

    # flux_choice =  'flat'
    flux_choice =  'overE'
    rxn = 0 # for plots
    e_bins = np.array([6.3, 7.0])
    T = 0
    mg_xs_array = evaluate_mg_xs_numeric(e_bins, wmp_object, T, fission=True,
                                 flux_choice=flux_choice)

    mg_xs_array_analytic = evaluate_mg_xs_analytic(e_bins, poles_res, T,
                                                   rxn, flux=flux_choice)

    print(mg_xs_array)
    print(mg_xs_array_analytic)

    pole_vec = data.pole_complex_to_vec(poles_res)
    E_low = 6.3
    n_Es = 10000
    Es = np.linspace(E_low, 7.0, n_Es)
    zs = Es**(1/2)
    xs_test_vals = xs_test(pole_vec, zs, rxn)
    ff_integral = []
    for i in range(n_Es):
        ff_integral.append(flat_flux_definite_integral_E(poles_res, 
                                                         [E_low, Es[i]], rxn))
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(zs**2, ff_integral, label='analytic integral')
    ax1.plot(zs**2, xs_test_vals, label='xs_test_vals')
    ax1.plot(zs**2, xs_test_2(pole_vec, zs, rxn), label='xs_test_2')
    ax1.plot(zs**2, wmp_object(zs**2, T=0)[rxn], label='wmp xs')
    trap = []
    for i in range(len(zs)):
        trap.append(np.trapz(xs_test_vals[:i], x=zs[:i]**2))
    ax2.plot(zs**2, trap, label='numerical integral')
    diff = np.array(trap)-np.array(ff_integral)
    ax2.plot(zs**2, diff, label='numerical - analytical')
    
    ax3 = fig.add_subplot(313, sharex=ax1)
    ratio = np.array(trap)/np.array(ff_integral)
    ax3.plot(zs**2, ratio, label='numerical/analytical')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()

