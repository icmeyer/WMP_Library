import numpy as np

if __name__ == '__main__':
    import  matplotlib.pyplot as plt
    one_pole_info = data.get_single_pole_example_dict()
    poles_res = one_pole_info['wmpdata']
    wmp_object = data.create_wmp_object(one_pole_info, 'single_pole')


    flux_choice =  'overE'
    rxn = 0 # for plots
    e_bins = np.array([6.3, 7.0])
    T = 0
    mg_xs_array = evaluate_mg_xs(e_bins, wmp_object, T, fission=True,
                                 flux_choice=flux_choice)

    mg_xs_array_analytic = evaluate_mg_xs_analytic(e_bins, poles_res, T,
                                                   rxn, flux=flux_choice)

    print(mg_xs_array)
    print(mg_xs_array_analytic)

    pole_vec = data.pole_complex_to_vec(poles_res)
    Es = np.linspace(6.3, 7.0, 10000)
    zs = Es**(1/2)
    xs_test_vals = xs_test(pole_vec, zs, rxn)
    ff_integral = flat_flux_indefinite_integral_E(pole_vec, Es, rxn)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax2.plot(zs**2, ff_integral, label='analytic integral')
    ax1.plot(zs**2, xs_test_vals, label='xs_test_vals')
    ax1.plot(zs**2, xs_test_2(pole_vec, zs, rxn), label='xs_test_2')
    ax1.plot(zs**2, wmp_object(zs**2, T=0)[rxn], label='wmp xs')
    trap = []
    for i in range(len(zs)):
        trap.append(np.trapz(xs_test_vals[:i], x=zs[:i]**2))
    ax2.plot(zs**2, trap, label='numerical integral')
    ax2.plot(zs**2, trap - ff_integral, label='numerical - analytical')
    ax1.legend()
    ax2.legend()
    plt.show()
