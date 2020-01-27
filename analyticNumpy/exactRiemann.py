import sod
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    gamma = 1.4
    dustFrac = 0.0
    npts = 256
    t = 0.2
    left_state = (1,1,0)
    right_state = (0.1, 0.125, 0.)

    # left_state and right_state set pressure, density and u (velocity)
    # geometry sets left boundary on 0., right boundary on 1 and initial
    # position of the shock xi on 0.5
    # t is the time evolution for which positions and states in tube should be
    # calculated
    # gamma denotes specific heat
    # note that gamma and npts are default parameters (1.4 and 500) in solve
    # function
    positions, regions, values = sod.solve(left_state=left_state, \
        right_state=right_state, geometry=(0., 1., 0.5), t=t,
        gamma=gamma, npts=npts, dustFrac=dustFrac)
    # Printing positions
    print('Positions:')
    for desc, vals in positions.items():
        print('{0:10} : {1}'.format(desc, vals))

    # Printing p, rho and u for regions
    print('Regions:')
    for region, vals in sorted(regions.items()):
        print('{0:10} : {1}'.format(region, vals))

    # Calculate variables e, tE, T, c, M, h
    e = values['p'] / (gamma - 1) / values['rho']
    E = values['p']/(gamma-1.) + 0.5*values['rho']*values['u']**2
    T = values['p'] / values['rho']
    c = np.sqrt(gamma * values['p'] / values['rho'])
    M = values['u'] / c
    h = e + values['p']/values['rho']

    # write data to file ("a" for add, "w" for write, "r" for read)
    myFile = open("analytic.dat", "w")
    myFile.write("Variables = x, rho, u, p, e, Et, T, c, M, h\n")
    for i in range(256):
        text = str(values['x'][i]) +' '+ str(values['rho'][i]) +' '+ str(values['u'][i]) +' '+ \
               str(values['p'][i]) +' '+ str(e[i]) + ' '+ str(E[i]) + ' '+ str(T[i]) + ' '+ \
               str(c[i]) + ' '+ str(M[i]) + ' '+ str(h[i]) + '\n'
        myFile.write(text)
    myFile.close()

    # Finally, let's plot the solutions
    '''
    # plot one chart
    f, axarr = plt.subplots(1, sharex=True)
    axarr.plot(values['x'], M, linewidth=1.5, color='b')
    axarr.set_ylabel('sound velocity')
    '''
    # plot 5 charts
    f, axarr = plt.subplots(len(values)-1, sharex=True)
    axarr[0].plot(values['x'], values['p'], linewidth=1.5, color='b')
    axarr[0].set_ylabel('pressure')
    axarr[0].set_ylim(0, 1.1)
    axarr[1].plot(values['x'], values['rho'], linewidth=1.5, color='r')
    axarr[1].set_ylabel('density')
    axarr[1].set_ylim(0, 1.1)

    axarr[2].plot(values['x'], values['u'], linewidth=1.5, color='g')
    axarr[2].set_ylabel('velocity')
    #axarr[2].set_ylim(0, 1.1)


    axarr[3].plot(values['x'], E, linewidth=1.5, color='c')
    axarr[3].set_ylabel('totalEnergy')
    #axarr[3].set_ylim(0, 1.0)

    axarr[4].plot(values['x'], T, linewidth=1.5, color='r')
    axarr[4].set_ylabel('temperature')
    #axarr[4].set_ylim(0, 1.1)
    #'''

    plt.suptitle('Shocktube results at t={0}\ndust fraction = {1}, gamma={2}'\
                 .format(t, dustFrac, gamma))
    plt.show()
