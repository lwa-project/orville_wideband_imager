#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#from lsl.common.stations import lwasv

#STATION = lwasv
#ANTENNAS = STATION.antennas

def _plot_matrices(correlations, averages):
    #Build the plot.
    pols = ['XX', 'XY', 'YX', 'YY']
    fig = plt.figure(1) #constrained_layout=True)
    axes = []
    gs = gridspec.GridSpec(2,2)
    for i in range(4):
        gsp = gridspec.GridSpecFromSubplotSpec(2, 1,
                subplot_spec=gs[i], wspace=0.0, hspace=0.0,
                height_ratios=[1.0,0.3])
        axes.append([])
        axes[-1].append(fig.add_subplot(gsp[1]))
        axes[-1].append(fig.add_subplot(gsp[0]))

    fig.canvas.draw()

    #Set the color scale.
    vmin, vmax = np.percentile(np.abs(correlations), q=[1,99])

    #Plot.
    x = y = np.arange(correlations.shape[0]) + 1
    mapper = {0: 'XX', 1: 'YX', 2: 'XY', 3: 'YY'}
    for i in range(2):
        for j in range(2):
            indx = 2*i + j
            ax = axes[indx][0]
            #ax.cla()
            ax.plot(x, averages[:,i,j], 'o')
            if indx > 1:
                ax.set_xlabel('Antenna Number', fontsize=12)
            ax.set_ylabel('Mean', fontsize=12)
            ax.tick_params(which='both', direction='in', length=8, labelsize=12)

            ax = axes[indx][1]
            ax.set_title(mapper[indx],fontsize=14)
            #ax.cla()
            c=ax.imshow(correlations[:,:,i,j], aspect='auto', origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax, extent=(x.min(),x.max(),y.min(),y.max()))
            ax.set_ylabel('Antenna Number', fontsize=12)
            ax.set_xticks([])
            ax.tick_params(which='both', direction='in', length=8, labelsize=12)

    cb = fig.colorbar(c, ax=axes)#, orientation='horizontal')
    cb.set_label(r'$|C_{ij}|$', fontsize=12)

    plt.show()  

def main(args):

    #Read in the data.
    corr = np.load(args.file)['data']

    #Build the full correlation matrix.
    #nstands = len(ANTENNAS) // 2
    nstands = 256
    cmatrix = np.zeros((nstands, nstands, 2, 2))
    count = 0
    for i in range(nstands):
        for j in range(i, nstands):
            cmatrix[i,j,:,:] = corr[count,:,:]
            cmatrix[j,i,:,:] = corr[count,:,:].conj()
            if i == j:
                cmatrix[i,j,1,0] = cmatrix[i,j,0,1].conj()
                cmatrix[j,i,1,0] = cmatrix[j,i,0,1].conj()

            count += 1

    #Compute the average for each antenna.
    avg = np.mean(cmatrix, axis=1)

    _plot_matrices(cmatrix, avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the correlation matrices output by the Orville Wideband Imager',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('file', type=str,
            help='.NPZ file output by the MatrixOp module')

    args = parser.parse_args()
    main(args)
