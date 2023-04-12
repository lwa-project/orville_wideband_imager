#!/usr/bin/env python3

import os
import sys
import warnings
import argparse
import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings(category=RuntimeWarning, action='ignore')

def summary_text(infile, mjd, time, mask, suspectX, suspectY, crossed):
    #Note: Stands are 1-indexed, so the +1 is required in the reporting.

    message = f"""Summary of Orville Wideband Imager Correlation Metrics
Data File: {infile}
MJD: {mjd}
Timetag: {time}

Number of bad antennas for XX pol: {np.sum(mask[:,0])}
List: {np.where(mask[:,0])[0] + 1}

Number of bad antennas for YY pol: {np.sum(mask[:,1])}
List: {np.where(mask[:,1])[0] + 1}

Number of suspect antennas for XX pol: {len(suspectX)}
List: {suspectX + 1}

Number of suspect antennas for YY pol: {len(suspectY)}
List: {suspectY + 1}

Number of potentially cross-polarized antennas: {crossed.size}
List: {crossed + 1}
    """
    return message

def _plot_matrices(matrix, title=None, mask=None, susX=None, susY=None, crossed=None):
    #Build the plot.
    if title is None:
        title = 'LWA-SV Correlation Matrices' 
        mapper = {0: 'XX', 1: 'XY', 2: 'YX', 3: 'YY'}
    else:
        title = title
        mapper = {0: 'XX - XY', 1: 'XX - YX', 2: 'YY - XY', 3: 'YY - YX'}

    fig = plt.figure(1)
    fig.suptitle(title, fontsize=16)
    axes = []
    gs = gridspec.GridSpec(2,2)
    for i in range(4):
        gsp = gridspec.GridSpecFromSubplotSpec(2, 1,
                subplot_spec=gs[i], wspace=0.0, hspace=0.0,
                height_ratios=[1.0,0.3])
        axes.append([])
        axes[-1].append(fig.add_subplot(gsp[0]))
        axes[-1].append(fig.add_subplot(gsp[1], sharex=axes[-1][-1]))

    fig.canvas.draw()

    #Set the color scale.
    vmin, vmax = np.nanpercentile(np.abs(matrix), q=[1,99])

    #Plot.
    x = y = np.arange(matrix.shape[0]) + 1
    for i in range(2):
        for j in range(2):
            avg = (np.sum(matrix[:,:,i,j], axis=0) - np.diag(matrix[:,:,i,j])) / (matrix.shape[0] - 1)

            indx = 2*i + j
            ax = axes[indx][1]
            ax.plot(x, avg, 'o')
            if j == i:
                if mask is not None:
                    ax.plot(x[mask[:,i]], avg[mask[:,i]], 'ro')
                if i == 0 and susX is not None:
                    ax.plot(x[susX], avg[susX], 'yo')
                if i == 1 and susY is not None:
                    ax.plot(x[susY], avg[susY], 'yo')
            if crossed is not None:
                ax.plot(x[crossed], avg[crossed], 'mo')
 
            if indx > 1:
                ax.set_xlabel('Antenna Number', fontsize=12)
                ax.set_xlim((x.min(), x.max()))
            ax.set_ylabel('Mean', fontsize=12)
            ax.tick_params(which='both', direction='in', length=8, labelsize=12)

            ax = axes[indx][0]
            ax.set_title(mapper[indx],fontsize=14)
            c=ax.imshow(matrix[:,:,i,j], aspect='auto', origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax, extent=(x.min(),x.max(),y.min(),y.max()))
            ax.set_ylabel('Antenna Number', fontsize=12)
            ax.tick_params(which='both', direction='in', length=8, labelsize=12)

    cb = fig.colorbar(c, ax=axes)
    cb.set_label(r'$|C_{ij}|$', fontsize=12)

    plt.show()  

def main(args):

    #Read in the data.
    corr = np.load(args.file)['data']

    #Build the full correlation matrix. 
    nstands = 256
    cmatrix = np.zeros((nstands, nstands, 2, 2), dtype=np.complex64)
    count = 0
    for i in range(nstands):
        for j in range(i, nstands):
            cmatrix[i,j,:,:] = corr[count,:,:]
            cmatrix[j,i,:,:] = corr[count,:,:].conj()
            if i == j:
                cmatrix[i,j,1,0] = cmatrix[i,j,0,1].conj()
                cmatrix[j,i,1,0] = cmatrix[j,i,0,1].conj()

            count += 1
    
    del corr

    #Compute the absolute value and normalize each polarization individually.
    cmatrix = np.abs(cmatrix)
    for i in range(cmatrix.shape[2]):
        for j in range(cmatrix.shape[3]):
            cmatrix[:,:,i,j] /= np.nanmax(cmatrix[:,:,i,j])

    #Flag totally dead antennas using XX and YY data.
    #The averaging ignores the autocorrelations.
    dead = []
    avg = np.zeros((cmatrix.shape[1], cmatrix.shape[2]),dtype=np.float64)
    for i in range(cmatrix.shape[2]):
        avg = np.nansum(cmatrix[:,:,i,i]-np.diag(cmatrix[:,:,i,i]), axis=0) / (cmatrix.shape[0]-1)

        dead.append(np.where( avg == 0 )[0])

    #Mask out dead antennas
    mask = np.ones((cmatrix.shape[1], cmatrix.shape[2]), dtype=bool)
    for i in range(mask.shape[1]):
        mask[dead[i],i] = False

    #Compute the cross-polarization metric, ignoring the dead antennas.
    cross_pol = []
    d = np.zeros_like(cmatrix)
    for i in range(cmatrix.shape[2]):
        for j,k in zip([0,1],[1,0]):
            d[:,:,i,j] = cmatrix[:,:,i,i] - cmatrix[:,:,j,k]
            cross_pol.append( np.nanmean(d[:,:,i,j], axis=0, where=mask[:,i]) )

    R = np.amax(cross_pol, axis=0)
    crossed = np.where( R < 0 )[0]

    #Plot the cross-correlation matrices, if requested.
    if args.plot:
        _plot_matrices(d, title='Cross-Correlation Matrices', mask=~mask, crossed=crossed)

    del d

    suspectX, suspectY = [], []
    #Iteratively recompute the flagging metric until no more are flagged.
    while True:
        for i in range(cmatrix.shape[2]):
            avg = np.mean(cmatrix[:,:,i,i], axis=0, where=mask[:,i])
 
            med = np.median(avg)
            bad = np.where( np.abs(avg-med) >= 3*np.std(avg) )[0]
            sus = np.where( (np.abs(avg-med) >= np.std(avg)) & (np.abs(avg-med) < 3*np.std(avg)) )[0]

            if i == 0:
                suspectX.append(sus)
            else:
                suspectY.append(sus)

            mask[bad,i] = False
            
            if i == 0:
                X_bad = bad.size
                X_sus = sus.size
            else:
                Y_bad = bad.size
                Y_sus = sus.size
 
        if (X_bad+X_sus == 0) and (Y_bad+Y_sus == 0):
            break
        else:
            continue

    suspectX = np.unique(np.concatenate(suspectX),0)
    suspectY = np.unique(np.concatenate(suspectY),0)

    #The outrigger tends to always get flagged
    #since it doesn't correlate strongly with any other
    #dipole due to the long baseline. Thus, a better
    #metric is probably to look at the XX and YY 
    #autocorrelations and compare to the other autos.
    for i in range(cmatrix.shape[2]):
        autos = np.diag(cmatrix[:,:,i,i])
        med = np.median(autos)
        std = np.std(autos)
        if (np.abs(autos[-1] - med) <= std):
            mask[-1,i] = True
        else:
            pass

    #Print summary of bad antennas.
    print(f'Found {np.sum(~mask[:,0])} bad antennas in XX')
    print(f'Found {np.sum(~mask[:,1])} bad antennas in YY')
    print(f'Found {suspectX.size} sus antennas in XX')
    print(f'Found {suspectY.size} sus antennas in YY')
    print(f'Found {crossed.size} potentially cross-polarized antennas')

    #Plot, if requested.
    if args.plot:
        _plot_matrices(cmatrix, mask=~mask, susX=suspectX, susY=suspectY, crossed=crossed)

    #Write out a summary file, if requested.
    if args.write:
        #Set up the output directory
        cwd = os.getcwd()
        try:
            os.mkdir('Summaries')
        except FileExistsError:
            pass
        path = os.path.join(cwd,'Summaries')

        #Timetag info from the name of the input file.
        mjd = args.file.split('/')[-1].split('.')[0].split('_')[1]
        time = args.file.split('/')[-1].split('.')[0].split('_')[2]

        #Set up the filename to be written.
        filename = os.path.join(path, 'Summary_'+mjd+'_'+time+'.txt')
        
        #Write the file.
        outfile = open(filename, 'w')
        outfile.write(summary_text(args.file.split('/')[-1], mjd, time, ~mask, suspectX, suspectY, crossed))
        outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze the correlation matrices output by the Orville Wideband Imager',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('file', type=str,
            help='.NPZ file output by the MatrixOp module')
    parser.add_argument('-p', '--plot', action='store_true',
            help='Plot the correlation matrices')
    parser.add_argument('-w', '--write', action='store_true',
            help='Write summary results to disk as a .txt file')
    args = parser.parse_args()
    main(args)
