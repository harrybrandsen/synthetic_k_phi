# -*- coding: utf-8 -*-
"""
This program produces a synthetic k-phi dataset (vs depth) from user-specified
Petrophysical Rock Types (PRT).
The result is a incidental lithology column based on the PRT entered by the
user with accompanying porosity/permeability data - all depth indexed.

The code works such that a random (but more structured than complete random)
ordering of the PRTs is generated: a synthetic lithology column. Due to this 
incidental structuring, this column resembles more "true geology" than what 
would be generated from a complete random state.
Once the "stacking" of PRTs is in place, the random variables are transformed
according to user-specified statistics for each of the PRTs:
    -- 

The user needs to specify the following (in the "USER SETUP" section below):
    -- litholib: a dictionary containing PRTs + characteristics
       {#, ('name prt', 'plot color', 'hash-symbol', 'correlation metric',
            'X5 (5th percentile) phi distribution, X95 phi, X5 k, X95 k)}
        example:   
        litholib = {0:('shale', 'brown',  '--',0.92, 0.07, 0.14, 0.008, 1.5),
                    3:('sst',   'yellow', '.', 0.97, 0.14, 0.28, 10,   5000)}
    -- Sampling rate (sr, in /m)
    -- Top of interval (top_int, m)
    -- Length of interval (length_of_interval, m)
    -- show_intermediate_plots: True/False. This option will show/hide the 
       plots that are created at intermediate stages. These could be useful
       for QC
    -- method defines the method how the correlated variables are created. The
       choices are "cholesky" or "eigh". First, uncorrelated, random, normally 
       distributed variables are generated, which are subsequently multiplied
       by a matrix such that CC^T = R. R is the desired covariance matrix. C is 
       created by either Cholesky decomposition of R (method="cholesky"), or 
       from the eigenvalues and eigenvectors of R (method="eigh").
       Is seems that "Cholesky" generally results in a little higher correlation
       coefficients on the final data tha "eigenvectors and eigenvalues"

It also is possible to change the probability distribution map also. This is 
somewhat trial and error, but the program works best when the peaks lie on a 
diagonal. Adjust the variables "peak_1" and "peak_2" to change the probability
distribution map. The variables "peak_n" are tuples containing the following 
elements:
    
    peak_n = (xoffset, yoffset, amplitude, peakwidth)
    -- xoffset, yoffset define the position of the peak on the probability 
       distribution map
    -- amplitude defines the height of the peak (larger number increases
       probability)
    -- peakwidth define the width of the distribution. The number needs to be 
       greater than 0. Larger numbers increase the width (variance) of the
       distribution

Unless a seed number is entered, the code will produce a new, random dateset
each time it is run. If repeatability is required, fix the random state by 
entering a (any) seed-number.

Known issues:
    -- The lithology/PRT column isn't drawn entirely correct. Especially when
       more than 2 PRTs are used, or when the column is relatively short, the
       colors visible do not seem to correspond to the occurrence of each of
       the PRTs (see percentages top of plot). The data (stored in the pandas 
       DataFrame) are correct though (i.e. it is a graphics or plotting issue)
       
    -- The correlation metrics in the "litholib"/PRT dictionary need to be 
       quite high, even for PRTs that will ultimately be poorly correlated.
       This is unfortunate, as it is somewhat trial and error to get a "good-
       looking" dataset.
       
       
Created December 2019
@author: HARBR (Harry Brandsen)

Acknowledgements:
The function "create_approximately_structured_single_variable" was written
by Co Stuifbergen in JavaScript and translated to Python by HARBR.
"""


import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.constants import g
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
from scipy import stats
from sklearn.cluster import KMeans


###############################################################################
#litholib = {10:('shale',      'saddlebrown', '--',0.92, 0.07, 0.14, 0.008, 1.5),
#            11:('sst',        'yellow',      '.', 0.97, 0.19, 0.28, 80,  5000),
#            12:('shaly sand', 'orange',      '-', 0.96, 0.12, 0.20, 0.5,  100)}
litholib = {10:('shale',      'saddlebrown', '--',0.92, 0.07, 0.14, 0.008, 2),
            11:('sst',        'yellow',      '.', 0.97, 0.14, 0.28, 5,  5000)}

sr = 0.1524                     # sampling rate
top_int = 1950                  # top of interval
length_of_interval = 150        # length of interval. NB: the interval will be 
                                # extended to the next sample increment

method = 'cholesky'             # method needs to be either 'cholesky' or 'eigh'

show_intermediate_plots = True  # show intermediate plots (True) or not (False)

# for probability map: positions, amplitude and width of peaks:
# syntax: peak_n = tuple(xoffset, yoffset, amplitude, peakwidth)
# peakwidth needs to be > 0. Larger number will give a wider peak
peak_1 = (0.75, 0.75, 2.00, 0.2)
peak_2 = (2.25, 2.25, 1.00, 0.3)

# if reproducible results are needed: use a specific seed number, otherwise 
# comment out
seednumber = 1000
#
###############################################################################

# calculate the base of the interval using the sampling rate (might not coincide
# with top_int + length_of_interval)
bot_int = top_int + math.ceil(length_of_interval/sr)*sr

# min/max for probability map
lbound = 0
ubound = peak_2[0] + peak_2[1]


# create df and add first column: TVDSS
df = pd.DataFrame()
df['TVDSS'] = np.linspace(top_int,bot_int,(bot_int-top_int)/sr)
n = len(df) # length of the array
nbins = 100; # resolution of the probability distributions


def create_approximately_structured_single_variable(opt_plot_prob_map=True, opt_plot_dummy_var=True):
    '''This function creates an array with a single dummy variable that is 
    "approximately sorted" (i.e. more structured than random) to resemble true 
    geology better. The "approximate sortingis obtained by picking a random 
    value based on a probability map defined by two normal distributions.
    
    The only arguments required are the variables that define a sum of gaussian
    distributions on the probability map (function "probability_definition") and 
    the length of the dummy data array (obtained automatically).
    
    -- The optional argument opt_plot_prob_map (default=True, otherwise False)
       shows/not shows a visual representation of the probability map
    -- The optional argument opt_plot_dummy_var (default=True, otherwise False)
       shows/not shows the creates (approximately sorted) dummy variable
     
    This function was written by Co Stuifbergen in JavaScript, translated to
    Python by HARBR (30th December 2019)'''
     
#    def probability_density(x,y,xoffset1,yoffset1,ampl1,peakwidth1,xoffset2,yoffset2,ampl2,peakwidth2,*xoffsetn,*yoffsetn,*ampln,*peakwidthn):
    def probability_definition(x,y,peak_1,peak_2):
        '''Defines the "map" with probability density distributions
        x, y are the x and y data
        peak_1 and peak_2 are tuples containing 4 elements each:
        -- xoffset defines the x position of the probability peak
        -- yoffset defines the y position of the probability peak
        -- ampl defines the amplitude of peak (ampl must be >0)
        -- peakwidth defines the width of peak (peakwidth must be >0)
        N.B. max 4 peaks can be passed to this function!
        '''
        result = 0
        for i in range(1,3): # make large enough - nobody will attempt to create a probability map with 4 peaks... ;-)
            xoffset = vars()['peak_'+str(i)][0]
            yoffset = vars()['peak_'+str(i)][1]
            ampl = vars()['peak_'+str(i)][2]
            peakwidth = vars()['peak_'+str(i)][3]
            r2 = math.pow(x-xoffset,2) + math.pow(y-yoffset,2)
            result += ampl*math.exp(-r2/peakwidth)
        return(result)
        
    
    def bins(start,end,n,yi,peak_1,peak_2):
        '''Small function to set the number of bins at the "cross-section"
        @yi: the cumulative distribution curve is divided into n equal-sized bins.
        Note that the result is not normalised, i.e. y[n-1] is not 1.0 '''
        width = (end-start)/n
        y = []
        y.append(probability_definition(yi,start+0.5*width,peak_1,peak_2))
        for i in range(1,n):
            y.append(y[i-1] + probability_definition(yi,(i+0.5)*width,peak_1,peak_2))
        return(y)   
    
    
    def scaled_random(mymin,mymax):
        '''Small function to scale the random value between lbound and ubound'''
        return(mymin + random.random()*(mymax-mymin))
     
    
    myresults = [] # create empty list to store results
    r = scaled_random(lbound,ubound)

    for i in range(n):
        # z = the binned cumulative probabilty at the specific value
        z = bins(lbound,ubound,nbins,r,peak_1,peak_2)
        # small function to scale random value
        r = scaled_random(0,z[nbins-1])
    
        if (r<z[0]):
            myresults.append(lbound+(ubound-lbound)/nbins*(r/z[0]))
        else:
            j = 1
            while(z[j]<r and j<nbins):
                j += 1
            
        myresults.append(lbound+(j+(r-z[j-1])/(z[j]-z[j-1]))*(ubound-lbound)/nbins)
        r = myresults[i]

    # This part is only to visualize the probability map - not necessary for 
    # the program to function properly 
    if opt_plot_prob_map == True:
        # Create arrays for the probability map
        # N.B. the number of points in the isa reduced by a factor 10 to speed up
        # (it1's for visualization only anyway)
        xprob = np.linspace(lbound,ubound,nbins/10) 
        yprob = np.linspace(lbound,ubound,nbins/10)
    
        # create a meshgrid
        xxprob,yyprob = np.meshgrid(xprob,yprob,sparse=True)
        prob = np.zeros((len(xprob),len(yprob)))

        # calculate "z"-value
        for i, X in zip(range(len(xprob)),xprob):
            for j,Y in zip(range(len(xprob)),xprob):
                prob[j,i]=probability_density(X,Y,peak_1, peak_2)
        
        # draw the probability density map
        fig = plt.figure(figsize=(6,6))
        plt.pcolor(xxprob,yyprob,prob)
        plt.colorbar().set_label('Probability density',fontsize=10,fontweight='bold')
        plt.xlabel('x-prob',fontweight='bold')
        plt.ylabel('y-prob',fontweight='bold')
        plt.title('Probability density map',fontweight='bold')
        plt.show()
    
    # convert results to pandas dataframe for easy plotting agains index
    myresults = pd.DataFrame(myresults,columns=['values'])

    if opt_plot_dummy_var == True:
        # show the created dummy variable
        fig = plt.figure(figsize=(4,8))
        plt.plot(myresults.values,myresults.index,'g-')
        plt.gca().invert_yaxis()
        plt.show()

    return(myresults)



def double_dummy_data(x,method):
    '''This small function creates a 2 by n array to later use to generate 
    correlated data. The upper row in this array is replaced by the earlier-
    created "approximately sorted" data (the "x" argument). Depending on the
    method chosen ("cholesky" or "eigh"), the upper resp. lower row needs to
    be replaced (this is the row that remains the same during the matrix 
    transformation)'''
    # generate a 2 by n array with random data
    xy = norm.rvs(size=(2, len(x)))
    
    # depending on the method, either row 0 or row 1 needs to be replaced by the 
    # "approximately sorted data
    if method == 'cholesky':
        row = 0
    elif method == 'eigh':
        row = 1
    
    cntr = 0
    for i in x.index:
        xy[row,cntr] = x.loc[i,:].values[0]
        cntr += 1
    return(xy)



def correlate_doubled_dummy_data(xy,method,corr):
    '''Creates aset of two correlated variables. The user can indicate:
    -- xy is the 2 by n array of dummy data in which the upper row is the
       "approximately sorted" dummy data
    -- the method = "cholesky" or "eigh",
    -- the correlation coefficient r
    
    See https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html
    for details.'''
            
    # The desired covariance matrix.
    if corr > 1 or corr < -1:
        print('The correlation coefficient must be between -1 and 1 incl.')
        return
    
    r = np.array([
            [  1, corr],
            [ corr,  1]])
    
    # Generate samples from three independent normally distributed random
    # variables (with mean 0 and std. dev. 1).
#    x = norm.rvs(size=(2, n))
    
    # We need a matrix `c` for which `c*c^T = r`.  We can use, for example,
    # the Cholesky decomposition, or the we can construct `c` from the
    # eigenvectors and eigenvalues.
    if method == 'cholesky':
        # Compute the Cholesky decomposition.
        c = cholesky(r, lower=True)
    elif method == 'eigh':
        # Compute the eigenvalues and eigenvectors.
        evals, evecs = eigh(r)
        # Construct c, so c*c^T = r.
        c = np.dot(evecs, np.diag(np.sqrt(evals)))
    else:
        print('Invalid method entered...!')
        return
    
    # Convert the data to correlated random variables. 
    return(np.dot(c, xy))



def rescale_k_phi_from_correlated_doubled_dummy_data(correlated_doubled_dummy_data,title, por_5th, por_95th, k_5th, k_95th, show_x_plots=True, plot_col='green'):
    '''Rescales the correlated dummy data. Takes dummary data (a 2 by n float-array)
    as input and returns k, phi rescaled according to entered 5th and 95th 
    percentile values. A plot with the transformed data is also returned by 
    default (set show_x_plots = False if you want no plots displayed).
    
    por_5th is the 5th percentile value of porosity, in fr.b.v. (e.g. 0.10)
    por_95th is the 95th percentile value of porosity, in fr.b.v. (e.g. 0.30)
    k_5th is the 5th percentile value of permeability, in mD (e.g. 2.5)
    k_95th is the 95th percentile value of permeability, in mD (e.g. 3000)'''
    
    # the first column will become porosity
    # for the scaling, the "np.nan" data needs to be filtered out!
    phi = correlated_doubled_dummy_data[0]
    phi = phi/(((np.percentile(phi.dropna(),95) - np.percentile(phi.dropna(),5))/(por_95th-por_5th))) # rescale to 5th-95th percentile porosity
    phi += (por_5th-np.percentile(phi.dropna(),5)) # shift by distance por_5th - 5th percentile dummy data
    
    # teh second column will become permeability
    # for the scaling, the "np.nan" data needs to be filtered out!
    k = correlated_doubled_dummy_data[1]
    k = k/(((np.percentile(k.dropna(),95) - np.percentile(k.dropna(),5))/(np.log10(k_95th)-np.log10(k_5th)))) # rescale to 5th-95th percentile permeability
    k += (np.log10(k_5th)-np.percentile(k.dropna(),5)) # shift by distance k_5th - 5th percentile dummy data
    k = 10**k

    # create plots (if option is set to True) with regression for each PRT sub-dataset
    if show_x_plots == True:
        plt.semilogy(phi,k,marker='o',color=plot_col,linewidth=0,alpha=0.1)
        b0,b1,r,reg_eq = yonx_linear_regression(phi.dropna(),np.log10(k.dropna()),'phi','log10(k)')
        if b1 < 0:
            b2 = ' - ' + str(round(abs(b1),3))
        else:
            b2 = ' + ' + str(round(b1,3))
        reg_eq = 'k = 10' + r'$^{' + str(round(b0,3)) + '*phi' + b2 + '}$' + '  (r = ' + str(round(r,3)) + ')'
        plt.semilogy([np.mean(phi.dropna())-2*np.std(phi.dropna()),np.mean(phi.dropna())+2*np.std(phi.dropna())],
                      [10**(b0*(np.mean(phi.dropna())-2*np.std(phi.dropna()))+b1),10**(b0*(np.mean(phi.dropna())+2*np.std(phi.dropna()))+b1)],
                      'k--',label=reg_eq)    
        plt.xlabel('phi (fr.b.v.)',fontweight='bold')
        plt.ylabel('k (mD)',fontweight='bold')
        plt.title('synthetic k/phi ['+title+']',fontweight='bold')
        plt.xlim([0,math.ceil(max(phi.dropna())*20)/20]) # round upward to the nearest 0.05 
        plt.ylim([10**math.floor(np.log10(min(k.dropna()))),10**math.ceil(np.log10(max(k.dropna())))]) # dynamic scale
        plt.grid(b=True, which='major',color='lightgrey',linestyle='-')
#        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='whitesmoke',linestyle='--')
        plt.legend(loc='best')
        plt.show()   
    return(k,phi)



def yonx_linear_regression(x,y,xvar='x',yvar='y'):
    '''Calculates the Y on X regression line from input x and y arrays.
    The variable labels are optional. When provided, these will be used in the
    regression equation instead of a simple "x" and "y".'''
    if len(x) != len(y):
        return('x and y are not the same length! No regression can be calculated...')
    else:
        try:
            stats.linregress(x,y)
            b0 = stats.linregress(x,y)[0] # slope
            b1 = stats.linregress(x,y)[1] # intercept
            r  = stats.linregress(x,y)[2] # R
            if xvar != '' or yvar != '':
                if b1 > 0:
                    reg_eq = '[Y on X]  ' + yvar + ' = ' + str(round(b0,3)) + '*' + xvar + ' + ' + str(round(b1,3)) + '  (r = ' + str(round(r,3)) + ')'
                else:
                    reg_eq = '[Y on X]  ' + yvar + ' = ' + str(round(b0,3)) + '*' + xvar + ' - ' + str(abs(round(b1,3))) + '  (r = ' + str(round(r,3)) + ')'
            else:
                if b1 > 0:
                    reg_eq = '[Y on X]  y = ' + str(round(b0,3)) + '*x + ' + str(round(b1,3)) + '  (r = ' + str(round(r,3)) + ')'
                else:
                    reg_eq = '[Y on X]  y = ' + str(round(b0,3)) + '*x - ' + str(abs(round(b1,3))) + '  (r = ' + str(round(r,3)) + ')'
            return(b0,b1,r,reg_eq)
        except:
            print('An error occurred. Check the passed input.')

        

# set flags to show/not show intermediate plots (defined by user at the top)
if show_intermediate_plots == True:
    opt_plot_prob_map=True
    opt_plot_dummy_var=True
    show_x_plots = True
else:
    opt_plot_prob_map=False
    opt_plot_dummy_var=False
    show_x_plots = False

random.seed(seednumber) 
x = create_approximately_structured_single_variable(opt_plot_prob_map=opt_plot_prob_map, opt_plot_dummy_var=opt_plot_dummy_var)
# STUPID FIX FOR BUG NOT UNDERSTOOD: trim length to be sure it is identical to df
#x = x[0:len(df)]


# create correlated data array, using the 1D array with approximately structured 
# data to replace one of the rows (depending on the method this is the first
# ('cholesky') or the second ('eigh')) for a the generated 2 by n array with
# random data
# This instance is used to generate a column with PRT-stacking. Properties will
# be added later 
corr=0.99   # start off with a high correlation coefficient. Do not make it equal
            # to 1 (then "eigh" method will crash)
xy = double_dummy_data(x,method)
xy = correlate_doubled_dummy_data(xy,method,corr)
xy = pd.DataFrame(xy.transpose())


# cluster the data
kmeans = KMeans(len(litholib)) # number of clusters = number of PRTs in  litholib 
kmeans.fit(xy)
# use cluster labels as PRTs (NOTE: cluster labels are assigned automatically 
# and there is no possibility to influence this. Instead, retrieve a list with 
# all unique labels and idem all unique PRTs (from litholib). Replace.
df['PRT'] = kmeans.labels_ 
df['PRT'].replace(sorted(pd.DataFrame(kmeans.labels_)[0].unique()),
  sorted(litholib.keys()),inplace=True)


# add columns so they can be used in list comprehension in loop below.
df['LITHOLOGY'] = None
df['k'] = None
df['phi'] = None

# run the synthetic data generator again, but now per rock-type
for key in litholib.keys(): # loop through all rocktypes
    # pick up user-set values for this PRT 
    rt, corr, lpor, hpor, lk, hk, plot_col = litholib[key][0], litholib[key][3], litholib[key][4], litholib[key][5], litholib[key][6], litholib[key][7], litholib[key][1]
    # create a 2 by n sized array with random, normally distributed data
    vars()['xy_'+rt] = double_dummy_data(x,method)
    # replace one row with the approximately structured data
    vars()['xy_'+rt] = correlate_doubled_dummy_data(vars()['xy_'+rt],method,corr)
    # transpose the results (so result is 2 columns instead of 2 rows)
    vars()['xy_'+rt] = pd.DataFrame(vars()['xy_'+rt].transpose())
    # replace all rows that are not belonging to this PRT with np.nan
    for i in range(len(vars()['xy_'+rt])):
        if df.loc[i,'PRT'] != key:
            vars()['xy_'+rt].loc[i,:] = np.nan
    # transform/scale the k/phi for this PRT
    vars()['k_'+rt],vars()['phi_'+rt] = rescale_k_phi_from_correlated_doubled_dummy_data(vars()['xy_'+rt],rt,lpor,hpor,lk,hk,show_x_plots=show_x_plots, plot_col=plot_col)
    df['LITHOLOGY'] = [rt if prt==key else li for prt, li in zip(df['PRT'],df['LITHOLOGY'])]
    df['k'] = [k if prt==key else k_ex for prt, k, k_ex in zip(df['PRT'],vars()['k_'+rt],df['k'])]
    df['phi'] = [phi if prt==key else phi_ex for prt, phi, phi_ex in zip(df['PRT'],vars()['phi_'+rt],df['phi'])]


# plot the results
fig = plt.figure(figsize=(13,6))
tmp_str = '['
for key in litholib.keys(): # loop through all rocktypes
    tmp_str += litholib[key][0] + ' (' + str(round(len(df[df['PRT']==key])/len(df)*100,1)) + '%), '
tmp_str = tmp_str[:-2] + ']\n'
fig.suptitle('PRT with synthetic data  '+ tmp_str,fontweight='bold')
gs = GridSpec(ncols=5,nrows=1,width_ratios=[2.0, 2.0, 0.5, 0.25, 5.0])
ax1 = fig.add_subplot(gs[0]) # the porosity track
ax2 = fig.add_subplot(gs[1]) # the permeability track
ax3 = fig.add_subplot(gs[2]) # the PRT track
ax4 = fig.add_subplot(gs[3]) # this is extra whitespace to avoid label cluttering
ax5 = fig.add_subplot(gs[4]) # cross-plot generated phi/k data per PRT

# porosity
ax1.plot(df['phi'],df['TVDSS'],'b-')
ax1.set_xlim([0,math.ceil(max(df['phi'])*20)/20]) # dynamic phi scale
ax1.set_ylim([top_int,bot_int])
ax1.set_xlabel('porosity (fr.b.v.)',fontweight='bold')
ax1.invert_yaxis()
#ax1.set_yticks()
#ax1.set_xticks()
ax1.grid(b=True, which='major',color='lightgrey',linestyle='-')
ax1.minorticks_on()
ax1.grid(b=True, which='minor', color='whitesmoke',linestyle='--')
ax1.set_ylabel('TVDSS (m)',fontweight='bold')

# permeability
plt.setp(ax2.get_yticklabels(),visible=False)
ax2.semilogx(df['k'],df['TVDSS'],'r-')
ax2.set_xlim([10**math.floor(np.log10(min(df['k']))),10**math.ceil(np.log10(max(df['k'])))]) # dynamic perm-scale
ax2.set_ylim([top_int,bot_int])
ax2.set_xlabel('permeability (mD)',fontweight='bold')
ax2.invert_yaxis()
#ax2.set_yticks()
#ax2.set_xticks()
ax2.grid(b=True, which='major',color='lightgrey',linestyle='-')
ax2.minorticks_on()
ax2.grid(b=True, which='minor', color='whitesmoke',linestyle='--')
ax2.set_xscale('log')


# lithology/PRTs
ax3.invert_yaxis()
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.plot(df['PRT'],df['TVDSS'],'k-',linewidth=0)
for li in sorted(df['PRT'].unique(),reverse=True):
    color, hatch, label = litholib[li][1], litholib[li][2], litholib[li][0]
    ax3.fill_betweenx(df['TVDSS'],-1,df['PRT'],where=df['PRT']==li,color=color,hatch=hatch,label=label,linewidth=0)
ax3.set_xlim([-1,0])
ax3.set_ylim([top_int,bot_int])
ax3.set_xlabel('PRT',fontweight='bold')
ax3.invert_yaxis()
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
if len(litholib)==2:
    ax3.legend(loc='lower center',bbox_to_anchor=(0.53,-0.35))
elif len(litholib)==3:
    ax3.legend(loc='lower center',bbox_to_anchor=(0.53,-0.45))
    
#plt.legend(loc='lower left',bbox_to_anchor=(1.04, 0.925))
#plt.show()

ax4.axis('off')

# cross-plot with k-phi data
#fig = plt.figure(figsize=(6,6))
for key in litholib.keys(): #rocktypes
    rt, color = litholib[key][0], litholib[key][1]
    ax5.plot(df['phi'][df['PRT']==key],df['k'][df['PRT']==key],color=color,label=rt,marker='o',markeredgewidth=1,markeredgecolor='k',linewidth=0,alpha=0.4)

    # regression for each Petrophysical Rock Type
    b0,b1,r,reg_eq = yonx_linear_regression(df['phi'][df['PRT']==key].dropna(),np.log10(df['k'][df['PRT']==key].dropna()),'phi','log10(k)')

#    b0,b1,r,reg_eq = yonx_linear_regression(x,y,'phi','log10(k)')
    reg_eq += '  (' + rt + ')'
    ax5.semilogy([np.mean(df['phi'][df['PRT']==key].dropna())-2*np.std(df['phi'][df['PRT']==key].dropna()),np.mean(df['phi'][df['PRT']==key].dropna())+2*np.std(df['phi'][df['PRT']==key].dropna())],
                  [10**(b0*(np.mean(df['phi'][df['PRT']==key].dropna())-2*np.std(df['phi'][df['PRT']==key].dropna()))+b1),10**(b0*(np.mean(df['phi'][df['PRT']==key].dropna())+2*np.std(df['phi'][df['PRT']==key].dropna()))+b1)],
                  color='k', linestyle='--',linewidth=1.5,label=reg_eq)    

ax5.set_xlabel('phi (fr.b.v.)',fontweight='bold')
ax5.set_ylabel('k (mD)',fontweight='bold')
ax5.set_xlim([0,math.ceil(max(df['phi'])*20)/20]) # dynamic phi scale
ax5.set_ylim([10**math.floor(np.log10(min(df['k']))),10**math.ceil(np.log10(max(df['k'])))]) # dynamic perm-scale
ax5.grid(b=True, which='major',color='lightgrey',linestyle='-')
ax5.minorticks_on()
ax5.grid(b=True, which='minor', color='whitesmoke',linestyle='--')
if len(litholib)==2:
    ax5.legend(loc='lower center',bbox_to_anchor=(0.51,-0.35))
elif len(litholib)==3:
    ax5.legend(loc='lower center',bbox_to_anchor=(0.53,-0.45))
plt.show()



