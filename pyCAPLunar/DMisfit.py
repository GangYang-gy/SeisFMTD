# -------------------------------------------------------------------
# Function to calculate the waveform misfit.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca  
# Modified By: Gang Yang (gangy.yang@mail.utoronto.ca)
# -------------------------------------------------------------------

#from numba import jit
import numpy as np
#@jit
def DMisfit(syn, data,raw_data,type='Zhu1996'):
    ''' Compute the waveform misfit. '''

    # L2 Misfit
    if str(type).upper() == ('L2'):
        return np.average(np.square(np.subtract(syn, data)))

    elif str(type).upper() == ('GRAD'):
        s = np.dot(np.fabs(raw_data), np.fabs(raw_data))
        return np.subtract(syn, data) / s

    # Misfit function:
    # Zhu, L., & Helmberger, D. V. (1996). Advancement in source estimation. BSSA, 86(5), 1634–1641
    else:
        #return np.sum(np.square(np.subtract(syn, data))) / np.dot(np.fabs(syn), np.fabs(data))
        s = np.dot(np.fabs(raw_data), np.fabs(raw_data))
        return np.sum(np.square(np.subtract(syn, data))) / s
        

