"""
Compilation of function for image and signal proceesing
SIREGAR Syahril
Originally written on 2017/3/13
    (1) 2017/03/15
    (2) 2017/06/22 --> Revised the B mode and a C mode
    (3) 2017/12/10 -

"""

import numpy as np
import cv2
from scipy.signal import butter, lfilter, filtfilt, convolve2d
import matplotlib.pyplot as plt


def initi_simplisma(d,nr,f):
    """
    Function to calculte the pure profiles
    Reference Matlab Code:
         J. Jaumot, R. Gargallo, A. de Juan, R. Tauler,
         Chemometrics and Intelligent Laboratoty Systems, 76 (2005) 101-110
    
   
    ---:
        input : float d(nrow,ncol) = original spectrum
                integer nr = the number of pure components
                float f = number of noise allowed eg. 0.1 (10%)
        
        output: float spout(nr,nrow) = purest number component profile
                integer imp(nr) = indexes of purest spectra                   
    """  
    nrow = d.shape[0]
    ncol = d.shape[1]
    
    s = d.std(axis=0)
    m = d.mean(axis=0)
    mf = m + m.max() * f
    p = s / mf

    # First Pure Spectral/Concentration profile
    imp = np.empty(nr, dtype=np.int) 
    imp[0] = p.argmax()

    #Calculation of correlation matrix
    l2 = s**2 + mf**2
    dl = d / np.sqrt(l2)
    c = (dl.T @ dl) / nrow

    #calculation of the first weight
    w = (s**2 + m**2) / l2
    p *= w
    #calculation of following weights
    dm = np.zeros((nr+1, nr+1))
    for i in range(1, nr):
        dm[1:i+1,1:i+1] = c[imp[:i],:][:,imp[:i]]
        for j in range(ncol):
            dm[0,0] = c[j,j]
            dm[0,1:i+1]=c[j,imp[:i]]
            dm[1:i+1,0]=c[imp[:i],j]
            w[j] = np.linalg.det(dm[0:i+1, 0:i+1])

        imp[i] = (p * w).argmax()
        
    ss = d[:,imp]
    spout = ss / np.sqrt(np.sum(ss**2, axis=0))
    return spout.T, imp


#slice the cmode
def cmodemip(d3image):
    i,j,k = np.shape(d3image)
    cc = np.zeros((k,j))
    brp = d3image[:,::-1,::-1]
    #brp = d3image[:,:,:]
    cmode1 = brp.max(0) #max
    c = np.transpose( np.expand_dims(cmode1, axis=2), (2, 1, 0) )
    cc[:,:] = c[0,:,:]
    return cc

#slice the bmode
def bmodemip(d3image):
    b = d3image.max(2) #2
    return b 


#histogram of image    
def imhist(img,fignum, nmin,nmax):
    plt.figure(fignum)
    plt.hist(img.ravel(), bins=256, range=(nmin, nmax), fc='k', ec='k')
   # plt.show()
    return 

###filter function 

#low pass filter
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

#band pass filter with lfiter
def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#band pass filter with filtfilt

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  #  y = lfilter(b, a, data)
    y =filtfilt(b, a, data)
    return y

def butter_bandpass_filter2(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
#    y =filtfilt(b, a, data)
    return y

#high pass filter

def butter_highpass(cutoff,fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b,a = butter(order,normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b,a,data)
    return y

#function for centroid of obk    
def centerofmass(imgin):
    imgbw = imgin.copy()
    #calculate the center of mass of binary image
    cnts = cv2.findContours(imgbw, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        cx,cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return cx, cy

#function to fill the binary image        
def imfill(imgbw):
    # Copy the thresholded image.
    im_floodfill = imgbw.copy()
     
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = imgbw.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
      
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    im_out = imgbw | im_floodfill_inv

    return im_out

def estimate_noise(I):
    H, W = I.shape
    M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))
    return sigma
