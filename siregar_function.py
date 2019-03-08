""" function for PA machine Seiryo
    copas from image_siregar
    2018/1/5
    Syahril Siregar
"""

import numpy as np
import cv2
from scipy.signal import butter, lfilter, filtfilt, convolve2d
import matplotlib.pyplot as plt


def cmodemip(d3image):
    i,j,k = np.shape(d3image)
    cc = np.zeros((k,j))
    brp = d3image[:,::-1,::-1]
    cmode1 = brp.max(0) #max
    c = np.transpose( np.expand_dims(cmode1, axis=2), (2, 1, 0) )
    cc[:,:] = c[0,:,:]
    cc = cc/np.max(cc)
    return cc

#slice the bmode
def bmodemip(d3image):
    b = d3image.max(2) #2
    b = b/np.max(b)
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
