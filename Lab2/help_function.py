import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal as ss
from scipy import signal
np.random.seed(1)

#prepros function too remove DC, and possible upsampling 
def prepros(data, upSamplingFactor=1):
    data = data[4000:] #removing first part of signal, to remove bad samples
    length = len(data[:,0])
    data = signal.detrend(data, axis=0) #remove DC from signal
    num_of_samples = data.shape[0] #length of signal

    if(upSamplingFactor != 1):
        data = signal.resample(data, upSamplingFactor*length, axis=0) #upsampling, needs testing
        num_of_samples = data.shape[0] 

    return data, num_of_samples

#find lag from correlation signal
def lag(corr,upSamplingFactor=1,maxdelay=9): #max delay on our circuit is 5, tor is 9.
    lag = np.argmax(corr)-(maxdelay*upSamplingFactor) #calculate lag, 
    return lag

#correlation function between two mics
#mode = full er kun brukt for plotting. 
def correlate(x, y, upSamplingFactor=1, maxdelay=9, mod='valid'): #maxdelay calculated from formula, bc of distance between mics
    corr = np.abs(np.correlate(x,y[(maxdelay*upSamplingFactor):-(maxdelay*upSamplingFactor)], mode=mod))
    return corr
#calculate degree from lags array, from the lag of the different mics
def degree(lags, positive_angl=True):
    theta = np.arctan2(np.sqrt(3)*(lags[0]+lags[1]), (lags[0]-lags[1]-2*lags[2])) #formula for theta
    if theta<0 and positive_angl:
        return 360 + np.degrees(theta) #not to get negative degrees. 
    return np.degrees(theta)


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data

def call_method(o, name,N): #used to call windows type in window function,  o er pakkenavnet, name er navnet på funksjonen i o, N er argumentet til funksjonen. 
    return getattr(o, name)(N)

def window(typ='hamming', N=0):
    f = signal.windows
    win = call_method(f,typ,N)
    return win



def direction_plot(angle, target, savefig=True, show=True):
    #variables
    
    N = 360 #max degrees
    bottom = 0 #The y coordinate(s) of the bars bases
    max_height = 1 #just has to be larger than 0
    colors = ['red','blue','green','purple','pink','orange'] #colors in the bar plot. 
    width = ((2*np.pi) / N) + 0.01 #dont need 0.01, but kan change size to make it easier to see on pdf.
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radi = np.zeros(N) # makes array with angles from 0 to  360
    org = np.zeros(N)
    org[target] = max_height

    if isinstance(angle, list) or isinstance(angle, np.ndarray): #checks if input is an array or a single degree
        for i in angle:
            radi[int(i)] = max_height #makes the angles found larger than 0.
    else:
        radi[int(round(angle))] = max_height #makes the angle found larger than 0.
    
    num_color = len(angle) #gets number of angles
    bar_color = colors[0:num_color] #color change in bar
    #ax = plt.subplot(111, polar=True) #makes a polar subplot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #ax.set_theta_zero_location('N') #makes 0/360 as the top
    ax.set_theta_direction(1) #makes it clockwise
    #bars = ax.bar(theta, radi, color=bar_color, width=width, bottom=bottom) #bar to show direction of noise /angle
    bars = ax.bar(theta, radi, width=width, bottom=bottom, label='Beregnet innfallsvinkler')
    targ = ax.bar(theta, org, width=width, bottom=bottom, label=f'Forventet innfallsvinkel av lydkilde.')
    #ax.bar(theta,org, width=width,bottom=bottom)
    ax.set_yticklabels([]) #removes the radius numbers/y axis numbers
    ax.set_title("Plot av vinkelen: " + str(int(round(target))) + " i grader:") #title, todo: add true angle

    # todo make the bars different colors
    
    # color_change_var = 0
    # for r, bar in zip(radi, bars): 
    #     bar.set_facecolor(plt.cm.jet(r / 10.))
    #     bar.set_alpha(0.8)
    #plt.legend(loc='upper right')
    plt.legend(handles=[bars, targ], bbox_to_anchor=(1.05, 1), loc='upper left')
    if savefig:
        name = f'./figurplot/wn{target}.png'
        plt.savefig(name)
    if show:    
        plt.show()


def lowpass_filter(data, cutoff_freq, lowercutoff,order=6):
    sample_freq = data.shape[0]
    b, a = signal.butter(order, [lowercutoff/(sample_freq/2), cutoff_freq/(sample_freq/2)], 'bandpass')
    data_filt = signal.filtfilt(b, a, data, padlen=0)
    return data

def get_cross_corr(data, mod='full', out='all', upsampling=1, preprocessed=True): #out='all' for all correlations, cc for only crosscorr, ac for autocorr

    if not preprocessed:
        print('Preprocessing...')
        data, num_of_samples = prepros(data, upSamplingFactor=upsampling)
    #cc=crosscorraltion, ac=autocorrelation
    if mod=='full':
        cc2_1 = correlate(data[:,1],data[:,0], upSamplingFactor=upsampling, mod=mod) #mic 2 and mic 1
        cc3_1 = correlate(data[:,2],data[:,0], upSamplingFactor=upsampling, mod=mod) #mic 3 and 1
        cc3_2 = correlate(data[:,2],data[:,1], upSamplingFactor=upsampling, mod=mod) #mic 3 and 2
        ac1_1 = correlate(data[:,0],data[:,0], upSamplingFactor=upsampling, mod=mod) #mic 1 mic 1, autocorr
    else:
        cc2_1 = correlate(data[:,1],data[:,0], upSamplingFactor=upsampling) #mic 2 and mic 1
        cc3_1 = correlate(data[:,2],data[:,0], upSamplingFactor=upsampling) #mic 3 and 1
        cc3_2 = correlate(data[:,2],data[:,1], upSamplingFactor=upsampling) #mic 3 and 2
        ac1_1 = correlate(data[:,0],data[:,0], upSamplingFactor=upsampling) #mic 1 mic 1, autocorr

        

    #this is bad code, but could bother now.
    if out=='ac':
        return ac1_1

    elif out=='cc':
        return cc2_1, cc3_1, cc3_2
    else:
        return ac1_1, cc2_1, cc3_1, cc3_2


def plot_raw(data):
    d1 = data[:,0]
    d2 = data[:,1]
    d3 = data[:,2]
    d4 = data[:,3]
    d5 = data[:,4]
    plt.subplot(3, 1, 1)
    plt.title("Raw data from ADC 1")
    plt.xlabel("Sample")
    plt.ylabel("Conversion value")
    plt.plot(np.arange(len(d1))*32e-6, d1)

    plt.subplot(3, 1, 2)
    plt.title("Raw data from ADC 2")
    plt.plot(np.arange(len(d2))*32e-6, d2)
    plt.xlabel("Sample")
    plt.ylabel("Conversion value")

    plt.subplot(3, 1, 3)
    plt.title("Raw data from ADC 3")
    plt.plot(np.arange(len(d3))*32e-6, d3)
    plt.xlabel("Sample")
    plt.ylabel("Conversion value")

    # plt.subplot(5, 1, 4)
    # plt.title("Raw data from ADC 4")
    # plt.plot(np.arange(len(d4))*32e-6, d4)
    # plt.xlabel("Sample")
    # plt.ylabel("Conversion value")

    # plt.subplot(5, 1, 5)
    # plt.title("Raw data from ADC 5")
    # plt.plot(np.arange(len(d5))*32e-6, d5)
    # plt.xlabel("Sample")
    # plt.ylabel("Conversion value")

    plt.tight_layout()
    plt.show()

def plot_corr(ac, cc1, cc2, cc3): #cc=crosscorraltion, ac=autocorrelation
    plotwin = 1000
    plt.subplot(4,1,1)
    plt.plot(range(-plotwin, plotwin), ac[len(ac)//2-plotwin:len(ac)//2+plotwin]/max(ac))
    plt.xlabel('samples, n')
    plt.ylabel('r_11')
    plt.title('Autocorr_1_  1')

    plt.subplot(4,1,2)
    plt.plot(range(-plotwin, plotwin), cc1[len(cc1)//2-plotwin:len(cc1)//2+plotwin]/max(cc1))
    plt.xlabel('samples, n')
    plt.ylabel('r_21')
    plt.title('crosscorr_2_1')

    plt.subplot(4,1,3)
    plt.plot(range(-plotwin, plotwin), cc2[len(cc2)//2-plotwin:len(cc2)//2+plotwin]/max(cc2))
    plt.xlabel('samples, n')
    plt.ylabel('r_31')
    plt.title('crosscorr_3_1')

    plt.subplot(4,1,4)
    plt.plot(range(-plotwin, plotwin), cc3[len(cc3)//2-plotwin:len(cc3)//2+plotwin]/max(cc3))
    plt.xlabel('samples, n')
    plt.ylabel('r_32')
    plt.title('crosscorr_3_2')

    plt.tight_layout()
    plt.show()

    

