import help_function as hf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import csv


def folder_degree_calc(folder,upsampl=1):
    angles = []
    cutoff_freq = 500
    lower_freq = 350
    for sample in os.listdir(folder):

        sample_period, data = hf.raspi_import(folder + sample) #path too signal
        data, num_of_samples = hf.prepros(data, upSamplingFactor=upsampl) #prepros data, here upsampling should be done
        #data = hf.lowpass_filter(data, cutoff_freq, lower_freq)
        crosscorr2_1 = hf.correlate(data[:,1],data[:,0], upSamplingFactor=upsampl) #mic 2 and mic 1
        crosscorr3_1 = hf.correlate(data[:,2],data[:,0], upSamplingFactor=upsampl) #mic 3 and 1
        crosscorr3_2 = hf.correlate(data[:,2],data[:,1], upSamplingFactor=upsampl) #mic 3 and 2
        autocorr1_1 = hf.correlate(data[:,0],data[:,0], upSamplingFactor=upsampl) #mic 1 mic 1, autocorr
        lags = np.array([hf.lag(crosscorr2_1, upSamplingFactor=upsampl), hf.lag(crosscorr3_1, upSamplingFactor=upsampl), hf.lag(crosscorr3_2, upSamplingFactor=upsampl)])
        degrees = hf.degree(lags)
        print(f'{sample} has {degrees} degrees calculated')
        angles.append(degrees)
    return np.array(angles)

#some quick fix code to make csv to make table in latex. Using append function of write, so if you run it more than once it just appends it on wich is not good. 
#should fix this, and check if the file is already there, and write not append. 
first_line_flag = False
def write_to_csvfile(degree, angles):
    global first_line_flag
    measurements = [0]*14
    measurements[0] = degree
    if len(angles) != 13:
        print('padder')
        diff = 13-len(angles)
        angles = np.pad(angles,[0,diff],constant_values=np.nan)
    
    with open('angledata.txt', 'a+') as fil:
        if not first_line_flag:
            fil.write('Vinkel, Måling1, Måling2, Måling3, Måling4, Måling5, Måling6, Måling7, Måling8, Måling9, Måling10, Måling11, Måling12, Måling13, std, mean \n')
            first_line_flag = True

        for i in range(1,len(measurements)):
            measurements[i]= round(angles[i-1],3)
        measurements.append(round(np.nanstd(angles),3))
        measurements.append(round(np.nanmean(angles),3))
        string = ''
        for i in measurements:
            string += str(i) + ','
        string = string[:-1]     
        
        fil.write(string)
        fil.write('\n')
        fil.close()

def run_calc_and_write():
    deg = [0,30,60,90,150,165,225,270,330,345]
    for i in deg:
        degree = i
        path = f'./samples/tor_cir_{degree}/'
        angles = folder_degree_calc(path)
        print(len(angles))
        print(f'Gjennomsnitt av vinkler er: {np.mean(angles)}')
        print(f'Standardavvik av vinkler er {np.std(angles)}') #bør ddof være 1? 
        hf.direction_plot(angles, degree, savefig=False, show=False)
        write_to_csvfile(degree,angles)

if __name__ == "__main__":
    #run_calc_and_write()
    path = './samples/tor_cir_330/torwn330d_4.bin'
    sample_period, data = hf.raspi_import(path)
    data, num_of_samples = hf.prepros(data, upSamplingFactor=10)
    ac, cc1, cc2, cc3 = hf.get_cross_corr(data, mod='full', out='all', upsampling=4, preprocessed=False)#dette er utrolig treigt hvis upsampling er større en 1, grunnet at mode til correlasjonen er full
    hf.plot_corr(ac, cc1, cc2, cc3)
    #hf.plot_raw(data)


