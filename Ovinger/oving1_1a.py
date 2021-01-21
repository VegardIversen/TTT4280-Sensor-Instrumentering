import numpy as np
import scipy.stats as st



x = np.array([20.6, 20.4, 20.4, 20.6, 20.4, 20.8, 20.5, 20.5, 20.5, 20.4,20.5, 20.5, 20.5, 20.5, 20.4, 20.4, 20.4, 20.5, 20.3, 20.6])
x_2 = np.array([20.4, 20.4, 20.4, 20.2, 20.4, 20.3, 20.4, 20.5, 20.4, 20.4,
20.4, 20.4, 20.1, 20.3, 20.3, 20.2, 20.3, 20.2, 20.3, 20.3])
x_3 = x_2[0:10]

def mean(x):
    return x.mean()



def opp1a(x,frigrader=1):
    std = np.std(x,ddof=frigrader) # In standard statistical practice, ddof=1 provides an unbiased estimator of the variance of the infinite population. ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.
    return std
#print(opp1a(x))

def opp1b(x,interval=0.95,frihetsgrader=1):
    conf_int = st.t.interval(interval, len(x)-frihetsgrader, loc=np.mean(x), scale=st.sem(x))
    m = mean(x)
    pm = u'\u00B1'
    diff = round(conf_int[1]-m,6)
    print(f"{interval*100} % Confidence interval equals {m} {pm} {diff} ")
    return conf_int

def opp1c(x,tp=1.96,interval=0.95):
    m = mean(x)
    pm = u'\u00B1'
    std = opp1a(x)
    pred_int = [m - tp*std*np.sqrt(1+1/len(x)),m + tp*std*np.sqrt(1+1/len(x))]
    diff = round(pred_int[1]-m,6)
    print(f"{interval*100} % Prediction interval equals {m} {pm} {diff} ")
    return pred_int

def opp3(): #endre verdier her
    r_values = 1000 + 0.02 * (np.random.rand(int(1e5)) - 0.5)
    relative_std = np.std(r_values)/np.mean(r_values)
    return relative_std

