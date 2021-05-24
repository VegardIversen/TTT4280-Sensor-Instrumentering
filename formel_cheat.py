from os import remove
import numpy as np
import scipy.stats as st
import scipy.constants as co
from re import findall
#beregninger for eksamen i sensor og instrumentering, pass på hva som er input i funksjonene. Har ikke lagt til noe feil sjekking.
#Bør hovedsaklig kun brukes til raske beregninger. 

def calculate_std(samples, ddof=1):
    return np.std(samples,ddof=ddof)

def calculate_uniform_std(val, prec): #bruk decimal for prosent, 1% = 0.01
    val_max = val+val*prec
    val_min = val-val*prec
    std = (val_max-val_min)/(np.sqrt(12))
    print(f'std for uniform: {std}')
    std_rel = std/val
    print(f'Relativ std: {std_rel}')
    return (val+val*prec - val*prec)/(np.sqrt(12))

#denne funker ikke.
'''
def calculate_serie_std_num(R,prec):
    nsamples = 10000
    dR = prec*R
    Rvec = R + 2*dR * (np.random.rand(nsamples,2)-0.5)
    #Rvecsum = np.sum(Rvec)
    print(f'mean: {np.mean(Rvec)}')
    print(f'Realtiv std {np.std(Rvec)/np.mean(Rvec)}')
'''

def calculate_mean(samples):
    return np.mean(samples)

def watt_to_dB(x):
    return 10*np.log10(x)

def watt_to_dBm(x):
    return 10*np.log10(x*1000)


def confidence_interval_st(samples,alpha=0.95):
    return st.t.interval(alpha, len(samples)-1, loc=np.mean(samples), scale=st.sem(samples))

def confidence_interval_samp(samples, alpha=0.05): # alpha-significance level 
    n = len(samples)
    df = n-1 #df-degrees of freedom,
    tp = st.t.ppf(1 - alpha/2, df) #check if this is correct
    mx = calculate_mean(samples)
    sx = calculate_std(samples)
    ci, pm = confidence_interval(mx,sx,tp,n)
    print(f'Confidence interval is {mx} +/- {pm} \n')
    print(f'Confidence interval is {ci}')

                                        
                                           

def confidence_interval(mx,sx,tp,n):
    pm = tp*sx/(np.sqrt(n)) #plus minus part
    ci = [mx-pm,mx+pm]
    return ci, pm

def prediction_interval(samples, alpha=0.05):
    n = len(samples)
    df = n-1 #df-degrees of freedom,
    tp = st.t.ppf(1 - alpha/2, df) #check if this is correct
    mx = calculate_mean(samples)
    sx = calculate_std(samples)
    pi_pm = tp*sx*np.sqrt(1+1/n)
    pi = [mx-pi_pm,mx+pi_pm]
    print(f'Prediction interval is {mx} +/- {pi_pm} \n')
    print(f'Prediction interval is {pi}')


#Noise

def calculate_G(Po,Pi): #P0 - P_out, Pi = P_in 
    G = 10*np.log10(Po/Pi)
    print(G)
    return G
def calculate_G_dB(Po,Pi): #P0 - P_out, Pi = P_in 
    G = Po-Pi
    print(G)
    return G

def calculate_SNR(Ps, N):
    SNR = Ps/N
    print(SNR)
    return SNR

def calculate_SNR_dB(Ps, N):
    SNR = 10*np.log10(Ps/N)
    print(SNR)
    return SNR

#Beregn Effekttetthetsspekteret Sn(f) til termisk støy, pass på hva slags type det er sjeldent brukt dette
def calculate_Sn_blackOb(f, T):
    hf = co.h * f
    kT = co.k * T
    Sn = (hf)/(np.exp(hf/kT)-1)
    print(Sn)
    return Sn
#Beregn Støy
def calc_N(T,B):
    N = co.k * T * B
    #print(N)
    return N

def N_dBm(T,B):
    N = watt_to_dBm(calc_N(T,B))
    return N

#effektive støytemperaturen
def calculate_T_e(N,B):
    return N/(co.k * B)



def calculate_T_e_nf(nf):
    p = nf/10
    te = 290*(10**p -1)
    print(f'egen temperatur: {te}')
    return te

#viktig at det sendes inn arrays
def calculate_Tcas(G_arr, T_arr):
    T_cas = 0
    G_temp = 1
    for i in range(len(T_arr)):
        T_cas += T_arr[i]/G_temp
        G_temp *= 10**(G_arr[i]/10)
    return T_cas

def calculate_Fcas(G_arr,F_arr):
    F_cas = F_arr[0]
    G_temp = 1
    for i in range(1,len(F_arr)):
        F_cas += (F_arr[i]-1)/G_temp
        G_temp *= G_arr[i-1]
    return F_cas







#Beregn forsterker system med en forsterker og en T_0
def amplifier_1_N(T,B,G,F,Ftype='dB'):
    if(Ftype=='dB'):
        NF = F
        p = F/10
        F = 10**p
    else:
        NF = 10*np.log10(F)
    j = G/10
    G = 10**j
    N = G*(calc_N(290,B)+calc_N(T,B))
    return N

def amplifier_1_N_Tr(Tr,B,G,F,Ftype='dB'):#når det kobles på en ideell motstand på inngangen
    if(Ftype=='dB'):
        NF = F
        p = F/10
        F = 10**p
        Te = calculate_T_e_nf(NF)
    else:
        NF = 10*np.log10(F)
        Te = calculate_T_e_nf(NF)
    j = G/10
    G = 10**j
    N = G*(calc_N(Tr,B)+calc_N(Te,B))
    return N

def amplifier_1_noT0(B,G,F,T=np.nan, Ftype='dB'):
    if(Ftype=='dB'):
        NF = F
        p = F/10
        F = 10**p
    else:
        NF = 10*np.log10(F)
    if(T==np.nan):
        T = calculate_T_e_nf(NF)
    j = G/10
    G = 10**j
    N = co.k * T *B*G
    return N



def amplifier_2_N(G_arr, T_arr, B, T0=290):#denne kan også brukes når man ser på antenner, sett da T0=T_A
    T_cas = calculate_Tcas(G_arr,T_arr)
    
    G = 1
    for i in range(len(G_arr)):
        G *= 10**(G_arr[i]/10)
    N = co.k * B * G*(T0+T_cas)
    print(N)
    return N

def calculate_Vrms(S_in, Z):#change to W before

    vrms = np.sqrt(S_in*Z)
    return vrms
def calculate_P_in_sin(vrms,Z): #beregn p_in når spissverdi og Z er gitt
    p_in = vrms**2 /(2*Z)
    return p_in


#AD konverter
def AD_SNR_max_dB(bits):
    SNR = 1.76+6.02*bits #sjekk at dette stemmer med notatene
    print(SNR)
    return SNR
    


def make_array_from_input(delim=',', numpy=True, re_char=False, replace_dec_sign=False):
    ws = ' '
    arr_str = input('Input array >>> ')
    if replace_dec_sign:
        arr_str = arr_str.replace(',','.')
    
    if ws in arr_str and not delim in arr_str:
        arr_str = arr_str.replace(' ', ',')
    if ws in arr_str and delim in arr_str:
        arr_str.strip(' ')
    arr = arr_str.split(delim)
    
    if re_char:
        arr = findall(r"[-+]?\d*\.\d+|\d+", ' '.join(arr))
        
    arr = [float(i) for i in arr]
    if numpy:
        arr = np.array(arr)
    return arr

def estimate_tau_d_num(rel_un_t,rel_un_d):
    nrand = 10000
    tau = (1 + 2*rel_un_t*(np.random.rand(nrand) - 0.5))
    d = (1 + 2*rel_un_d*(np.random.rand(nrand) - 0.5))
    f = tau/d # Since c is constant, we can leave it out
    expect_f = 1 # Since both tau and d have expected value one
    max_error = np.max(np.abs(f - expect_f))/expect_f
    print(f"Maximum relative error is {100*max_error:.2f}%")

def calculate_fase_noise_PLL(fref=np.nan,N=np.nan,fout=np.nan):
    if (N==np.nan):
        fasenoise = 20*np.log10(fout/fref)
    else:
        fasenoise = 20 * np.log10(N)
    return fasenoise




if __name__ == '__main__':
    #x = make_array_from_input(delim=',',re_char=True,replace_dec_sign=True) #pass på at det ikke er noe \n (newline) i inputen.
    # g = [20,-10,16]
    # t = [calculate_T_e_nf(2.5),290,calculate_T_e_nf(8)]
    # Tcas = calculate_Tcas(g,t)
    # print(watt_to_dBm(amplifier_2_N(g,t,3*10**6,50)))
    x = calculate_P_in_sin(0.5,50)
    print(watt_to_dBm(x))
    #print(calculate_std(x))
    #confidence_interval_samp(x)
    #prediction_interval(x)
    #y = make_array_from_input(delim=',')
    #confidence_interval_samp(y)

    #calculate_uniform_std(1000,0.01)
    
    pass
    
    




    
    


