import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#path = 'bp_filter_2mHz_2MHz.csv'
path = 'dc-filter.csv'
df = pd.read_csv(path, engine='python')
freq = df['Frequency (Hz)'].dropna(axis=0, how='all')
chan1 = df['Channel 1 Magnitude (dB)'].dropna(axis=0, how='all')
chan2 = df['Channel 2 Magnitude (dB)'].dropna(axis=0, how='all')
phase = df['Channel 2 Phase (°)'].dropna(axis=0, how='all')

threshold = phase < 0
phase[threshold] = 360+phase[threshold]

#plt.plot(np.log(freq),chan2)
# fig = plt.figure()
# plt.subplot(2,1,1)

plt.title('Bodeplot from DC filter')
plt.plot(freq[:100],chan2[:100], label='Channel 2')
plt.plot(freq[:100],chan1[:100], label='Channel 1')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.xscale('log')
plt.legend()
plt.grid('on')

plt.savefig('dcfilter_ish1000hz.png',dpi=200)

# plt.subplot(2,1,2)
# #fig.tight_layout(pad=0.01)
# plt.plot(freq,phase, color='orange')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Phase (°)')
# plt.xscale('log')
# plt.grid('on')

#plt.show()