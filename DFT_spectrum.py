import numpy as np

file_path_300 = "/home/luke/dipoleData/water/dipoles.dat"


data_list_300 = []


with open(file_path_300, 'r') as file:
    for line in file:
        
        values = line.split()
        
        
        row_array = np.array([float(val) for val in values])*(1/0.2081943)
        
        
        data_list_300.append(row_array)


for row in data_list_300[:5]:
    print(row)

#these were electric dipole moments in units of eA but we have converted them to debye 
    

import numpy as np

file_path_400 = "/home/luke/Desktop/water_400K_/dipoles_400.txt"


data_list_400 = []


with open(file_path_400, 'r') as file:
    for line in file:
        
        values = line.split()
        
        
        row_array = np.array([float(val) for val in values])*(1/0.2081943)
        
        
        data_list_400.append(row_array)


for row in data_list_400[:5]:
    print(row)


t_array_300 = np.arange(0,len(data_list_300),1) #this is in femto seconds

t_array_400 = np.arange(0,len(data_list_400),1) #this is in femto seconds

Tmax = min(len(t_array_300), len(t_array_400))  # Use the minimum length to avoid index out of range

I_list_300 = []
data_list_new_300 = np.array(data_list_300)

I_list_400 = []
data_list_new_400 = np.array(data_list_400)

for t300, t400 in zip(t_array_300[:Tmax], t_array_400[:Tmax]):
    print('t = ', t300, t400)

    I0_300 = 0

    if t300 == 0:
        I0_300 = np.sum(data_list_new_300 ** 2)
    else:
        I0_300 = np.sum(data_list_new_300[:-t300] * data_list_new_300[t300:], axis=0).sum()

    I_list_300.append(I0_300)

    I0_400 = 0

    if t400 == 0:
        I0_400 = np.sum(data_list_new_400 ** 2)
    else:
        I0_400 = np.sum(data_list_new_400[:-t400] * data_list_new_400[t400:], axis=0).sum()

    I_list_400.append(I0_400)

normalise_300 = np.arange(1,len(I_list_300)+1, 1)

normalise_300 = 1/normalise_300

normalise_300 = np.flip(normalise_300)

normalise_400 = np.arange(1,len(I_list_400)+1, 1)

normalise_400 = 1/normalise_400

normalise_400 = np.flip(normalise_400)

I_array_new_300 = np.array(I_list_300)*normalise_300

I_array_new_400 = np.array(I_list_400)*normalise_400

#now that we have taken the ensemble average we can take the fourier transform 

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


fft_result_300 = fft(I_array_new_300)  #this is in the units of Debye squared * 1/whatever unit of frequency we use. Returns FFT of our data 



N_300 = len(fft_result_300)



spacing_seconds_300 = (t_array_300[1] - t_array_300[0])/1e15        #this spacing is in femto second. lets convert it to second



frequencies_300 = fftfreq(N_300, d = spacing_seconds_300)
#This will return the discrete fourier transofmr sample frequenies

#N is how many frequencies you want. d is the sample spacing
#The returned float array f contains the frequency bin centers in cycles 
# per unit of the sample spacing (with zero at the start). 
# For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.


magnitude_300 = np.abs(fft_result_300)
#this should now be in units of Debye squared / Hz



fft_result_400 = fft(I_array_new_400)

N_400 = len(fft_result_400)

spacing_seconds_400 = (t_array_400[1] - t_array_400[0])/1e15  

frequencies_400 = fftfreq(N_400, d = spacing_seconds_400)

magnitude_400 = np.abs(fft_result_400)

c = 2.998e10   #in units of cm/s


mask_300 = frequencies_300 != 0.0 #just get rid of any that are 0

#we are going to convert our frequencies to wave number

wave_number_300 = frequencies_300[mask_300]/(c)

new_magnitude_300 = magnitude_300[mask_300]


plt.plot(wave_number_300, new_magnitude_300, label = '300 K')
plt.title('Magnitude of the FFT')
plt.xlabel('Wavenumber$  (cm^{-1}$)')
plt.ylabel('Magnitude (Arb units)')


mask_400 = frequencies_400 != 0.0 #just get rid of any that are 0

#we are going to convert our frequencies to wave number

wave_number_400 = frequencies_400[mask_400]/(c)

new_magnitude_400 = magnitude_400[mask_400]


plt.plot(wave_number_400, new_magnitude_400, label = '400 K')
plt.legend()
plt.xlim(0,10000)
plt.ylim(0,100)

plt.figure()
plt.subplot(211)
plt.plot(wave_number_300, new_magnitude_300, label = '300 K')
plt.xlim(0,6000)
plt.ylim(0,30)
plt.grid(True)
plt.legend()
plt.title('IR spectrum of single gas-phase water molecule')

plt.subplot(212)
plt.plot(wave_number_400, new_magnitude_400, label = '400 K', color = 'orange')
plt.xlim(0,6000)
plt.ylim(0,30)
plt.legend()
plt.xlabel('Wavenumber$  (cm^{-1}$)')
plt.ylabel('Magnitude (Arb units)')

plt.grid(True)

plt.show()

