import numpy as np
import os
import pandas as pd
from scipy.constants import G, c
import matplotlib.pyplot as plt
from IPython.display import clear_output

def gw_waveform(t, m1, m2, dist):
    """Generate a gravitational wave signal for a binary black hole system with specified masses and distance.
    
    Args:
        t (numpy.ndarray): Time array for the waveform.
        m1 (float): Mass of the first black hole in solar masses.
        m2 (float): Mass of the second black hole in solar masses.
        dist (float): Distance to the binary system in megaparsecs.
        
    Returns:
        numpy.ndarray: Gravitational wave signal for the binary black hole system.
    """
    # Convert input parameters to SI units
    m1_kg = m1 * 1.989e30
    m2_kg = m2 * 1.989e30
    dist_m = dist * 3.086e22
    
    # Calculate total mass and reduced mass of the binary system
    m_tot = m1_kg + m2_kg
    mu = m1_kg * m2_kg / m_tot
    
    # Calculate chirp mass and dimensionless mass ratio
    m_chirp = mu**(3/5) * m_tot**(2/5)
    q = m2_kg / m1_kg
    
    # Calculate frequency and amplitude of the waveform
    f = np.linspace(10, 1000, len(t))
    h_amp = np.sqrt(5 * np.pi / 24) * (G * m_chirp / c**2)**(5/6) / (dist_m * (np.pi * f)**(2/3))
    
    # Calculate phase and polarization of the waveform
    phi_c = np.pi / 4 + (5 / 256) * (G * m_tot / c**3)**(-5/3) * (2 * np.pi * f)**(-5/3)
    psi = np.random.uniform(-np.pi, np.pi)
    phi = 2 * np.pi * f * t + phi_c
    
    # Calculate the waveform
    h_plus = h_amp * (1 + np.cos(psi)**2) * np.cos(phi)
    h_cross = 2 * h_amp * np.cos(psi) * np.sin(phi)
    h = h_plus + 1j * h_cross
    
    return h.real

def simulate_data(num_samples, sample_rate, m1_range, m2_range, dist_range, gw_prob,path=''):
    """Simulate gravitational wave and non-gravitational wave data with specified parameters.
    
    Args:
        num_samples (int): Number of samples in each waveform.
        sample_rate (float): Sampling rate in Hz.
        m1_range (tuple): Range of masses for the first black hole in solar masses.
        m2_range (tuple): Range of masses for the second black hole in solar masses.
        dist_range (tuple): Range of distances to the binary system in megaparsecs.
        gw_prob (float): Probability of a waveform being a gravitational wave.
        
    Returns:
        numpy.ndarray: Array of waveforms.
        numpy.ndarray: Array of labels for the waveforms (0 for non-gravitational waves, 1 for gravitational waves).
        numpy.ndarray: Array of tuples containing the masses and distance for each waveform.
    """
    # Calculate time array for waveforms
    delta_t = 1 / sample_rate
    t = np.arange(sample_rate) * delta_t
    
    # Initialize arrays for waveforms, labels, and parameters
    waveforms = np.zeros((0, sample_rate))
    waveforms_ = np.zeros((0, sample_rate))
    labels = np.zeros(0)
    params = np.zeros((0, 3))

    path=os.getcwd()
    
    ld=os.listdir()
    if "GW" not in ld:
        os.makedirs('GW')
    if "NGW" not in ld:
        os.makedirs('NGW')
    
    df=pd.DataFrame([],columns=['file','m1','m2','d'])
    # Simulate waveforms
    

    for i in range(num_samples):
        if i%100==0:
            clear_output(wait=True)
            print(f'The waves {i} waves are generated successfully')
         # Randomly choose whether to generate a gravitational wave or non-gravitational wave
        if np.random.random() < gw_prob:
            # Generate a gravitational wave
            m1 = np.random.uniform(*m1_range)
            m2 = np.random.uniform(*m2_range)
            dist = np.random.uniform(*dist_range)
            wave = gw_waveform(t, m1, m2, dist)
            noise= np.random.normal(0, 1, sample_rate)
            waveform=wave+noise
            waveform_=wave+noise
            label = 1
            params = np.concatenate([params, np.array([[m1, m2, dist]])])
            waveforms_ = np.concatenate([waveforms_, waveform_[np.newaxis,:]])
            np.save(path+f'Data/GW/{i}.npy',waveform)
            path_=path+f'{i}.npy'
            df.loc[len(df.index)]=[path_,m1,m2,dist]
        else:
            # Generate a non-gravitational wave
            waveform = np.random.normal(0, 1, sample_rate)
            label = 0
            np.save(path+f'Data/NGW/{i}.npy',waveform)
        waveforms = np.concatenate([waveforms, waveform[np.newaxis,:]])
        labels = np.concatenate([labels, np.array([label])])

        
    plt.plot(waveform)
        
    print(f'The waves {num_samples} waves are generated successfully')
    df.to_csv(path+'labels.csv')

if __name__=="main":
    # Simulate data
    waveform,labels,waveform_,param=simulate_data(20000,4096,(10,50),(10,50),(50,100),0.5)
    
