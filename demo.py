import numpy as np
from scipy.constants import G, c
from tensorflow.keras.models import save_model,load_model

def simualte_single_wave(m1,m2,dist,GW=True):
    """Generate a single wave a gravitational wave signal for a binary black hole system with specified masses and distance.
    
    Args:
        m1 (float): Mass of the first black hole in solar masses.
        m2 (float): Mass of the second black hole in solar masses.
        dist (float): Distance to the binary system in megaparsecs.
        GW (bool):(Default:True) True to generate gravitational wave, false to generate non-gravitational waves
        
    Returns:
        numpy.ndarray: Gravitational wave signal for the binary black hole system.
    """
    num_samples = 4096
    if GW:
        
        sample_rate = 4096
        delta_t = 1 / sample_rate
        t = np.arange(num_samples) * delta_t

        wave=gw_waveform(t,m1,m2,dist)
        
    else:
        wave = np.random.normal(0, 1, num_samples)
    
    return wave

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


def Testing_model(clas_model,reg_model,wave):
    """Testing the accuracy of the models
    
    Args:
        clas_model(tensorflow.python.keras.engine.sequential.Sequential): model to classify the gravitational waves and non-gravitational waves
        reg_model(tensorflow.python.keras.engine.sequential.Sequential): model to estimate the parameter from the waveform
        wave(numpy.ndarray):wave signal
    """
    wave=wave[..., np.newaxis]
    wave=wave.reshape(1,4096,1)
    if clas_model.predict(wave)>0.5:
        print('It is a gravitational wave')
        parameter=reg_model.predict(wave)[0]
        print(f'paramters of the wave are \nM1:{parameter[0]}\nM2:{parameter[1]}\nD:{parameter[2]}')
    else:
        print('It is not a gravitational wave')

wave=simualte_single_wave(20,40,70)

clas_model=load_model('clas_model.h5')
clas_model.summary()

reg_model=load_model('reg_model.h5')
reg_model.summary()

Testing_model(clas_model,reg_model,wave)