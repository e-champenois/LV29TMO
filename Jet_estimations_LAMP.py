import numpy as np
# Diameter of nozzle opening:
d_nozzle = 0.01 # [cm]
# Diameter of skimmer 2:
d_skimmer2 = 0.1 # [cm]
# Distance of skimmer 1 from nozzle opening:
x_skimmer1 = 1.2 # [cm]
# Distance of skimmer 2 from nozzle opening:
x_skimmer2 = x_skimmer1+5.35*2.54 # [cm]
d_skimmer1 = d_nozzle + (d_skimmer2-d_nozzle)/x_skimmer2*x_skimmer1
print('Diameter of cone defined by nozzle and skimmer 2 at skimmer 1:',d_skimmer1, 'cm')
print('=> Cone is not determined by skimmer1 opening (0.1 cm)')

# Assumed spot size of x-rays in interaction region (I only know that optical laser is 40 microns)
d_x = 2.0E-5*1E2 # [cm]
x_interaction = x_skimmer2+17 # [cm]
d_M = d_nozzle+(d_skimmer2-d_nozzle)/x_skimmer2*x_interaction # [cm]
print('Molecular beam diameter in interaction region: {:0.4e} cm'.format( d_M))

V_interaction = np.square(d_x)/4.0*np.pi*d_M # [cm^3]
print('Interaction volume: {:0.4e} cm^3'.format(V_interaction))

# Vapor pressure of acetylacetone at ambient conditions:
p_AcAc = 6.6E2 # [Pa]
print('Vapor pressure of acetylacetone at ambient conditions: {:0.4e} Pa'.format(p_AcAc))

# Diameter of sample cone at skimmer 1:
alpha = np.pi/180*25
d_cone = (0.5*d_nozzle+x_skimmer1*np.tan(alpha))*2
print('Diameter of sample cone at skimmer 1: {:0.4e} cm'.format(d_cone))

# Acetylacetone pressure at skimmer1:
p_AcAc_skimmer1 = p_AcAc*d_nozzle**2/d_cone**2
print('Partial pressure of acetylacetone at skimmer 1: {:0.4e} Pa'.format(p_AcAc_skimmer1) )

# Acetylacetone pressure in interaction region:
p_AcAc_inter = p_AcAc_skimmer1*d_skimmer1**2/d_M**2
print('Partial pressure of acetylacetone in interaction region: {:0.4e} Pa'.format(p_AcAc_inter) )

# Acetylacetone number density:
N_A = 6.022E23 # [mol^-1]
R = 8.314 # [J/(mol*K)
T = 293 # [K]
ND_AcAc_inter = p_AcAc_inter*N_A/(R*T)*1E-6 # [cm^-3]
print('Number density of acetylacetone in interaction region: {:0.4e} cm^-3'.format(ND_AcAc_inter))

# Molecules in interaction region:
N_AcAc_inter = ND_AcAc_inter*V_interaction
print('Number of acetylacetone molecules in interaction region: {:.4e}'.format(N_AcAc_inter))

# Excitation fraction:
N_photons = 1E11
xsec = 1E-18
A = (d_x/2)**2*np.pi
f = N_photons*xsec/A
print('Excitation probability per molecule: {:0.4e}'.format(f))

# Excited molecules:
N_AcAc_Ex = N_AcAc_inter*f
print('Number of excited molecules in interaction region: {:0.4e}'.format(N_AcAc_Ex))
