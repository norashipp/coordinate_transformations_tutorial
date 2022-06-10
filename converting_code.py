
# Lina Necib, Jan 8, 2018, Caltech, 
# Trying to analyze the data set compiled from SDSS-GAIA
# Will be using the extreme deconvolution methods from Jo Bovy

# April 25, 2018, NYC
# Cleaning up the dataset from Gaia DR2

# get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib 
import matplotlib as mpl
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from uncertainties import unumpy
from pylab import *
from scipy.special import erf
import pandas as pd
import os


import astropy.coordinates as coord
import astropy.units as u
#import gala.coordinates as gc


# Importing data
input_dir = '/project/projectdirs/theory/ianmoult/mw_potential_clean/mw_potential/data/'
data_dir = '/project/projectdirs/theory/lnecib/mw_potential/data/'
plots_dir = '../plots/dr2/'


print "About to start :-) "

dataset = pd.read_csv(input_dir + 'gaia_dr2_all.csv', usecols = ['source_id', 'ra', 'dec', 'pmra', 'pmra_error', 'parallax', 'parallax_error', 'l', 'b', 'pmdec', 'pmdec_error', 'rv_template_fe_h', 'pmra_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr', 'radial_velocity', 'radial_velocity_error', 'parallax_over_error'])

print len(dataset)
# Clean ups!

dataset.replace('NOT_AVAILABLE', np.nan, inplace=True)
broken = np.where(dataset['pmra_error'].values<0)[0]

print len(dataset)
dataset.drop(broken, inplace=True)
print len(dataset)


# Organizing data:
parallax = dataset['parallax'].values
distance = 1.0/parallax

parallax_err = dataset['parallax_error'].values
distance_err = abs(parallax_err/parallax**2)

distance_all = unumpy.uarray(distance, distance_err)

l = dataset['l'].values
b = dataset['b'].values
feH = dataset['rv_template_fe_h'].values #np.random.normal(-0.5, 1.5, len(l)) # I'm gonna put random numbers here

ra = dataset['ra'].values
dec = dataset['dec'].values

pmra = dataset['pmra'].values
pmdec = dataset['pmdec'].values
pm_ra_errors = dataset['pmra_error'].values
pm_dec_errors = dataset['pmdec_error'].values
pmra_pmdec_corr = dataset['pmra_pmdec_corr'].values
correlation_pmra_parallax = dataset['parallax_pmra_corr'].values
correlation_pmdec_parallax = dataset['parallax_pmdec_corr'].values
source_id_list = np.array(dataset['source_id'], dtype = str)

vrad = np.array(dataset['radial_velocity'].values, dtype=float)
vrad_errors = np.array(dataset['radial_velocity_error'].values)

vr_all = unumpy.uarray(vrad, vrad_errors)
pmra_all = unumpy.uarray(pmra, pm_ra_errors)
pmdec_all = unumpy.uarray(pmdec, pm_dec_errors)


# Getting the cartesian coordinates here

# c = coord.SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=distance*u.kpc,  frame='icrs')



# solar_positions = np.array([-8, 0, 0.015])
# solar_velocities = np.array([11.1, 239.08, 7.25])

# galactic_coordinates = c.transform_to(coord.Galactocentric(galcen_distance=8*u.kpc,z_sun=15. * u.pc)) 

# x = np.array(galactic_coordinates.x)
# y = np.array(galactic_coordinates.y)
# z = np.array(galactic_coordinates.z)



##### 
rSun = 8.0
x_shift = rSun
y_shift = 0.
z_shift = 0.015 #kpc Double check this

x_list_error = distance_all * np.cos(np.radians(b)) * np.cos(np.radians(l)) + x_shift
y_list_error = distance_all * np.cos(np.radians(b)) * np.sin(np.radians(l)) + y_shift
# Getting the z_values here
z_list_error = distance_all * np.sin(np.radians(b)) + z_shift 

z_list_values = unumpy.nominal_values(z_list_error)
z_list_std = unumpy.std_devs(z_list_error)

x = unumpy.nominal_values(x_list_error)
y = unumpy.nominal_values(y_list_error)
z = unumpy.nominal_values(z_list_error)

x_err = unumpy.std_devs(x_list_error)
y_err = unumpy.std_devs(y_list_error)
z_err = unumpy.std_devs(z_list_error)

radial_dis = np.linalg.norm([x, y, z], axis = 0)
radial_err = (np.abs(x*x_err) + np.abs(y*y_err) + np.abs(z*z_err) )/radial_dis
# vU, vV, vW

factor = 4.74 #*1e-3 factor off here 

d_pm_ra = distance_all*pmra_all*factor 
print d_pm_ra[0:5]
d_pm_de = distance_all*pmdec_all*factor 

ra = np.radians(ra)
dec = np.radians(dec)


# In[ ]:

a_matrix = np.zeros((len(ra),3,3))
a1_matrix = np.zeros((len(ra),3,3))
a2_matrix = np.zeros((len(ra),3,3))


for i in range(len(a1_matrix)):
    a1_matrix[i] = [[np.cos(ra[i]), -np.sin(ra[i]), 0],[np.sin(ra[i]), np.cos(ra[i]), 0],[0,0,1]]
    a2_matrix[i] = [[np.cos(dec[i]),0,-np.sin(dec[i])],[0,1,0],[np.sin(dec[i]), 0, np.cos(dec[i])]]
    a_matrix[i] = np.dot(a1_matrix[i], a2_matrix[i])


# In[ ]:

theta = np.radians(123)
delta = np.radians(27.4)
alpha = np.radians(192.25)

t1_matrix = [[np.cos(theta), np.sin(theta), 0],[np.sin(theta), -np.cos(theta),0],[0,0,1]]
t2_matrix = [[-np.sin(delta),0,np.cos(delta)],[0,1,0],[np.cos(delta), 0, np.sin(delta)]]
t3_matrix = [[np.cos(alpha), np.sin(alpha),0],[-np.sin(alpha), np.cos(alpha),0],[0,0,1]]

t_matrix = np.dot(np.dot(t1_matrix, t2_matrix), t3_matrix)




vU = np.zeros(len(dec))
vV = np.zeros(len(dec))
vW = np.zeros(len(dec))


vU_error = np.zeros(len(dec))
vV_error = np.zeros(len(dec))
vW_error = np.zeros(len(dec))

covariance_UVW = np.zeros((len(dec), 3, 3)) 

for i in range(len(dec)):
    matrix = np.dot(t_matrix, a_matrix[i])
    matrix_UVA = np.dot(matrix, [vr_all[i], d_pm_ra[i], d_pm_de[i] ])
    vU[i] = matrix_UVA[0].nominal_value
    vV[i] = matrix_UVA[1].nominal_value
    vW[i] = matrix_UVA[2].nominal_value

    # Getting the correlation.... which is a nightmare
    pmra_val =  pmra[i]
    pmra_s = pm_ra_errors[i]

    pmde_val = pmdec[i]
    pmde_s = pm_dec_errors[i]

    dis = distance[i]
    dis_s = distance_err[i]

    corr_mu_mu = pmra_pmdec_corr[i]*pmra_s*pmde_s

    # if parallax:
    corr_d_alpha = correlation_pmra_parallax[i]*dis_s*pmra_s 
    corr_d_delta = correlation_pmdec_parallax[i]*dis_s*pmde_s
    # else:
    #     corr_d_alpha = 0
    #     corr_d_delta = 0

    vr_s = vrad_errors[i]


    # Now let's multiply the matrices for the error propagation
    covariance_d_alpha_delta = np.array([[dis_s**2, corr_d_alpha, corr_d_delta ],[corr_d_alpha, pmra_s**2, corr_mu_mu],[corr_d_delta, corr_mu_mu, pmde_s**2] ])
    bmatrix = factor*np.array([[pmra_val, dis, 0],[pmde_val, 0, dis]])

    covariance_valpha_vdelta = np.dot(np.dot(bmatrix, covariance_d_alpha_delta), np.transpose(bmatrix))

    covariance_vr_valpha_vdelta = np.array([[vr_s**2,0,0],[0,covariance_valpha_vdelta[0][0], covariance_valpha_vdelta[0][1] ],[0,covariance_valpha_vdelta[1][0], covariance_valpha_vdelta[1][1]]])


    covariance_UVW[i] = np.dot(np.dot(matrix, covariance_vr_valpha_vdelta), np.transpose(matrix) )



for i in range(len(covariance_UVW)):
    if covariance_UVW[i][0][0] >= 0:
        vU_error[i] = np.sqrt(covariance_UVW[i][0][0]) 
    else:
        vU_error[i] = float('NaN')

    if covariance_UVW[i][1][1] >= 0:
        vV_error[i] = np.sqrt(covariance_UVW[i][1][1]) 
    else:
        vV_error[i] = float('NaN')

    if covariance_UVW[i][2][2] >= 0:
        vW_error[i] = np.sqrt(covariance_UVW[i][2][2]) 
    else:
        vW_error[i] = float('NaN')



# In[21]:

solar_velocities = np.array([11.1, 239.08, 7.25])

vU_all = unumpy.uarray(vU, vU_error)
vV_all = unumpy.uarray(vV, vV_error)
vW_all = unumpy.uarray(vW, vW_error)

vU_shifted = vU_all + solar_velocities[0]
vV_shifted = vV_all + solar_velocities[1]
vW_shifted = vW_all + solar_velocities[2]


# absolute magnitude of velocity and its error

vabs = np.zeros_like(vU)
vabs_err = np.zeros_like(vU)
vabs_err_nocov = np.zeros_like(vU)

# using full UVW covariance matrix, compute vabs and error on vabs.
for i in range(len(dec)):
    vmean = np.array([vU[i]+ solar_velocities[0], vV[i]+ solar_velocities[1], vW[i]+ solar_velocities[2]])
    Nvmean = np.linalg.norm(vmean)
    dv = np.array([vmean[0],vmean[1],vmean[2]])/Nvmean
    vabs[i] = Nvmean
    vabs_err[i] = np.sqrt(np.dot( dv, np.dot(covariance_UVW[i],dv)))
    vabs_err_nocov[i] = np.sqrt( dv[0]**2*vU_error[i]**2 + 
           dv[1]**2*vV_error[i]**2 +  dv[2]**2*vW_error[i]**2   )

print " max absolute velocity: ", max(vabs)


# In[22]:

r = radial_dis
phi = np.arctan2(y,x)
theta = np.arccos(z/r)


print "Double check distances in kpc or pc!!!! Assuming kpc right now!"

r_helio = np.linalg.norm(np.transpose([x+rSun, y, z]), axis=1)
print "r_helio is ", r_helio[0:10]
# In[23]:

vr = np.array(vU_shifted*np.cos(phi)*np.sin(theta) + vV_shifted*np.sin(phi)*np.sin(theta) + vW_shifted*np.cos(theta) )
vphi = np.array(-vU_shifted*np.sin(phi) + vV_shifted*np.cos(phi) )
vtheta = np.array(vU_shifted*np.cos(phi)*np.cos(theta) + vV_shifted*np.sin(phi)*np.cos(theta) - vW_shifted*np.sin(theta) )

vr_nom = unumpy.nominal_values(vr)
vtheta_nom = unumpy.nominal_values(vtheta)
vphi_nom = unumpy.nominal_values(vphi)

vr_std = unumpy.std_devs(vr)
vtheta_std = unumpy.std_devs(vtheta)
vphi_std = unumpy.std_devs(vphi)

dictionary = {}
dictionary["vr"] = vr_nom
dictionary["vtheta"] = vtheta_nom
dictionary["vphi"] = vphi_nom
dictionary["vr_err"] = vr_std
dictionary["vtheta_err"] = vtheta_std
dictionary["vphi_err"] = vphi_std
dictionary["feH"] = feH
dictionary["x"] = x
dictionary["y"] = y
dictionary["z"] = z

dictionary["r"] = radial_dis
dictionary["r_err"] = radial_err

dictionary["r_helio"] = r_helio
dictionary["x_err"] = x_err
dictionary["y_err"] = y_err
dictionary["z_err"] = z_list_std
dictionary["vU"] = vU
dictionary["vV"] = vV
dictionary["vW"] = vW
dictionary["vU_err"] = vU_error
dictionary["vV_err"] = vV_error
dictionary["vW_err"] = vW_error

dictionary["plx_over_error"] = dataset['parallax_over_error'].values
dictionary["v_radial"] = vrad
dictionary["v_radial_error"] = vrad_errors
dictionary["pmra"] = pmra 
dictionary["pmdec"] = pmdec

dictionary["vabs"] = vabs
dictionary["vabs_err"] = vabs_err
dictionary["vabs_err_nocov"] = vabs_err_nocov

dictionary["source_id"] = source_id_list #.asstr()
dictionary["ra"] = dataset['ra'].values 
dictionary["dec"] = dataset['dec'].values 
dictionary["l"] = dataset['l'].values 
dictionary["b"] =  dataset['b'].values 

# Construct a pandas dataframe
df = pd.DataFrame(data=dictionary)
df.to_pickle("../data/gaia_dr2_velocities.pkl")
