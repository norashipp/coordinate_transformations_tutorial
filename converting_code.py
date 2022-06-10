######################################
# Based on code from Lina Necib 2018 #
######################################

import numpy as np

import os

from scipy.special import erf

import pandas as pd

from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u

from uncertainties import unumpy

# from pylab import * # Is this needed?


#################################
## UPDATE DIRECTORY NAMES HERE ##
#################################
input_dir = '/project/projectdirs/theory/ianmoult/mw_potential_clean/mw_potential/data/'
data_dir = '/project/projectdirs/theory/lnecib/mw_potential/data/'
output_file = 'gaia_dr3_velocities.pkl'

print('input_dir = %s' % input_dir)
print('data_dir = %s' % data_dir)
print('outfile = %s' % output_file)

#######################

print()
print('Loading data...')
######################
# CHECK COLUMN NAMES #
######################
dataset = pd.read_csv(input_dir + 'gaia_dr2_all.csv', usecols=['source_id', 'ra', 'dec', 'pmra', 'pmra_error', 'parallax', 'parallax_error', 'l', 'b', 'pmdec',
                                                               'pmdec_error', 'rv_template_fe_h', 'pmra_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr', 'radial_velocity', 'radial_velocity_error', 'parallax_over_error'])
print('%i rows' % len(dataset))

print()
print('Cleaning data...')
dataset.replace('NOT_AVAILABLE', np.nan, inplace=True)
broken = np.where(dataset['pmra_error'].values < 0)[0]

print('%i rows' % len(dataset))
dataset.drop(broken, inplace=True)
print('%i rows' % len(dataset))

#######################

print()
print('Organizing data...')

# SOURCE IDS
source_id_list = np.array(dataset['source_id'], dtype=str)

# POSITIONS

# RA DEC
ra = dataset['ra'].values
dec = dataset['dec'].values

# GALACTIC COORDINATES
l = dataset['l'].values
b = dataset['b'].values

# DISTANCE
parallax = dataset['parallax'].values
distance = 1.0 / parallax
parallax_err = dataset['parallax_error'].values
distance_err = abs(parallax_err / parallax**2)
distance_all = unumpy.uarray(distance, distance_err)

# PROPER MOTIONS (AND ERRORS)
pmra = dataset['pmra'].values
pmdec = dataset['pmdec'].values
pm_ra_errors = dataset['pmra_error'].values
pm_dec_errors = dataset['pmdec_error'].values
pmra_pmdec_corr = dataset['pmra_pmdec_corr'].values
correlation_pmra_parallax = dataset['parallax_pmra_corr'].values
correlation_pmdec_parallax = dataset['parallax_pmdec_corr'].values
pmra_all = unumpy.uarray(pmra, pm_ra_errors)
pmdec_all = unumpy.uarray(pmdec, pm_dec_errors)

# RADIAL VELOCITY
vrad = np.array(dataset['radial_velocity'].values, dtype=float)
vrad_errors = np.array(dataset['radial_velocity_error'].values)
vr_all = unumpy.uarray(vrad, vrad_errors)

# METALLICITY
feH = dataset['rv_template_fe_h'].values


##################################

print()
print('Converting coordinates...')

# Sun

# 2018
# rSun = 8.0  # kpc
# x_shift = rSun
# y_shift = 0.  # kpc
# z_shift = 0.015  # kpc

# astropy 4.0
rSun = 8.122  # kpc
x_shift = rSun
y_shift = 0.  # kpc
z_shift = 0.0208  # kpc
print('Using solar parameters rsun = %.1f, x_shift = %.1f, y_shift = %.1f, z_shift = %.1f' % (rSun, x_shift, y_shift, z_shift))

# Converting...
x_list_error = distance_all * np.cos(np.radians(b)) * np.cos(np.radians(l)) + x_shift
y_list_error = distance_all * np.cos(np.radians(b)) * np.sin(np.radians(l)) + y_shift
z_list_error = distance_all * np.sin(np.radians(b)) + z_shift

z_list_values = unumpy.nominal_values(z_list_error)
z_list_std = unumpy.std_devs(z_list_error)

x = unumpy.nominal_values(x_list_error)
y = unumpy.nominal_values(y_list_error)
z = unumpy.nominal_values(z_list_error)

x_err = unumpy.std_devs(x_list_error)
y_err = unumpy.std_devs(y_list_error)
z_err = unumpy.std_devs(z_list_error)

radial_dis = np.linalg.norm([x, y, z], axis=0)
radial_err = (np.abs(x * x_err) + np.abs(y * y_err) + np.abs(z * z_err)) / radial_dis

# vU, vV, vW
print()
print('Calculating vU, vV, vW...')

# CHECK NUMBERS
# factor = 4.74  # *1e-3 factor off here
factor = 4.74047
print('Using factor = %.5f' % factor)


d_pm_ra = distance_all * pmra_all * factor
d_pm_de = distance_all * pmdec_all * factor
print(d_pm_ra[0:5])

# Convert ra, dec to radians
ra = np.radians(ra)
dec = np.radians(dec)

a_matrix = np.zeros((len(ra), 3, 3))
a1_matrix = np.zeros((len(ra), 3, 3))
a2_matrix = np.zeros((len(ra), 3, 3))

for i in range(len(a1_matrix)):
    a1_matrix[i] = [[np.cos(ra[i]), -np.sin(ra[i]), 0], [np.sin(ra[i]), np.cos(ra[i]), 0], [0, 0, 1]]
    a2_matrix[i] = [[np.cos(dec[i]), 0, -np.sin(dec[i])], [0, 1, 0], [np.sin(dec[i]), 0, np.cos(dec[i])]]
    a_matrix[i] = np.dot(a1_matrix[i], a2_matrix[i])


# CHECK NUMBERS
theta = np.radians(123)
delta = np.radians(27.4)
alpha = np.radians(192.25)
print('Using theta = %.0f, detla = %.1f, alpha = %.2f' % (theta, delta, alpha))

t1_matrix = [[np.cos(theta), np.sin(theta), 0], [np.sin(theta), -np.cos(theta), 0], [0, 0, 1]]
t2_matrix = [[-np.sin(delta), 0, np.cos(delta)], [0, 1, 0], [np.cos(delta), 0, np.sin(delta)]]
t3_matrix = [[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]]

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
    matrix_UVA = np.dot(matrix, [vr_all[i], d_pm_ra[i], d_pm_de[i]])
    vU[i] = matrix_UVA[0].nominal_value
    vV[i] = matrix_UVA[1].nominal_value
    vW[i] = matrix_UVA[2].nominal_value

    # Getting the correlation.... which is a nightmare
    pmra_val = pmra[i]
    pmra_s = pm_ra_errors[i]

    pmde_val = pmdec[i]
    pmde_s = pm_dec_errors[i]

    dis = distance[i]
    dis_s = distance_err[i]

    corr_mu_mu = pmra_pmdec_corr[i] * pmra_s * pmde_s

    # if parallax:
    corr_d_alpha = correlation_pmra_parallax[i] * dis_s * pmra_s
    corr_d_delta = correlation_pmdec_parallax[i] * dis_s * pmde_s
    # else:
    #     corr_d_alpha = 0
    #     corr_d_delta = 0

    vr_s = vrad_errors[i]

    # Now let's multiply the matrices for the error propagation
    covariance_d_alpha_delta = np.array([[dis_s**2, corr_d_alpha, corr_d_delta], [corr_d_alpha, pmra_s**2, corr_mu_mu], [corr_d_delta, corr_mu_mu, pmde_s**2]])
    bmatrix = factor * np.array([[pmra_val, dis, 0], [pmde_val, 0, dis]])

    covariance_valpha_vdelta = np.dot(np.dot(bmatrix, covariance_d_alpha_delta), np.transpose(bmatrix))

    covariance_vr_valpha_vdelta = np.array([[vr_s**2, 0, 0], [0, covariance_valpha_vdelta[0][0], covariance_valpha_vdelta[0][1]], [0, covariance_valpha_vdelta[1][0], covariance_valpha_vdelta[1][1]]])

    covariance_UVW[i] = np.dot(np.dot(matrix, covariance_vr_valpha_vdelta), np.transpose(matrix))


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


# solar_velocities = np.array([11.1, 239.08, 7.25]) # 2018
solar_velocities = np.array([12.9, 245.6, 7.78])  # astropy 4.0
print('Using solar velocities %.2f, %.2f, %.2f' % (solar_velocities[0], solar_velocities[1], solar_velocities[2]))

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
    vmean = np.array([vU[i] + solar_velocities[0], vV[i] + solar_velocities[1], vW[i] + solar_velocities[2]])
    Nvmean = np.linalg.norm(vmean)
    dv = np.array([vmean[0], vmean[1], vmean[2]]) / Nvmean
    vabs[i] = Nvmean
    vabs_err[i] = np.sqrt(np.dot(dv, np.dot(covariance_UVW[i], dv)))
    vabs_err_nocov[i] = np.sqrt(dv[0]**2 * vU_error[i]**2 +
                                dv[1]**2 * vV_error[i]**2 + dv[2]**2 * vW_error[i]**2)

print('Max absolute velocity = %.1f km/s' % (np.max(vabs)))

print()
print('Calculating v_r, v_theta, v_phi...')

r = radial_dis
phi = np.arctan2(y, x)
theta = np.arccos(z / r)

r_helio = np.linalg.norm(np.transpose([x + rSun, y, z]), axis=1)
print('r_helio = %.1f kpc' % r_helio)

# Velocities
vr = np.array(vU_shifted * np.cos(phi) * np.sin(theta) + vV_shifted * np.sin(phi) * np.sin(theta) + vW_shifted * np.cos(theta))
vphi = np.array(-vU_shifted * np.sin(phi) + vV_shifted * np.cos(phi))
vtheta = np.array(vU_shifted * np.cos(phi) * np.cos(theta) + vV_shifted * np.sin(phi) * np.cos(theta) - vW_shifted * np.sin(theta))

vr_nom = unumpy.nominal_values(vr)
vtheta_nom = unumpy.nominal_values(vtheta)
vphi_nom = unumpy.nominal_values(vphi)

vr_std = unumpy.std_devs(vr)
vtheta_std = unumpy.std_devs(vtheta)
vphi_std = unumpy.std_devs(vphi)

#################################

print()
print('Saving data...')

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

dictionary["source_id"] = source_id_list  # .asstr()
dictionary["ra"] = dataset['ra'].values
dictionary["dec"] = dataset['dec'].values
dictionary["l"] = dataset['l'].values
dictionary["b"] = dataset['b'].values

############################################

filename = data_dir + output_file
print()
print('Writing data to %s...' % filename)

# Construct a pandas dataframe
df = pd.DataFrame(data=dictionary)
df.to_pickle(filename)


print()
print('Done!')
