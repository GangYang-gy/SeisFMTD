# -------------------------------------------------------------------
# Utility functions for moment tensor coordinate transformations
# and gradient calculations.
#
# Author: Gang Yang
# Email: gangy.yang@mail.utoronto.ca
# -------------------------------------------------------------------

from pyrocko import moment_tensor as pmt
import numpy as np
from MTTools.DMomentTensors import DMT_enz
from pyCAPLunar.DCAPUtils import mag2moment

def mt2moment(mt):
    mt = ENU2NED(mt)
    m = pmt.values_to_matrix(mt)
    scalar_moment = pmt.MomentTensor(m = m).scalar_moment()

    return scalar_moment


def mt2sdrm(mt):
    mt = ENU2NED(mt)
    m = pmt.values_to_matrix(mt)
    sdr = pmt.MomentTensor(m = m).both_strike_dip_rake()[0]
    #mag = pmt.MomentTensor(m = m).magnitude
    moment = pmt.MomentTensor(m = m).scalar_moment()
    mag = (np.log10(moment)  - 9.1)/ 1.5
    mt = list(sdr)
    mt.append(mag)
    sdrm = np.array(mt)
    return sdrm

def ENU2NED(mt):
    new_mt = np.zeros_like(mt)
    new_mt[0]=mt[1]
    new_mt[1]=mt[0]
    new_mt[2]=mt[2]
    new_mt[3]=mt[3]
    new_mt[4]=-1*mt[5]
    new_mt[5]=-1*mt[4]
    return np.array(new_mt)  

def USE2ENU(mt):
    new_mt = np.zeros_like(mt)
    new_mt[0]=mt[2]
    new_mt[1]=mt[1]
    new_mt[2]=mt[0]
    new_mt[3]=-1*mt[5]
    new_mt[4]=mt[4]
    new_mt[5]=-1*mt[3]
    return new_mt
    
def NEZ2ENZ(SGTs):
    SGTs_new = SGTs.copy()
    SGTs_new[:, 0, :] = SGTs[:, 1, :]
    SGTs_new[:, 1, :] = SGTs[:, 0, :]
     
    return SGTs_new
 
def NED2ENU(mt):
    new_mt = np.zeros_like(mt)
    new_mt[0]=mt[1]
    new_mt[1]=mt[0]
    new_mt[2]=mt[2]
    new_mt[3]=mt[3]
    new_mt[4]=-1*mt[5]
    new_mt[5]=-1*mt[4]
    return new_mt

def ENU2USE(mt):

    new_mt = USE2ENU(mt)

    return new_mt


def grad_mt2sdr(grad_mt_ENU, sdrm):
    from numpy import sin, cos
    '''
    Aki,Keiiti, Quantitative Seismology, P112
    '''
    phi = sdrm[0]
    delta = sdrm[1]
    lambta = sdrm[2]
    phi,delta,lambta = map(lambda x : np.pi / 180 * x, [phi,delta,lambta])
    mag = sdrm[3]
    M0 = mag2moment(mag)

    grad_mt_NED = ENU2NED(grad_mt_ENU)
    grad_mxx = grad_mt_NED[0]
    grad_myy = grad_mt_NED[1]
    grad_mzz = grad_mt_NED[2]
    grad_mxy = grad_mt_NED[3]
    grad_mxz = grad_mt_NED[4]
    grad_myz = grad_mt_NED[5]


    grad_Mxx_strike =  -M0*(2*sin(delta)*cos(lambta)*cos(2*phi) + 2*sin(2*delta)*sin(lambta)*sin(phi)*cos(phi))
    grad_Myy_strike =  M0*(2*sin(delta)*cos(lambta)*cos(2*phi) + 2*sin(2*delta)*sin(lambta)*sin(phi)*cos(phi))
    grad_Mzz_strike =  0
    grad_Mxy_strike =  M0*(-2*sin(delta)*sin(2*phi)*cos(lambta) + 1.0*sin(2*delta)*sin(lambta)*cos(2*phi))
    grad_Mxz_strike =  -M0*(sin(lambta)*cos(2*delta)*cos(phi) - sin(phi)*cos(delta)*cos(lambta))
    grad_Myz_strike =  -M0*(sin(lambta)*sin(phi)*cos(2*delta) + cos(delta)*cos(lambta)*cos(phi))
    
    grad_Mxx_rake=  -M0*(-sin(delta)*sin(lambta)*sin(2*phi) + sin(2*delta)*sin(phi)**2*cos(lambta))
    grad_Myy_rake=  M0*(-sin(delta)*sin(lambta)*sin(2*phi) - sin(2*delta)*cos(lambta)*cos(phi)**2)
    grad_Mzz_rake=  M0*sin(2*delta)*cos(lambta)
    grad_Mxy_rake=  M0*(-sin(delta)*sin(lambta)*cos(2*phi) + 0.5*sin(2*delta)*sin(2*phi)*cos(lambta))
    grad_Mxz_rake=  -M0*(-sin(lambta)*cos(delta)*cos(phi) + sin(phi)*cos(2*delta)*cos(lambta))
    grad_Myz_rake=  -M0*(-sin(lambta)*sin(phi)*cos(delta) - cos(2*delta)*cos(lambta)*cos(phi))
    
    grad_Mxx_dip =  -M0*(2*sin(lambta)*sin(phi)**2*cos(2*delta) + sin(2*phi)*cos(delta)*cos(lambta))
    grad_Myy_dip =  M0*(-2*sin(lambta)*cos(2*delta)*cos(phi)**2 + sin(2*phi)*cos(delta)*cos(lambta))
    grad_Mzz_dip =  2*M0*sin(lambta)*cos(2*delta)
    grad_Mxy_dip =  M0*(1.0*sin(lambta)*sin(2*phi)*cos(2*delta) + cos(delta)*cos(lambta)*cos(2*phi))
    grad_Mxz_dip =  -M0*(-sin(delta)*cos(lambta)*cos(phi) - 2*sin(2*delta)*sin(lambta)*sin(phi))
    grad_Myz_dip =  -M0*(-sin(delta)*sin(phi)*cos(lambta) + 2*sin(2*delta)*sin(lambta)*cos(phi))
    
    grad_Mxx_M0 =  -sin(delta)*sin(2*phi)*cos(lambta) - sin(2*delta)*sin(lambta)*sin(phi)**2
    grad_Myy_M0 =  sin(delta)*sin(2*phi)*cos(lambta) - sin(2*delta)*sin(lambta)*cos(phi)**2
    grad_Mzz_M0 =  sin(2*delta)*sin(lambta)
    grad_Mxy_M0 =  sin(delta)*cos(lambta)*cos(2*phi) + 0.5*sin(2*delta)*sin(lambta)*sin(2*phi)
    grad_Mxz_M0 =  -sin(lambta)*sin(phi)*cos(2*delta) - cos(delta)*cos(lambta)*cos(phi)
    grad_Myz_M0 =  sin(lambta)*cos(2*delta)*cos(phi) - sin(phi)*cos(delta)*cos(lambta)

    grad_strike = grad_mxx * grad_Mxx_strike + grad_myy * grad_Myy_strike + grad_mzz * grad_Mzz_strike \
                    + grad_mxy * grad_Mxy_strike + grad_mxz * grad_Mxz_strike + grad_myz * grad_Myz_strike
    grad_dip = grad_mxx * grad_Mxx_dip + grad_myy * grad_Myy_dip + grad_mzz * grad_Mzz_dip \
                    + grad_mxy * grad_Mxy_dip + grad_mxz * grad_Mxz_dip + grad_myz * grad_Myz_dip
    grad_rake = grad_mxx * grad_Mxx_rake + grad_myy * grad_Myy_rake + grad_mzz * grad_Mzz_rake \
                    + grad_mxy * grad_Mxy_rake + grad_mxz * grad_Mxz_rake + grad_myz * grad_Myz_rake
    grad_M0 = grad_mxx * grad_Mxx_M0 + grad_myy * grad_Myy_M0 + grad_mzz * grad_Mzz_M0 \
                    + grad_mxy * grad_Mxy_M0 + grad_mxz * grad_Mxz_M0 + grad_myz * grad_Myz_M0

    grad_M0_mag =  1.5*10**(1.5*mag + 9.1)*np.log(10)
    grad_mag = grad_M0 * grad_M0_mag

    grad_sdrm = np.array([grad_strike,grad_dip,grad_rake,grad_mag])
    grad_sdrm[:3] = grad_sdrm[:3] / 180 *np.pi 

    return grad_sdrm


def grad_mt2sdr_lune(grad_mt_ENU, sdrm_lune):
    from numpy import sin, cos, sqrt
    from numpy import arccos as acos
    import numpy as np
    '''
    sdrm_lune: strike, dip, rake, mag, colat, long
    '''
    kappa = sdrm_lune[0]
    h = cos(np.deg2rad(sdrm_lune[1]))
    sigma = sdrm_lune[2]

    mag = sdrm_lune[3]
    rho = 10**(1.5*mag+9.1)*sqrt(2)

    beta = sdrm_lune[4]
    gamma = sdrm_lune[5]

    kappa, sigma, beta, gamma = map(lambda x : np.pi / 180 * x, [kappa, sigma, beta, gamma])

    grad_mt_USE = ENU2USE(grad_mt_ENU)
    grad_mrr = grad_mt_USE[0]
    grad_mtt = grad_mt_USE[1]
    grad_mpp = grad_mt_USE[2]
    grad_mrt = grad_mt_USE[3]
    grad_mrp = grad_mt_USE[4]
    grad_mtp = grad_mt_USE[5]

    grad_Mrr_kappa =  0
    grad_Mtt_kappa =  0.0294627825494395*rho*((-48.0*sqrt(1 - h**2)*cos(2.0*kappa)*cos(sigma) - 48.0*sin(kappa)*sin(sigma)*sin(2.0*acos(h))*cos(kappa))*cos(gamma) + (-41.5692193816531*h*sin(2.0*sigma)*cos(2.0*kappa) - 10.3923048454133*(1.0 - 3.0*cos(2.0*sigma))*sin(2.0*kappa) + 41.5692193816531*sin(kappa)*cos(kappa)*cos(sigma)**2*cos(2.0*acos(h)))*sin(gamma))*sin(beta)
    grad_Mpp_kappa =  0.117851130197758*rho*(-3.46410161513775*h**2*(3.0*cos(2.0*sigma) + 1.0)*sin(gamma)*sin(kappa)*cos(kappa) + 3.0*h*(8.0*sqrt(1 - h**2)*sin(kappa)*sin(sigma)*cos(gamma)*cos(kappa) + 3.46410161513775*sin(gamma)*sin(2.0*sigma)*cos(2.0*kappa)) + 12.0*sqrt(1 - h**2)*cos(gamma)*cos(2.0*kappa)*cos(sigma) + 6.92820323027551*(1 - h**2)*sin(gamma)*sin(kappa)*cos(kappa) + 2*(1.73205080756888 - 5.19615242270663*cos(2.0*sigma))*sin(gamma)*sin(kappa)*cos(kappa))*sin(beta)
    grad_Mrt_kappa =  -0.353553390593274*rho*(3.46410161513775*sqrt(1 - h**2)*(h*cos(kappa)*cos(sigma) + sin(kappa)*sin(sigma))*sin(gamma)*cos(sigma) + (-2.0*h*sin(kappa)*cos(sigma) + 2.0*sin(sigma)*cos(kappa)*cos(2.0*acos(h)))*cos(gamma))*sin(beta)
    grad_Mrp_kappa =  -0.353553390593274*rho*((-2.0*h*cos(gamma)*cos(sigma) + 1.73205080756888*sqrt(1 - h**2)*sin(gamma)*sin(2.0*sigma))*cos(kappa) - (1.73205080756888*sin(gamma)*sin(2.0*acos(h))*cos(sigma)**2 + 2.0*sin(sigma)*cos(gamma)*cos(2.0*acos(h)))*sin(kappa))*sin(beta)
    grad_Mtp_kappa =  -0.0883883476483184*rho*((-13.856406460551*h*sin(2.0*kappa)*sin(2.0*sigma) + 3.46410161513775*(-2.0*cos(sigma)**2*cos(2.0*acos(h)) - 3.0*cos(2.0*sigma) + 1.0)*cos(2.0*kappa))*sin(gamma) + (-16.0*sqrt(1 - h**2)*sin(2.0*kappa)*cos(sigma) + 8.0*sin(sigma)*sin(2.0*acos(h))*cos(2.0*kappa))*cos(gamma))*sin(beta)
    
    grad_Mrr_sigma =  0.058925565098879*rho*(-3.46410161513775*(6.0 - 6.0*h**2)*sin(gamma)*sin(2.0*sigma) + 12.0*sin(2.0*acos(h))*cos(gamma)*cos(sigma))*sin(beta)
    grad_Mtt_sigma =  0.0294627825494395*rho*((24.0*sqrt(1 - h**2)*sin(2.0*kappa)*sin(sigma) - 24.0*sin(kappa)**2*sin(2.0*acos(h))*cos(sigma))*cos(gamma) + (-41.5692193816531*h*sin(2.0*kappa)*cos(2.0*sigma) + 10.3923048454133*(3.0*cos(2.0*kappa) + 1.0)*sin(2.0*sigma) - 41.5692193816531*sin(kappa)**2*sin(sigma)*cos(sigma)*cos(2.0*acos(h)))*sin(gamma))*sin(beta)
    grad_Mpp_sigma =  0.117851130197758*rho*(-10.3923048454133*h**2*sin(gamma)*sin(2.0*sigma)*cos(kappa)**2 + 3.0*h*(-4.0*sqrt(1 - h**2)*cos(gamma)*cos(kappa)**2*cos(sigma) + 3.46410161513775*sin(gamma)*sin(2.0*kappa)*cos(2.0*sigma)) - 6.0*sqrt(1 - h**2)*sin(2.0*kappa)*sin(sigma)*cos(gamma) + 10.3923048454133*sin(gamma)*sin(kappa)**2*sin(2.0*sigma))*sin(beta)
    grad_Mrt_sigma =  -0.353553390593274*rho*(3.46410161513775*sqrt(1 - h**2)*(-h*sin(kappa)*sin(sigma) - cos(kappa)*cos(sigma))*sin(gamma)*cos(sigma) - 3.46410161513775*sqrt(1 - h**2)*(h*sin(kappa)*cos(sigma) - sin(sigma)*cos(kappa))*sin(gamma)*sin(sigma) + (-2.0*h*sin(sigma)*cos(kappa) + 2.0*sin(kappa)*cos(sigma)*cos(2.0*acos(h)))*cos(gamma))*sin(beta)
    grad_Mrp_sigma =  -0.353553390593274*rho*((2.0*h*sin(sigma)*cos(gamma) + 3.46410161513775*sqrt(1 - h**2)*sin(gamma)*cos(2.0*sigma))*sin(kappa) + (-3.46410161513775*sin(gamma)*sin(sigma)*sin(2.0*acos(h))*cos(sigma) + 2.0*cos(gamma)*cos(sigma)*cos(2.0*acos(h)))*cos(kappa))*sin(beta)
    grad_Mtp_sigma =  -0.0883883476483184*rho*((13.856406460551*h*cos(2.0*kappa)*cos(2.0*sigma) + 1.73205080756888*(4.0*sin(sigma)*cos(sigma)*cos(2.0*acos(h)) + 6.0*sin(2.0*sigma))*sin(2.0*kappa))*sin(gamma) + (-8.0*sqrt(1 - h**2)*sin(sigma)*cos(2.0*kappa) + 4.0*sin(2.0*kappa)*sin(2.0*acos(h))*cos(sigma))*cos(gamma))*sin(beta)
    
    grad_Mrr_h =  0.058925565098879*rho*((-20.7846096908265*h*cos(2.0*sigma) - 10.3923048454133*sin(2.0*acos(h))/sqrt(1 - h**2))*sin(gamma) - 24.0*sin(sigma)*cos(gamma)*cos(2.0*acos(h))/sqrt(1 - h**2))*sin(beta)
    grad_Mtt_h =  0.0294627825494395*rho*((-20.7846096908265*sin(2.0*kappa)*sin(2.0*sigma) + 41.5692193816531*sin(kappa)**2*sin(2.0*acos(h))*cos(sigma)**2/sqrt(1 - h**2))*sin(gamma) + (24.0*h*sin(2.0*kappa)*cos(sigma)/sqrt(1 - h**2) + 48.0*sin(kappa)**2*sin(sigma)*cos(2.0*acos(h))/sqrt(1 - h**2))*cos(gamma))*sin(beta)
    grad_Mpp_h =  0.117851130197758*rho*(12.0*h**2*sin(sigma)*cos(gamma)*cos(kappa)**2/sqrt(1 - h**2) + 3.46410161513775*h*(3.0*cos(2.0*sigma) + 1.0)*sin(gamma)*cos(kappa)**2 + 6.92820323027551*h*sin(gamma)*cos(kappa)**2 - 6.0*h*sin(2.0*kappa)*cos(gamma)*cos(sigma)/sqrt(1 - h**2) - 12.0*sqrt(1 - h**2)*sin(sigma)*cos(gamma)*cos(kappa)**2 + 5.19615242270663*sin(gamma)*sin(2.0*kappa)*sin(2.0*sigma))*sin(beta)
    grad_Mrt_h =  -0.353553390593274*rho*(-3.46410161513775*h*(h*sin(kappa)*cos(sigma) - sin(sigma)*cos(kappa))*sin(gamma)*cos(sigma)/sqrt(1 - h**2) + 3.46410161513775*sqrt(1 - h**2)*sin(gamma)*sin(kappa)*cos(sigma)**2 + (2.0*cos(kappa)*cos(sigma) + 4.0*sin(kappa)*sin(sigma)*sin(2.0*acos(h))/sqrt(1 - h**2))*cos(gamma))*sin(beta)
    grad_Mrp_h =  -0.353553390593274*rho*((-1.73205080756888*h*sin(gamma)*sin(2.0*sigma)/sqrt(1 - h**2) - 2.0*cos(gamma)*cos(sigma))*sin(kappa) + (-3.46410161513775*sin(gamma)*cos(sigma)**2*cos(2.0*acos(h))/sqrt(1 - h**2) + 4.0*sin(sigma)*sin(2.0*acos(h))*cos(gamma)/sqrt(1 - h**2))*cos(kappa))*sin(beta)
    grad_Mtp_h =  -0.0883883476483184*rho*((6.92820323027551*sin(2.0*sigma)*cos(2.0*kappa) - 6.92820323027551*sin(2.0*kappa)*sin(2.0*acos(h))*cos(sigma)**2/sqrt(1 - h**2))*sin(gamma) + (-8.0*h*cos(2.0*kappa)*cos(sigma)/sqrt(1 - h**2) - 8.0*sin(2.0*kappa)*sin(sigma)*cos(2.0*acos(h))/sqrt(1 - h**2))*cos(gamma))*sin(beta)
    
    grad_Mrr_rho =  0.058925565098879*(1.73205080756888*(6.0*(1 - h**2)*cos(2.0*sigma) - 3.0*cos(2.0*acos(h)) - 1.0)*sin(gamma) + 12.0*sin(sigma)*sin(2.0*acos(h))*cos(gamma))*sin(beta) + 0.577350269189626*cos(beta)
    grad_Mtt_rho =  0.0294627825494395*(-24.0*(sqrt(1 - h**2)*sin(2.0*kappa)*cos(sigma) + sin(kappa)**2*sin(sigma)*sin(2.0*acos(h)))*cos(gamma) + 1.73205080756888*(-12.0*h*sin(2.0*kappa)*sin(2.0*sigma) + (1.0 - 3.0*cos(2.0*sigma))*(3.0*cos(2.0*kappa) + 1.0) + 12.0*sin(kappa)**2*cos(sigma)**2*cos(2.0*acos(h)))*sin(gamma))*sin(beta) + 0.577350269189626*cos(beta)
    grad_Mpp_rho =  0.117851130197758*(1.73205080756888*h**2*(3.0*cos(2.0*sigma) + 1.0)*sin(gamma)*cos(kappa)**2 + 3.0*h*(-4.0*sqrt(1 - h**2)*sin(sigma)*cos(gamma)*cos(kappa)**2 + 1.73205080756888*sin(gamma)*sin(2.0*kappa)*sin(2.0*sigma)) + 6.0*sqrt(1 - h**2)*sin(2.0*kappa)*cos(gamma)*cos(sigma) - 3.46410161513775*(1 - h**2)*sin(gamma)*cos(kappa)**2 + (1.73205080756888 - 5.19615242270663*cos(2.0*sigma))*sin(gamma)*sin(kappa)**2)*sin(beta) + 0.577350269189626*cos(beta)
    grad_Mrt_rho =  (-1.22474487139159*sqrt(1 - h**2)*(h*sin(kappa)*cos(sigma) - sin(sigma)*cos(kappa))*sin(gamma)*cos(sigma) - 0.707106781186547*(h*cos(kappa)*cos(sigma) + sin(kappa)*sin(sigma)*cos(2.0*acos(h)))*cos(gamma))*sin(beta)
    grad_Mrp_rho =  (-0.353553390593274*(-2.0*h*cos(gamma)*cos(sigma) + 1.73205080756888*sqrt(1 - h**2)*sin(gamma)*sin(2.0*sigma))*sin(kappa) - 0.353553390593274*(1.73205080756888*sin(gamma)*sin(2.0*acos(h))*cos(sigma)**2 + 2.0*sin(sigma)*cos(gamma)*cos(2.0*acos(h)))*cos(kappa))*sin(beta)
    grad_Mtp_rho =  (-0.153093108923949*(4.0*h*sin(2.0*sigma)*cos(2.0*kappa) + (-2.0*cos(sigma)**2*cos(2.0*acos(h)) - 3.0*cos(2.0*sigma) + 1.0)*sin(2.0*kappa))*sin(gamma) - 0.353553390593274*(2.0*sqrt(1 - h**2)*cos(2.0*kappa)*cos(sigma) + sin(2.0*kappa)*sin(sigma)*sin(2.0*acos(h)))*cos(gamma))*sin(beta)
    
    grad_Mrr_beta =  0.058925565098879*rho*((1.73205080756888*(6.0*(1 - h**2)*cos(2.0*sigma) - 3.0*cos(2.0*acos(h)) - 1.0)*sin(gamma) + 12.0*sin(sigma)*sin(2.0*acos(h))*cos(gamma))*cos(beta) - 9.79795897113271*sin(beta))
    grad_Mtt_beta =  0.0294627825494395*rho*((-24.0*(sqrt(1 - h**2)*sin(2.0*kappa)*cos(sigma) + sin(kappa)**2*sin(sigma)*sin(2.0*acos(h)))*cos(gamma) + 1.73205080756888*(-12.0*h*sin(2.0*kappa)*sin(2.0*sigma) + (1.0 - 3.0*cos(2.0*sigma))*(3.0*cos(2.0*kappa) + 1.0) + 12.0*sin(kappa)**2*cos(sigma)**2*cos(2.0*acos(h)))*sin(gamma))*cos(beta) - 19.5959179422654*sin(beta))
    grad_Mpp_beta =  0.117851130197758*rho*((1.73205080756888*h**2*(3.0*cos(2.0*sigma) + 1.0)*sin(gamma)*cos(kappa)**2 + 3.0*h*(-4.0*sqrt(1 - h**2)*sin(sigma)*cos(gamma)*cos(kappa)**2 + 1.73205080756888*sin(gamma)*sin(2.0*kappa)*sin(2.0*sigma)) + 6.0*sqrt(1 - h**2)*sin(2.0*kappa)*cos(gamma)*cos(sigma) - 3.46410161513775*(1 - h**2)*sin(gamma)*cos(kappa)**2 + (1.73205080756888 - 5.19615242270663*cos(2.0*sigma))*sin(gamma)*sin(kappa)**2)*cos(beta) - 4.89897948556636*sin(beta))
    grad_Mrt_beta =  -0.353553390593274*rho*(3.46410161513775*sqrt(1 - h**2)*(h*sin(kappa)*cos(sigma) - sin(sigma)*cos(kappa))*sin(gamma)*cos(sigma) + 2.0*(h*cos(kappa)*cos(sigma) + sin(kappa)*sin(sigma)*cos(2.0*acos(h)))*cos(gamma))*cos(beta)
    grad_Mrp_beta =  -0.353553390593274*rho*((-2.0*h*cos(gamma)*cos(sigma) + 1.73205080756888*sqrt(1 - h**2)*sin(gamma)*sin(2.0*sigma))*sin(kappa) + (1.73205080756888*sin(gamma)*sin(2.0*acos(h))*cos(sigma)**2 + 2.0*sin(sigma)*cos(gamma)*cos(2.0*acos(h)))*cos(kappa))*cos(beta)
    grad_Mtp_beta =  -0.0883883476483184*rho*(1.73205080756888*(4.0*h*sin(2.0*sigma)*cos(2.0*kappa) + (-2.0*cos(sigma)**2*cos(2.0*acos(h)) - 3.0*cos(2.0*sigma) + 1.0)*sin(2.0*kappa))*sin(gamma) + 4.0*(2.0*sqrt(1 - h**2)*cos(2.0*kappa)*cos(sigma) + sin(2.0*kappa)*sin(sigma)*sin(2.0*acos(h)))*cos(gamma))*cos(beta)
    
    grad_Mrr_gamma =  0.058925565098879*rho*((10.3923048454133*(1 - h**2)*cos(2.0*sigma) - 5.19615242270663*cos(2.0*acos(h)) - 1.73205080756888)*cos(gamma) - 12.0*sin(gamma)*sin(sigma)*sin(2.0*acos(h)))*sin(beta)
    grad_Mtt_gamma =  0.0294627825494395*rho*(-(-24.0*sqrt(1 - h**2)*sin(2.0*kappa)*cos(sigma) - 24.0*sin(kappa)**2*sin(sigma)*sin(2.0*acos(h)))*sin(gamma) + (-20.7846096908265*h*sin(2.0*kappa)*sin(2.0*sigma) + 1.73205080756888*(1.0 - 3.0*cos(2.0*sigma))*(3.0*cos(2.0*kappa) + 1.0) + 20.7846096908265*sin(kappa)**2*cos(sigma)**2*cos(2.0*acos(h)))*cos(gamma))*sin(beta)
    grad_Mpp_gamma =  0.117851130197758*rho*(1.73205080756888*h**2*(3.0*cos(2.0*sigma) + 1.0)*cos(gamma)*cos(kappa)**2 + 3.0*h*(4.0*sqrt(1 - h**2)*sin(gamma)*sin(sigma)*cos(kappa)**2 + 1.73205080756888*sin(2.0*kappa)*sin(2.0*sigma)*cos(gamma)) - 6.0*sqrt(1 - h**2)*sin(gamma)*sin(2.0*kappa)*cos(sigma) - 3.46410161513775*(1 - h**2)*cos(gamma)*cos(kappa)**2 + (1.73205080756888 - 5.19615242270663*cos(2.0*sigma))*sin(kappa)**2*cos(gamma))*sin(beta)
    grad_Mrt_gamma =  -0.353553390593274*rho*(3.46410161513775*sqrt(1 - h**2)*(h*sin(kappa)*cos(sigma) - sin(sigma)*cos(kappa))*cos(gamma)*cos(sigma) - (2.0*h*cos(kappa)*cos(sigma) + 2.0*sin(kappa)*sin(sigma)*cos(2.0*acos(h)))*sin(gamma))*sin(beta)
    grad_Mrp_gamma =  -0.353553390593274*rho*((2.0*h*sin(gamma)*cos(sigma) + 1.73205080756888*sqrt(1 - h**2)*sin(2.0*sigma)*cos(gamma))*sin(kappa) + (-2.0*sin(gamma)*sin(sigma)*cos(2.0*acos(h)) + 1.73205080756888*sin(2.0*acos(h))*cos(gamma)*cos(sigma)**2)*cos(kappa))*sin(beta)
    grad_Mtp_gamma =  -0.0883883476483184*rho*((6.92820323027551*h*sin(2.0*sigma)*cos(2.0*kappa) + 1.73205080756888*(-2.0*cos(sigma)**2*cos(2.0*acos(h)) - 3.0*cos(2.0*sigma) + 1.0)*sin(2.0*kappa))*cos(gamma) - (8.0*sqrt(1 - h**2)*cos(2.0*kappa)*cos(sigma) + 4.0*sin(2.0*kappa)*sin(sigma)*sin(2.0*acos(h)))*sin(gamma))*sin(beta)
    
    grad_rho_mag =  2.12132034355964*10**(1.5*mag + 9.1)*np.log(10)
    grad_h_dip =  -sin(np.deg2rad(sdrm_lune[1]))


    grad_mt_kappa = grad_mrr * grad_Mrr_kappa + grad_mtt * grad_Mtt_kappa + grad_mpp * grad_Mpp_kappa \
                    + grad_mrt * grad_Mrt_kappa + grad_mrp * grad_Mrp_kappa + grad_mtp * grad_Mtp_kappa
    
    grad_mt_h = grad_mrr * grad_Mrr_h + grad_mtt * grad_Mtt_h + grad_mpp * grad_Mpp_h \
                    + grad_mrt * grad_Mrt_h + grad_mrp * grad_Mrp_h + grad_mtp * grad_Mtp_h
    
    grad_mt_sigma = grad_mrr * grad_Mrr_sigma + grad_mtt * grad_Mtt_sigma + grad_mpp * grad_Mpp_sigma \
                    + grad_mrt * grad_Mrt_sigma + grad_mrp * grad_Mrp_sigma + grad_mtp * grad_Mtp_sigma
    
    grad_mt_rho = grad_mrr * grad_Mrr_rho + grad_mtt * grad_Mtt_rho + grad_mpp * grad_Mpp_rho \
                    + grad_mrt * grad_Mrt_rho + grad_mrp * grad_Mrp_rho + grad_mtp * grad_Mtp_rho
    
    grad_mt_beta = grad_mrr * grad_Mrr_beta + grad_mtt * grad_Mtt_beta + grad_mpp * grad_Mpp_beta \
                    + grad_mrt * grad_Mrt_beta + grad_mrp * grad_Mrp_beta + grad_mtp * grad_Mtp_beta
    
    grad_mt_gamma = grad_mrr * grad_Mrr_gamma + grad_mtt * grad_Mtt_gamma + grad_mpp * grad_Mpp_gamma \
                    + grad_mrt * grad_Mrt_gamma + grad_mrp * grad_Mrp_gamma + grad_mtp * grad_Mtp_gamma

    grad_strike = grad_mt_kappa
    grad_dip = grad_mt_h * grad_h_dip
    grad_rake = grad_mt_sigma
    grad_mag = grad_mt_rho * grad_rho_mag
    grad_beta = grad_mt_beta
    grad_gamma = grad_mt_gamma

    grad_sdrm_lune = np.array([grad_strike,grad_dip,grad_rake,grad_mag, grad_beta, grad_gamma])

    for i in range(len(sdrm_lune)):
        if i != 3:
            grad_sdrm_lune[i] = grad_sdrm_lune[i] / 180 *np.pi 

    return grad_sdrm_lune