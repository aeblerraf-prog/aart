from aart_func import *
from aart_func.accretion_models import get_accretion_model
from params import *

ACCRETION_MODEL = get_accretion_model()
G_DISK = ACCRETION_MODEL.g_disk
G_GAS = ACCRETION_MODEL.g_gas

#calculate the observed brightness for a purely radial profile
def bright_radial(grid,mask,redshift_sign,a,rs,isco,thetao):
    """
    Calculate the brightness of a rotationally symmetric disk
    (Eq. 50 P1)
    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
    :param mask: mask out the lensing band, see lb_f.py for detail
    :param redshift_sign: sign of the redshift
    :param a: black hole spin
    :param rs: source radius
    :param isco: radius of the inner-most stable circular orbit
    :param thetao: observer inclination

    :return: image of a lensed equitorial source with only radial dependence. 
    """
    alpha = grid[:,0][mask]
    beta = grid[:,1][mask]

    rs = rs[mask]

    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)

    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]

    brightness[rs >= isco] = (
        G_DISK(
            rs[rs >= isco],
            a,
            redshift_sign[rs >= isco],
            lamb[rs >= isco],
            eta[rs >= isco],
        )
        ** gfactor
        * ilp.profile(rs[rs >= isco], a, gammap, mup, sigmap)
    )
    brightness[rs < isco] = (
        G_GAS(
            rs[rs < isco],
            a,
            redshift_sign[rs < isco],
            lamb[rs < isco],
            eta[rs < isco],
        )
        ** gfactor
        * ilp.profile(rs[rs < isco], a, gammap, mup, sigmap)
    )
    
    r_p = 1+np.sqrt(1-a**2)
    brightness[rs<=r_p] = 0
    
    I = np.zeros(mask.shape)
    I[mask] = brightness
    
    return(I)


# TODO: This expression will just work for sure for the Keplerian velocity.
# I need to check if it has to be modified for the general four-velocity
def CosAng(r,a,b,lamb,eta):
    """
    Calculates the cosine of the emission angle
    :param r: radius of the source
    :param a: spin of the black hole
    :param b: sign for the redshift
    :param lamb: angular momentum
    :param eta: Carter constant

    :return: the  cosine of the emission angle
    """
    # From eta, solve for Sqrt(p_\theta/p_t)
    kthkt=np.sqrt(eta)
    # Sqrt(g^{\theta\theta}) Evaluated at the equatorial plane
    thth=1/r
    return thth*G_DISK(r,a,b,lamb,eta)*kthkt


#calculate the observed brightness for an arbitrary profile, passed in as the interpolation object
#but ignoring the time delay due to lensing
def fast_light(grid,mask,redshift_sign,a,isco,rs,th,interpolation,thetao):
    """
    Calculate the black hole image ignoring the time delay due to lensing or geometric effect
    (Eq. 116 P1)
    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
    :param mask: mask out the lensing band, see lb_f.py for detail
    :param redshift_sign: sign of the redshift
    :param a: black hole spin
    :param isco: radius of the inner-most stable circular orbit
    :param rs: source radius
    :param th: source angle, polar coordinate
    :param interpolation: 2 dimensional brightness function of the source, interpolation object
    :param thetao: observer inclination

    :return: image of a lensed equitorial source with only radial dependence. 
    """
    alpha = grid[:,0][mask]
    beta = grid[:,1][mask]
    rs = rs[mask]
    th = th[mask]
    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]

    x_aux=rs*np.cos(th)
    y_aux=rs*np.sin(th)
 
    brightness[rs >= isco] = (
        G_DISK(
            rs[rs >= isco],
            a,
            redshift_sign[rs >= isco],
            lamb[rs >= isco],
            eta[rs >= isco],
        )
        ** gfactor
        * interpolation(np.vstack([x_aux[rs >= isco], y_aux[rs >= isco]]).T)
    )
    brightness[rs < isco] = (
        G_GAS(
            rs[rs < isco],
            a,
            redshift_sign[rs < isco],
            lamb[rs < isco],
            eta[rs < isco],
        )
        ** gfactor
        * interpolation(np.vstack([x_aux[rs < isco], y_aux[rs < isco]]).T)
    )
    
    r_p = 1+np.sqrt(1-a**2)
    brightness[rs<=r_p] = 0
    
    I = np.zeros(mask.shape)
    I[mask] = brightness
    return(I)

#calculate the observed brightness for an arbitrary, evolving profile, passed in as the interpolation object
def slow_light(grid,mask,redshift_sign,a,isco,rs,th,ts,interpolation,thetao):
    """
    Calculate the black hole image including the time delay due to lensing and geometric effect
    (Eq. 50 P1)

    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
    :param mask: mask out the lensing band, see lb_f.py for detail
    :param redshift_sign: sign of the redshift
    :param a: black hole spin
    :param isco: radius of the inner-most stable circular orbit
    :param rs: source radius
    :param th: source angle, polar coordinate
    :param ts: time of emission at the source
    :param interpolation: a time series of 2 dimensional brightness function of the source, 3d interpolation object
    :param thetao: observer inclination

    :return: image of a lensed equitorial source with only radial dependence. 
    """
    alpha = grid[:,0][mask]
    beta = grid[:,1][mask]
    rs = rs[mask]
    th = th[mask]
    ts = ts[mask]
    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]
    
    x_aux=rs*np.cos(th)
    y_aux=rs*np.sin(th)

    brightness[rs >= isco] = (
        G_DISK(
            rs[rs >= isco],
            a,
            redshift_sign[rs >= isco],
            lamb[rs >= isco],
            eta[rs >= isco],
        )
        ** gfactor
        * interpolation(np.vstack([ts[rs >= isco], x_aux[rs >= isco], y_aux[rs >= isco]]).T)
    )
    brightness[rs < isco] = (
        G_GAS(
            rs[rs < isco],
            a,
            redshift_sign[rs < isco],
            lamb[rs < isco],
            eta[rs < isco],
        )
        ** gfactor
        * interpolation(np.vstack([ts[rs < isco], x_aux[rs < isco], y_aux[rs < isco]]).T)
    )

    r_p = 1+np.sqrt(1-a**2)
    brightness[rs<=r_p] = 0
    
    I = np.zeros(mask.shape)
    I[mask] = brightness
    return(I)

def br(supergrid0,mask0,N0,rs0,sign0,supergrid1,mask1,N1,rs1,sign1,supergrid2,mask2,N2,rs2,sign2):
    """
    Calculate and save the radial brightness profile
    """
    bghts0 = bright_radial(supergrid0,mask0,sign0,spin_case,rs0,isco,thetao)
    bghts1 = bright_radial(supergrid1,mask1,sign1,spin_case,rs1,isco,thetao)
    bghts2 = bright_radial(supergrid2,mask2,sign2,spin_case,rs2,isco,thetao)

    I0 = bghts0.reshape(N0,N0).T
    I1 = bghts1.reshape(N1,N1).T
    I2 = bghts2.reshape(N2,N2).T

    filename=path+"Intensity_a_%s_i_%s.h5"%(spin_case,i_case)
    h5f = h5py.File(filename, 'w')

    h5f.create_dataset('bghts0', data=I0)
    h5f.create_dataset('bghts1', data=I1)
    h5f.create_dataset('bghts2', data=I2)

    h5f.close()

    print("File ",filename," created.")

def br_bv(supergrid0,mask0,N0,rs0,sign0):
    """
    Calculate and save the radial brightness profile
    """
    bghts0 = bright_radial(supergrid0,mask0,sign0,spin_case,rs0,isco,thetao)

    I0 = bghts0.reshape(N0,N0).T

    filename=path+"Intensity_bv_a_%s_i_%s.h5"%(spin_case,i_case)
    h5f = h5py.File(filename, 'w')

    h5f.create_dataset('bghts0', data=I0)

    h5f.close()

    print("File ",filename," created.")

def gfactorf(grid,mask,redshift_sign,a,isco,rs,thetao):
    """
    Calculate the redshift factor
    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
    :param mask: mask out the lensing band, see lb_f.py for detail
    :param redshift_sign: sign of the redshift
    :param a: black hole spin
    :param isco: radius of the inner-most stable circular orbit
    :param rs: source radius
    :param thetao: observer inclination

    :return: redshift factor at each point.

    """
    
    alpha = grid[:,0][mask]
    beta = grid[:,1][mask]
    rs = rs[mask]
    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
    gfact = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]
    
    gfact[rs >= isco] = G_DISK(
        rs[rs >= isco],
        a,
        redshift_sign[rs >= isco],
        lamb[rs >= isco],
        eta[rs >= isco],
    )
    gfact[rs < isco] = G_GAS(
        rs[rs < isco],
        a,
        redshift_sign[rs < isco],
        lamb[rs < isco],
        eta[rs < isco],
    )
    
    r_p = 1+np.sqrt(1-a**2)
    gfact[rs<=r_p] = 0
    
    gs = np.zeros(mask.shape)
    gs[mask] = gfact
    return(gs)

# orbit for the centroid with radhs=Radius of the hotspot and velhs = 0.01 (angular frequency)
# one may put an arbitrary orbit
def x0(t):
    return(radhs*np.cos(t*velhs))

def y0(t):
    return(radhs*np.sin(t*velhs))

def flare_model(grid,mask,redshift_sign,a,rs,th,ts,thetao,rwidth,delta_t):

    """
    Calculate the black hole image including the time delay due to lensing and geometric effect
    :param grid: alpha and beta grid on the observer plane on which we evaluate the observables
    :param mask: mask out the lensing band, see lb_f.py for detail
    :param redshift_sign: sign of the redshift
    :param mbar: lensing band index 0,1,2,...
    :param a: black hole spin
    :param isco: radius of the inner-most stable circular orbit
    :param rs: source radius
    :param th: source angle, polar coordinate
    :param ts: time of emission at the source
    :param interpolation: a time series of 2 dimensional brightness function of the source, 3d interpolation object
    :param thetao: observer inclination
    
    :return: image of a lensed equitorial source with only radial dependence. 
    """

    alpha = grid[:,0][mask]
    beta = grid[:,1][mask]
    rs = rs[mask]
    th = th[mask]
    ts = ts[mask]
    lamb,eta = rt.conserved_quantities(alpha,beta,thetao,a)
    brightness = np.zeros(rs.shape[0])
    redshift_sign = redshift_sign[mask]
    
    x_aux = rs*np.cos(th)
    y_aux = rs*np.sin(th)
    # x0 and y0 is now a function of t, where one can specify an arbitrary equitorial orbit
    brightness = np.exp(-(x_aux-x0(ts+delta_t))**2/rwidth**2-(y_aux-y0(ts+delta_t))**2/rwidth**2)
    
    brightness[rs >= isco] *= (
        G_DISK(
            rs[rs >= isco],
            a,
            redshift_sign[rs >= isco],
            lamb[rs >= isco],
            eta[rs >= isco],
        )
        ** gfactor
    )
    brightness[rs < isco] *= (
        G_GAS(
            rs[rs < isco],
            a,
            redshift_sign[rs < isco],
            lamb[rs < isco],
            eta[rs < isco],
        )
        ** gfactor
    )

    r_p = 1+np.sqrt(1-a**2)
    brightness[rs<=r_p] = 0
    
    I = np.zeros(mask.shape)
    I[mask] = brightness
    return(np.nan_to_num(I))
