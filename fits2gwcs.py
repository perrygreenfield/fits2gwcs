from __future__ import print_function, division

'''
This module contains prototype code to convert FITS WCS representations into
GWCS equivalent representations.
'''

import numpy as np
from astropy import wcs
from astropy import units as u
from astropy import coordinates as coord
import astropy.modeling.rotations as rotations
import astropy.modeling.projections as projections
from astropy.modeling.mappings import Identity, Mapping
from astropy.modeling.functional_models import Shift, Const1D
import gwcs
from gwcs import wcs as ggwcs
from gwcs import coordinate_frames as cf

def convert_wcs(fitswcs):
    '''
    Accepts an astropy.wcs wcs object (based on the FITS standard).

    Returns the GWCS equivalent WCS object.
    '''
    # this first version only handles non distorted ra,dec tangent projections
    # check that it is that case
    radesys_dict = {
        'ICRS': coord.ICRS,
        'FK5': coord.FK5,
        'FK4': coord.FK4,
        }
    fctypes = fitswcs.wcs.ctype
    fcrval = fitswcs.wcs.crval
    fcrpix = fitswcs.wcs.crpix
    if fitswcs.naxis != 2:
        raise ValueError("currently only handles 2d images")
    for ctype in fctypes:
        if ctype not in ['RA---TAN', 'DEC--TAN', 'HPLN-TAN', 'HPLT-TAN']:
            raise ValueError("currently only supports RA,DEC tangent projections (no SIP)")
    if fitswcs.cpdis1 or fitswcs.cpdis1:
        raise ValueError("currently doesn't support distortion")    
    # construct transformation
    if fitswcs.wcs.has_cd():
        trans = ((Shift(-fcrpix[0]) & Shift(-fcrpix[1])) | 
                 projections.AffineTransformation2D(fitswcs.wcs.cd) | 
                 projections.Pix2Sky_TAN() | 
                 rotations.RotateNative2Celestial(fcrval[0], fcrval[1], 180.))
    elif fitswcs.wcs.has_pc():
        trans = ((Shift(-fcrpix[0]) & Shift(-fcrpix[1])) | 
                 (projections.AffineTransformation2D(fitswcs.wcs.pc) * 
                 (Const1D(fitswcs.wcs.cdelt[0]) & Const1D(fitswcs.wcs.cdelt[1])))|
                 projections.Pix2Sky_TAN() | 
                 rotations.RotateNative2Celestial(fcrval[0], fcrval[1], 180.))
    else:
        cdelt = fitswcs.wcs.cdelt
        crota2 = fitswcs.wcs.crota[1]*np.pi/180 # unware of any crota1 case
        pscale_ratio = cdelt[1]/cdelt[0] 
        pcmatrix = np.array([[np.cos(crota2), -pscale_ratio*np.sin(crota2)],
                             [np.sin(crota2)/pscale_ratio, np.cos(crota2)]])
        trans = ((Shift(-fcrpix[0]) & Shift(-fcrpix[1])) | 
                 (projections.AffineTransformation2D(pcmatrix) * 
                 (Const1D(fitswcs.wcs.cdelt[0]) & Const1D(fitswcs.wcs.cdelt[1])))|
                 projections.Pix2Sky_TAN() | 
                 rotations.RotateNative2Celestial(fcrval[0], fcrval[1], 180.))
    detector_frame = cf.Frame2D(name="detector", axes_names=('x', 'y'),
                                unit=(u.pix, u.pix))
    # Now see if a standard frame is referenced.
    if fitswcs.wcs.radesys:
        if fitswcs.wcs.radesys in radesys_dict:
            reference_frame = radesys_dict[fitswcs.wcs.radesys]()
            sky_frame = cf.CelestialFrame(reference_frame=reference_frame,
                                          name=fitswcs.wcs.radesys.lower())
    else:
        sky_frame = '' # or None?
   
    wcsobj = ggwcs.WCS(forward_transform=trans, input_frame=detector_frame,
                     output_frame=sky_frame)
    return wcsobj