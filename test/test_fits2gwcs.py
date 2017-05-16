# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, unicode_literals, print_function

import numpy as np
import astropy.io.fits as fits
import astropy.wcs as awcs
import sys
import os
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import fits2gwcs as fg 

datapath = os.path.dirname(__file__)

test_files = ['acs1.fits', 'solar.fits', 'wise.fits']

def grid(shape, nsamples):
    '''
    generate a 2d grid of x, y
    '''
    x1d = np.arange(nsamples) * shape[0] / nsamples
    y1d = np.arange(nsamples) * shape[1] / nsamples
    x = x1d * np.ones((nsamples,))[:, np.newaxis]
    y = y1d[:, np.newaxis] * np.ones((nsamples,))
    return x, y


@pytest.mark.parametrize("filename", test_files)
def test_compare(filename): 
    '''Simple case '''
    hdul = fits.open(os.path.join(datapath, filename))
    oldwcs = awcs.WCS(hdul[0])
    newwcs = fg.convert_wcs(oldwcs)
    x, y = grid((7000, 7000), 50)
    oldx, oldy = oldwcs.all_pix2world(x, y, 1)
    newx, newy = newwcs(x, y)
    assert np.allclose(oldx, newx, atol=1.e-13, rtol=1.e-13)
    assert np.allclose(oldx, newx, atol=1.e-13, rtol=1.e-13)
