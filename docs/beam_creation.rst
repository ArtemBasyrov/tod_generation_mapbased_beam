Creating an Artificial Beam File
=================================

This page shows how to generate a synthetic beam FITS file suitable for use
as a pipeline input.  The example produces a 2-D elliptical Gaussian beam on a
CAR (Plate Carrée) grid using `pixell / enmap
<https://pixell.readthedocs.io/en/latest/>`_, which is the same library the
pipeline uses to read beam files.

.. note::

   The pipeline applies its own normalisation to the beam based on the
   ``power_fraction_threshold_*`` values: selected pixels are re-weighted so
   that their sum equals one.  You therefore do **not** need to normalise the
   beam before saving — the signal amplitude in the output TOD is determined
   entirely by the sky-map values and is not affected by the absolute scale of
   the beam file.

Dependencies
------------

.. code-block:: bash

   pip install numpy pixell

The ``pixell.utils`` module provides the ``arcmin`` conversion constant used
below.

Example: Elliptical Gaussian Beam
----------------------------------

.. code-block:: python

   import numpy as np
   from pixell import enmap, utils


   # ---------------------------------------------------------------------------
   # Helper 1 — build a square CAR grid and return RA / Dec offset arrays
   # ---------------------------------------------------------------------------
   def create_coord_with_resol(res, shape=(201, 201)):
       """Return (ra, dec, imap) for a uniformly spaced CAR grid.

       Parameters
       ----------
       res : float
           Pixel size in arcmin.  The same resolution is used along both axes.
       shape : tuple of int
           ``(n_rows, n_cols)`` of the 2-D grid.  Using an odd number of pixels
           in each dimension (e.g. 201) guarantees that the central pixel sits
           exactly at (0, 0) in sky coordinates.

       Returns
       -------
       ra : ndarray, shape ``shape``
           Right-ascension offset from the grid centre [arcmin].
           Positive values point west (enmap convention).
       dec : ndarray, shape ``shape``
           Declination offset from the grid centre [arcmin].
       imap : enmap.ndmap, shape ``(3, n_rows, n_cols)``
           Zero-filled enmap with the correct WCS header for I, Q, U.
           Write the beam values into ``imap[0]``, ``imap[1]``, ``imap[2]``
           before saving.
       """
       # enmap.geometry returns the shape and WCS for a CAR projection centred
       # at (ra, dec) = (0, 0) with the requested pixel size.
       shape, wcs = enmap.geometry(
           pos=(0, 0), shape=shape, proj='car', res=res * utils.arcmin
       )
       # Allocate a 3-component map (I, Q, U) filled with zeros.
       imap = enmap.zeros((3,) + shape, wcs=wcs)

       # posmap() returns two arrays (shape ``shape``) with the RA and Dec
       # of every pixel in radians, according to the WCS header.
       ra, dec = imap.posmap()

       # Identify the central pixel — integer division gives the middle index
       # for any odd-sized grid.
       center_ra  = shape[1] // 2
       center_dec = shape[0] // 2

       # Sky coordinates of the central pixel (radians).
       ra_c  = ra [center_dec, center_ra]
       dec_c = dec[center_dec, center_ra]

       # Subtract the centre and convert to arcmin so that the returned arrays
       # represent angular offsets rather than absolute sky positions.
       ra  = np.array(ra  - ra_c ).copy() / utils.arcmin
       dec = np.array(dec - dec_c).copy() / utils.arcmin

       return ra, dec, imap


   # ---------------------------------------------------------------------------
   # Helper 2 — evaluate a 2-D elliptical Gaussian
   # ---------------------------------------------------------------------------
   def anisotropic_gaussian_2d(X, Y, fwhm_x=1.0, fwhm_y=1.0):
       """Evaluate a 2-D Gaussian with independent widths along X and Y.

       The Gaussian is centred at the origin ``(X=0, Y=0)``, which corresponds
       to the beam axis.  Both widths are specified as Full-Width at Half
       Maximum (FWHM) in arcmin.

       Parameters
       ----------
       X : ndarray
           RA offsets [arcmin] — typically the ``ra`` array from
           ``create_coord_with_resol``.
       Y : ndarray
           Dec offsets [arcmin] — typically the ``dec`` array.
       fwhm_x : float
           FWHM along the RA axis [arcmin].
       fwhm_y : float
           FWHM along the Dec axis [arcmin].

       Returns
       -------
       gaussian : ndarray, same shape as ``X``
           Beam amplitude at each pixel.  Peak value is 1 at the origin.
           No normalisation is applied; the pipeline handles that internally.
       """
       # Convert FWHM → sigma for each axis.
       # The standard relation is: FWHM = 2 * sqrt(2 * ln(2)) * sigma
       sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
       sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))

       # Standard bivariate Gaussian exponent (axes aligned, no rotation).
       exponent = -(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2))

       return np.exp(exponent)


   # ---------------------------------------------------------------------------
   # Main: build the beam and write it to disk
   # ---------------------------------------------------------------------------

   # Create a 201 × 201 grid with 1 arcmin pixels.
   # The grid spans roughly ±100 arcmin (~1.67°) in each direction, which is
   # sufficient for beams up to ~60 arcmin FWHM with negligible aliasing.
   ra, dec, imap = create_coord_with_resol(res=1.0, shape=(201, 201))

   # Evaluate an elliptical Gaussian beam.
   # Here both axes are set to 30 arcmin FWHM (circular beam), but you can
   # use different values for fwhm_x and fwhm_y to model an elliptical beam.
   fwhm_arcmin = 30.0
   beam = anisotropic_gaussian_2d(ra, dec, fwhm_x=fwhm_arcmin, fwhm_y=fwhm_arcmin)

   # Assign the same beam pattern to all three Stokes components.
   # For an experiment with distinct polarised beam shapes, use separate
   # arrays for imap[1] (Q) and imap[2] (U).
   imap[0] = beam   # Stokes I beam
   imap[1] = beam   # Stokes Q beam
   imap[2] = beam   # Stokes U beam

   # Save to a FITS file.  The WCS header written by enmap ensures that the
   # pipeline can recover the pixel coordinates and correctly identify the
   # beam centre.
   imap.write('example_beam.fits')


Notes on beam centring
----------------------

The pipeline locates the beam axis by reading RA / Dec offsets from the WCS
header and finding the pixel closest to ``(RA, Dec) = (0, 0)``.  As long as
``create_coord_with_resol`` (or any equivalent procedure) sets the grid centre
to zero offset, the centring will be correct.  For a grid of shape
``(H, W)`` this corresponds to pixel index ``(H // 2, W // 2)``.

Notes on normalisation
-----------------------

The pipeline selects beam pixels that together carry a fraction
``power_fraction_threshold`` of the total beam power, then re-normalises those
weights to sum to one.  This means:

* You do **not** need to normalise the beam before saving.
* Changing the peak amplitude of the beam file has no effect on the TOD.
* Raising ``power_fraction_threshold`` toward ``1.0`` retains more pixels
  (slower but more accurate); lowering it toward ``0.9`` aggressively prunes
  faint sidelobes.

See :doc:`data_formats` for the full specification of the beam file format.
