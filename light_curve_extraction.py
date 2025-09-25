import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
from photutils import aperture
import astropy as ap 
from PIL import Image
import glob
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.table import Table
from photutils.detection import DAOStarFinder, IRAFStarFinder


# i am going to calibrate my picture here since we do not have a proper class structure yet....

sci_data = fits.getdata(r"images\WASP-12b_example_uncalibrated_images\uncalibrated\WASP-12b_00040.fits").astype(float)
bias_data = fits.getdata(r"images\WASP-12b_example_raw_biases\bias_00100.fits").astype(float)
dark_data = fits.getdata(r"images\WASP-12b_example_raw_darks\dark_00150.fits").astype(float)
flat_data = fits.getdata(r"images\WASP-12b_example_raw_flats\flat_r_00002.fits").astype(float)

dark_corrected = sci_data - bias_data - dark_data
flat_norm = flat_data / np.median(flat_data)
calibrated = dark_corrected / flat_norm

mean, median, std = sigma_clipped_stats(calibrated, sigma=3.0)

#---------------------------

FWHM = 3.0


daofind = DAOStarFinder(fwhm=FWHM, threshold=5.*std)
IRAFfind = IRAFStarFinder(fwhm=FWHM, threshold=5.*std)
sources = daofind(calibrated - median)
sources_2 = IRAFfind(calibrated - median)

cords  = list(zip(sources["xcentroid"], sources["ycentroid"]))
cords = [(3499.825554241658, 39.37473823979557)] # override for now for inspection
ap_radius  = 1.5 * FWHM
ann_inner = 3 * FWHM
ann_width = 2 * FWHM 

aper = CircularAperture(cords, r=ap_radius)
ann = CircularAnnulus(cords, r_in=ann_inner, r_out=ann_inner+ann_width)

apertures = [aper, ann]
photom_table = aperture_photometry(calibrated, apertures) # photom_table will contain aperture_sum and annulus_sum for each entry

print(photom_table)



plt.figure(figsize=(7,7))
plt.imshow(calibrated, cmap='grey', origin='lower', vmin=median-2*std, vmax=median+5*std)
aper.plot(color='green')
ann.plot(color='cyan')
plt.scatter(sources['xcentroid'], sources['ycentroid'], s=40, facecolors='none', edgecolors='r', label = "DOA")
plt.scatter(sources_2['xcentroid'], sources_2['ycentroid'], s=40, facecolors='none', edgecolors='b', label = "IRAF")
plt.legend(loc = "upper left")
plt.title("Calibrated Image with photometry")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.show()
