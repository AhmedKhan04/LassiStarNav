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
import astroalign as aa
import os 
#import cv2



def get_max(img): 
    maxVal = np.amax(img)
    #print('--------------')
    #print(maxVal)
    maxLoc = np.unravel_index(np.argmax(img), img.shape)
    maxLoc = [(maxLoc[1], maxLoc[0])]
    #print(maxLoc)
    #print('--------------')
    return maxLoc

# i am going to calibrate my picture here since we do not have a proper class structure yet....

sci_data = fits.getdata(r"images\WASP-12b_example_uncalibrated_images\uncalibrated\WASP-12b_00040.fits").astype(float)
bias_data = fits.getdata(r"images\WASP-12b_example_raw_biases\bias_00100.fits").astype(float)
dark_data = fits.getdata(r"images\WASP-12b_example_raw_darks\dark_00150.fits").astype(float)
flat_data = fits.getdata(r"images\WASP-12b_example_raw_flats\flat_r_00002.fits").astype(float)

dark_corrected = sci_data - bias_data - dark_data
flat_norm = flat_data / np.median(flat_data)
calibrated = dark_corrected / flat_norm

folder_path = r"images\WASP-12b_example_uncalibrated_images\uncalibrated"
initial = True 

for filename in os.listdir(folder_path):
    #print(filename)
    file_path = os.path.join(folder_path, filename)
    if(initial):
        initial = False
        continue 
    print(file_path)
    # pulling in second image...eventually loop this. 
    sci_data_second = fits.getdata(fr"{file_path}").astype(float)
    dark_corrected_second = sci_data_second - bias_data - dark_data
    calibrated_second = dark_corrected_second / flat_norm

    #from scipy.ndimage import rotate, zoom
    #calibrated_second = rotate(calibrated_second, angle=30.0, reshape=False)
    #calibrated_second = zoom(calibrated_second, 1.5, order=2)

    mean, median, std = sigma_clipped_stats(calibrated, sigma=3.0)

    mean_s, median_s, std_s = sigma_clipped_stats(calibrated_second, sigma=3.0)

    #img_aligned, footprint = aa.register(calibrated_second, calibrated, detection_sigma=3.0)

    r"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(calibrated_second, cmap='grey', interpolation='none', origin='lower', vmin=median_s-2*std_s, vmax=median_s+5*std_s)
    axes[0, 0].axis('off')
    axes[0, 0].set_title("Specific Image")


    axes[0, 1].imshow(calibrated, cmap='grey', interpolation='none', origin='lower', vmin=median-2*std, vmax=median+5*std)
    axes[0, 1].axis('off')
    axes[0, 1].set_title("Target Image")

    axes[1, 0].imshow(img_aligned, cmap='grey', interpolation='none', origin='lower', vmin=median_s-2*std_s, vmax=median_s+5*std_s)
    axes[1, 0].axis('off')
    axes[1, 0].set_title("Specific Image aligned with Target")

    axes[1, 1].imshow(footprint, cmap='grey', interpolation='none', origin='lower')
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Footprint of the transformation")

    axes[1, 0].axis('off')

    plt.tight_layout()
    plt.show()
    plt.figure()
    """

    #---------------------------

    FWHM = 3.0


    #daofind = DAOStarFinder(fwhm=FWHM, threshold=5.*std)
    #IRAFfind = IRAFStarFinder(fwhm=FWHM, threshold=5.*std)
    #sources = daofind(calibrated - median)
    #sources_2 = IRAFfind(calibrated - median)

    #cords  = list(zip(sources["xcentroid"], sources["ycentroid"]))

    cords = np.array([(3499.825554241658, 39.37473823979557)])  # coordinates from image A
    transform, (src_list, ref_list) = aa.find_transform(calibrated_second, calibrated)

    #print(transform)  # see what transform was found

    cords_transformed = transform(cords)

    #print("Original coords:", cords)
    #print("Transformed coords:", cords_transformed)


    # override for now for inspection
    ap_radius  = 1.5 * FWHM
    ann_inner = 3 * FWHM
    ann_width = 2 * FWHM 

    #aper = CircularAperture(cords, r=ap_radius)
    #ann = CircularAnnulus(cords, r_in=ann_inner, r_out=ann_inner+ann_width)

    ny, nx = calibrated_second.shape
    t_p = cords_transformed
    x, y = t_p[0]
    x, y = int(x), int(y)
    half_box = 50
    #print(x)
    #print(y)
    x1, x2 = max(0, x - half_box), min(nx, x + half_box)
    y1, y2 = max(0, y - half_box), min(ny, y + half_box)

    mask = np.zeros_like(calibrated_second)
    mask[y1:y2, x1:x2] = 1

    masked_data = calibrated_second *  mask

    #plt.imshow(masked_data, cmap='gray', origin='lower',
            #vmin=median_s - 2*std_s, vmax=median_s + 5*std_s)
    #plt.show()
    past_cord = cords_transformed
    
    #cords_transformed = get_max(masked_data) # DAOStarFinder(fwhm=FWHM, threshold=10*std)(masked_data - median_s)
    if(cords_transformed is None):
        print("No stars found, using previous coordinates") 
        cords_transformed = past_cord
    else:
        cords_transformed = get_max(masked_data) 
        increment = 5 # degrees to break up
        radius = 1 #radius between centers 
        degrees = np.arange(360/increment) * increment
        radians = np.radians(degrees)
        direction_matrix = np.array([np.cos(radians), np.sin(radians)]).T
        direction_matrix *= radius
        #plt.figure()
        
        #plt.imshow(calibrated_second, cmap='grey', origin='lower', vmin=median_s-2*std_s, vmax=median_s+5*std_s)
        
        aper_t = CircularAperture(cords_transformed[0], r=ap_radius)
        aper_t.plot(color='green')
        #plt.scatter(cords_transformed[0][0], cords_transformed[0][1])
        phot_table = aperture_photometry(calibrated_second, aper_t)
        
        best_mean = phot_table['aperture_sum'][0] / (np.pi * ap_radius**2)
        best_aperture = aper_t
        best_vec = cords_transformed[0]
        best_achieved = False
        best_previous = cords_transformed[0]
        while(best_achieved == False):
            for vec in direction_matrix: 
                #print('NEXT CORD')
                test_coord = (cords_transformed + np.flip(vec))[0]

                #print(test_coord)
                aper_t = CircularAperture(test_coord, r=ap_radius)
                #aper_t.plot(color='green')
                #plt.scatter(test_coord[0], test_coord[1])
                phot_table = aperture_photometry(calibrated_second, aper_t)
                mean_flux = phot_table['aperture_sum'][0] / (np.pi * ap_radius**2)
                if mean_flux > best_mean:
                    best_mean = mean_flux
                    best_aperture = aper_t
                    best_vec = vec
            if(np.array_equal(best_previous, best_vec)):
                best_achieved = True
            else: 
                best_previous = best_vec
                cords_transformed = cords_transformed + np.flip(best_vec)
                #print(f"New best mean flux: {best_mean}")
                #print(f"New best vector offset: {best_vec}")
                #print(f"New best transformed coords: {(cords_transformed + np.flip(best_vec))[0]}")
        
        if best_aperture is not None:
            best_aperture.plot(color='red', lw=2)  # highlight the best one
            #plt.scatter((cords_transformed + np.flip(best_vec))[0][0], (cords_transformed + np.flip(best_vec))[0][1], color='red', s=60, marker='x')
            print(f"Best aperture mean flux: {best_mean}")
            print(f"Best vector offset: {best_vec}")
            print(f"Best transformed coords: {(cords_transformed + np.flip(best_vec))[0]}") 
        
        #print(degrees)
        #print(direction_matrix) 

    
    #print(f"refinied cords {cords_transformed}")
    #plt.scatter((cords_transformed)[0][0], (cords_transformed)[0][1]) # add in our initial centroid. 
    
    #plt.show()
    #plt.close()
    #cords_transformed = DAOStarFinder(fwhm=FWHM, threshold=5.*std, xycoords=cords)
    #cords_transformed = np.array([(3499.825554241658, 39.37473823979557)]) 

    aper_t = CircularAperture(cords_transformed[0], r=ap_radius)
    ann_t = CircularAnnulus(cords_transformed[0], r_in=ann_inner, r_out=ann_inner+ann_width)


    
     

    #apertures = [aper, ann]
    #photom_table = aperture_photometry(calibrated, apertures) # photom_table will contain aperture_sum and annulus_sum for each entry

    #print(photom_table)


    """
    fig, ax = plt.subplots(figsize = (7,7))
    plt.imshow(calibrated, cmap='grey', origin='lower', vmin=median-2*std, vmax=median+5*std)
    aper.plot(color='green')
    ann.plot(color='cyan')
    plt.figure()
    """

    fig, ax = plt.subplots(figsize = (7,7))
    plt.imshow(calibrated_second, cmap='grey', origin='lower', vmin=median_s-2*std_s, vmax=median_s+5*std_s)
    aper_t.plot(color='red')
    ann_t.plot(color='cyan')
    #print(aper_t)

    xmin_pixel = -50 + cords_transformed[0][0]
    xmax_pixel = 50 + cords_transformed[0][0]
    ymin_pixel = -50 + cords_transformed[0][1]
    ymax_pixel = 50 + cords_transformed[0][1]
    
    if(xmin_pixel < 0):
        xmin_pixel = 0
    if(ymin_pixel < 0):
        ymin_pixel = 0
    
    if(xmax_pixel > nx):
        xmax_pixel = nx
    if(ymax_pixel > ny):
        ymax_pixel = ny 
    
    ax.set_xlim(xmin_pixel, xmax_pixel)
    ax.set_ylim(ymin_pixel, ymax_pixel)
    #print(xmin_pixel, xmax_pixel, ymin_pixel, ymax_pixel) 
    #plt.scatter(sources['xcentroid'], sources['ycentroid'], s=40, facecolors='none', edgecolors='r', label = "DOA")
    #plt.scatter(sources_2['xcentroid'], sources_2['ycentroid'], s=40, facecolors='none', edgecolors='b', label = "IRAF")
    #plt.legend(loc = "upper left")
    
    plt.title("Calibrated Image with photometry")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.savefig(fr"images\aperture_masks\{filename}.png")
    plt.close()
 
