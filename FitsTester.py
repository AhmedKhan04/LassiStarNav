from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

fits_file = r"C:\Users\ahmed\Downloads\Alderamin (Alpha Cephi)_00001.fits"
hdul = fits.open(fits_file)

print("\n=== FITS Header ===")
print(repr(hdul[0].header)) 

data = hdul[0].data
if data is None and len(hdul) > 1:
    data = hdul[1].data

if data is not None:
    data = np.nan_to_num(data)
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap='gray', origin='lower')
    plt.colorbar(label='Pixel value')
    plt.title(f"FITS Image: {fits_file}")
    plt.show()
else:
    print("No image data found in the FITS file.")

hdul.close()
