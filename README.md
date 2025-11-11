# LassiStarNav
**Photometry & Star Navigation Toolkit**

LassiStarNav is a Python-based toolkit designed to build the applied **star navigation** and photometric light-curve extraction workflows.  

It provides modules for centroid detection, FITS-file analysis, **navigation algorithms**, and photometric testing utilities. This toolkit is especially useful for **spacecraft autonomy**, exoplanet research, and **star tracker applications.**

This work was done in collaboration with the Laboratory of Advanced Space System at Illinois as well as the Astrodynamics and Planetary Exploration research group at **UIUC**

---

## 🌌 Overview
This project enables:
- Extracting stellar centroids from telescope or simulated imagery  
- Performing aperture photometry and background subtraction  
- Running a navigation algorithm for star-field mapping  
- Generating light curves from observational data  
- Testing routines for photometry and navigation workflows  

If you’re working on **satellite navigation, star trackers, or light curve analysis, LassiStarNav offers a modular foundation.**


> **Note:**  
> Some data files, outputs, and logs are **locally hosted** and intentionally excluded from this repository as indicated in the `.gitignore`.  
> These include large datasets, FITS images, and test results not suitable for public hosting.


---

## ⚙️ Features
- **Centroid Detection** → via `Centroid_Tester.py` and `tester_methods.py`  
- **FITS Handling** → through `FitsTester.py` (load, inspect, and process FITS data)  
- **Photometry Pipeline** → `Lassi_photometry.py` for aperture-based flux extraction  
- **Star Navigation Algorithm** → `Navigation_algo.py` performs star-field matching and navigation computations  
- **Light Curve Extraction** → `light_curve_extraction.py` and `lightkurve_testing.py` for photometric analysis  
- **Sample Real-Data Maps** → Includes datasets for *Alderamin*, *Delta Scuti*, and *Tau Cygni* fields  

---

## 🧠 Getting Started

### Prerequisites
You’ll need:
```bash
Python >= 3.8
numpy
scipy
astropy
pandas
matplotlib
lightkurve   # optional, for testing with the lightkurve library
