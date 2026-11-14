# 🪐 AstroBlink-IR: Automated TNO & Planet X Detection Pipeline

An end-to-end Python data pipeline designed to automate the discovery of Trans-Neptunian Objects (TNOs) and hypothetical Planet X candidates using wide-field infrared survey data from NASA/Caltech (NEOWISE / SPHEREx).

## 🚀 Key Features
- **Direct ADQL/TAP Ingestion:** Automated querying of NASA IRSA astronomical databases with local caching and auto-retry mechanisms.
- **3D Spatial Deduplication:** Efficient sky coordinate matching and grouping using `cKDTree`.
- **Dual-Scale Astrometric Filtering:** 
  - *Intra-Epoch Tracklets:* Real-time tracking of fast Solar System bodies within a 30-hour window.
  - *6-Month Baseline:* Precision background star subtraction ($< 2.5''$) and motion pairing for remote outer-solar-system objects.
- **Keplerian Distance Estimator:** Computes orbital semi-major axes ($a$ in AU) via Kepler's Third Law based on 6-month parallax/proper motion.
- **Spectro-Photometric Classification:** Color-magnitude analysis ($W1 - W2$ vs $W2$) to separate warm galactic stars from cold planetary bodies.
- **Professional Visualization:** Generates a comprehensive **4-Panel Scientific Dashboard** and an animated **Blink Comparator GIF**.

## 📊 Pipeline Architecture (Data Processing Levels)
1. **Level 1b Ingestion:** Raw single-exposure source measurements from NASA IRSA (`neowiser_p1bs_psd`).
2. **Level 2/3 Processing:** Astrometric cleaning, spatial deduplication, and background subtraction.
3. **Level 4 Analytics:** Final candidate catalogs, Keplerian distances, and thermal classifications.

## 🛠️ Tech Stack
- **Python 3.x**
- **Astropy** (SkyCoord, units)
- **Astroquery** (NASA IRSA TAP client)
- **SciPy** (cKDTree spatial indexing)
- **Pandas & NumPy** (Data manipulation)
- **Matplotlib** (Dashboard & Animation rendering)

## 🏃‍♂️ Quick Start
1. Clone the repository.
2. Ensure dependencies are installed (`pip install astropy astroquery scipy pandas matplotlib`).
3. Run the main script:
   ```bash
   python tno_detector.py
