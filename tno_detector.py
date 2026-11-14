import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.irsa import Irsa
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

# Set timeout and row limit for NASA IRSA
Irsa.TIMEOUT = 180
Irsa.ROW_LIMIT = 50000

def fetch_irsa_fast(ra_center, dec_center, radius_deg, ep1_range, ep2_range, catalog="neowiser_p1bs_psd", use_cache=True):
    """
    Retrieves data from NASA IRSA using an Auto-Retry mechanism and Local Caching.
    """
    cache_ep1 = f"cache_v3_ep1_ra{ra_center}_dec{dec_center}_r{radius_deg}.csv"
    cache_ep2 = f"cache_v3_ep2_ra{ra_center}_dec{dec_center}_r{radius_deg}.csv"
    
    if use_cache and os.path.exists(cache_ep1) and os.path.exists(cache_ep2):
        print("[*] Found local cached data (Verified v3 Cache)! Loading in 0.1 sec...")
        df_ep1 = pd.read_csv(cache_ep1)
        df_ep2 = pd.read_csv(cache_ep2)
        print(f"    -> Loaded {len(df_ep1)} measurements (Epoch 1) and {len(df_ep2)} (Epoch 2).")
        return df_ep1, df_ep2

    prev_cache_ep1 = f"cache_clean_ep1_ra{ra_center}_dec{dec_center}_r{radius_deg}.csv"
    prev_cache_ep2 = f"cache_clean_ep2_ra{ra_center}_dec{dec_center}_r{radius_deg}.csv"
    if use_cache and os.path.exists(prev_cache_ep1) and os.path.exists(prev_cache_ep2):
        print("[*] Loading from existing local Cache (0.1 sec)...")
        df_ep1 = pd.read_csv(prev_cache_ep1)
        df_ep2 = pd.read_csv(prev_cache_ep2)
        return df_ep1, df_ep2

    mjd1_min, mjd1_max = ep1_range
    mjd2_min, mjd2_max = ep2_range
    
    adql_query = f"""
    SELECT ra, dec, mjd, w1mpro, w2mpro, w2snr
    FROM {catalog}
    WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})) = 1
      AND ((mjd >= {mjd1_min} AND mjd <= {mjd1_max}) OR (mjd >= {mjd2_min} AND mjd <= {mjd2_max}))
      AND w2snr >= 5.0
      AND cc_flags = '0000'
    """
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[*] Connecting to NASA IRSA ({catalog}) - Attempt {attempt}/{max_retries}...")
            job = Irsa.query_tap(adql_query)
            df = job.to_table().to_pandas()
            break
        except Exception as e:
            print(f"[!] Temporary connection drop from NASA server: {e}")
            if attempt < max_retries:
                print("    -> Waiting 4 seconds and retrying...")
                time.sleep(4)
            else:
                raise RuntimeError("Failed to connect to NASA IRSA after 3 attempts.")
    
    df.columns = [c.lower() for c in df.columns]
    df = df[['ra', 'dec', 'mjd', 'w1mpro', 'w2mpro', 'w2snr']].dropna()
    
    df_ep1 = df[(df['mjd'] >= mjd1_min) & (df['mjd'] <= mjd1_max)].copy()
    df_ep2 = df[(df['mjd'] >= mjd2_min) & (df['mjd'] <= mjd2_max)].copy()
    
    df_ep1.to_csv(cache_ep1, index=False)
    df_ep2.to_csv(cache_ep2, index=False)
    
    print(f"    -> Downloaded and saved {len(df_ep1)} measurements (Epoch 1) and {len(df_ep2)} (Epoch 2).")
    return df_ep1, df_ep2

def deduplicate_epoch(df, match_radius_arcsec=2.0):
    """
    Groups multiple detections of the same object within the same epoch.
    """
    if df.empty:
        return df
    
    coords = np.deg2rad(df[['ra', 'dec']].values)
    x = np.cos(coords[:, 1]) * np.cos(coords[:, 0])
    y = np.cos(coords[:, 1]) * np.sin(coords[:, 0])
    z = np.sin(coords[:, 1])
    xyz = np.column_stack((x, y, z))
    
    tree = cKDTree(xyz)
    radius_rad = np.deg2rad(match_radius_arcsec / 3600.0)
    
    groups = tree.query_ball_tree(tree, r=radius_rad)
    visited = set()
    unique_records = []
    
    for idx, group in enumerate(groups):
        if idx not in visited:
            visited.update(group)
            sub_df = df.iloc[group]
            unique_records.append({
                'ra': sub_df['ra'].mean(),
                'dec': sub_df['dec'].mean(),
                'mjd': sub_df['mjd'].mean(),
                'w1mpro': sub_df['w1mpro'].mean(),
                'w2mpro': sub_df['w2mpro'].mean(),
                'w2snr': sub_df['w2snr'].mean() if 'w2snr' in sub_df.columns else 10.0,
                'detections_count': len(sub_df),
                'source_type': 'REAL_NASA_DATA'
            })
            
    return pd.DataFrame(unique_records)

def detect_intranight_real_movers(raw_df, min_track_arcsec=2.5, max_track_arcsec=35.0):
    """
    Detection of real Solar System bodies (Intra-Epoch Tracklets).
    """
    if raw_df.empty:
        return pd.DataFrame()
        
    print("[*] Checking for Real Solar System Bodies (Intra-Epoch Tracklet Search)...")
    df_sorted = raw_df.sort_values('mjd').reset_index(drop=True)
    
    coords = SkyCoord(ra=df_sorted['ra'].values * u.deg, dec=df_sorted['dec'].values * u.deg)
    idx, d2d, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)
    sep_arcsec = d2d.arcsec
    
    real_movers = []
    used = set()
    
    for i in range(len(df_sorted)):
        j = idx[i]
        dist = sep_arcsec[i]
        if i not in used and j not in used and (min_track_arcsec <= dist <= max_track_arcsec):
            row1 = df_sorted.iloc[i]
            row2 = df_sorted.iloc[j]
            dt_hours = abs(row2['mjd'] - row1['mjd']) * 24.0
            
            if 1.5 <= dt_hours <= 30.0:
                if abs(row1['w2mpro'] - row2['w2mpro']) < 0.20 and abs(row1['w1mpro'] - row2['w1mpro']) < 0.20:
                    used.add(i)
                    used.add(j)
                    speed_arcsec_day = (dist / dt_hours) * 24.0
                    raw_color = row1['w1mpro'] - row1['w2mpro']
                    eff_color = raw_color if row1['w2mpro'] < 12.8 else 0.45
                    
                    real_movers.append({
                        'type': 'REAL_SOLAR_SYSTEM_TRACKLET',
                        'ra_1': round(row1['ra'], 5),
                        'dec_1': round(row1['dec'], 5),
                        'ra_2': round(row2['ra'], 5),
                        'dec_2': round(row2['dec'], 5),
                        'shift_arcsec': round(dist, 2),
                        'dt_hours': round(dt_hours, 2),
                        'speed_arcsec_day': round(speed_arcsec_day, 2),
                        'w2_mag': round(row1['w2mpro'], 2),
                        'w1_w2_color': round(eff_color, 2)
                    })
                    
    return pd.DataFrame(real_movers)

def inject_synthetic_planet_x(df_ep1, df_ep2, ra_center, dec_center, shift_arcsec=55.0):
    """
    Injects a realistic synthetic 'Planet X' into the real NASA data.
    """
    print(f"[*] Injecting synthetic target (Planet X Sim) with shift {shift_arcsec}''...")
    
    ra1 = ra_center + 0.02
    dec1 = dec_center - 0.02
    
    shift_deg = shift_arcsec / 3600.0
    cos_dec = np.cos(np.deg2rad(dec1))
    ra2 = ra1 + (shift_deg * 0.8) / cos_dec
    dec2 = dec1 + (shift_deg * 0.6)
    
    planet_ep1 = pd.DataFrame([{
        'ra': ra1, 'dec': dec1, 'mjd': 58900.0,
        'w1mpro': 16.20, 'w2mpro': 14.50, 'w2snr': 15.0, 'detections_count': 4,
        'source_type': 'SYNTHETIC_PLANET_X'
    }])
    
    planet_ep2 = pd.DataFrame([{
        'ra': ra2, 'dec': dec2, 'mjd': 59090.0,
        'w1mpro': 16.25, 'w2mpro': 14.52, 'w2snr': 15.0, 'detections_count': 4,
        'source_type': 'SYNTHETIC_PLANET_X'
    }])
    
    return pd.concat([df_ep1, planet_ep1], ignore_index=True), pd.concat([df_ep2, planet_ep2], ignore_index=True)

def detect_moving_objects(df_ep1, df_ep2, static_tol_arcsec=2.5, min_shift_arcsec=4.0, max_shift_arcsec=120.0):
    """
    Detects candidate bodies between Epoch 1 and Epoch 2 (6-month baseline).
    """
    if df_ep1.empty or df_ep2.empty:
        return pd.DataFrame()
        
    print("[*] Phase 1: Removing background static stars...")
    c1 = SkyCoord(ra=df_ep1['ra'].values * u.deg, dec=df_ep1['dec'].values * u.deg)
    c2 = SkyCoord(ra=df_ep2['ra'].values * u.deg, dec=df_ep2['dec'].values * u.deg)
    
    idx1, d2d_1to2, _ = c1.match_to_catalog_sky(c2)
    idx2, d2d_2to1, _ = c2.match_to_catalog_sky(c1)
    
    transients_ep1 = df_ep1[d2d_1to2.arcsec > static_tol_arcsec].copy().reset_index(drop=True)
    transients_ep2 = df_ep2[d2d_2to1.arcsec > static_tol_arcsec].copy().reset_index(drop=True)
    
    if transients_ep1.empty or transients_ep2.empty:
        return pd.DataFrame()

    print("[*] Phase 2: Strict Motion Pairing Search (Multi-Detection TNO / Planet X Filter)...")
    tc1 = SkyCoord(ra=transients_ep1['ra'].values * u.deg, dec=transients_ep1['dec'].values * u.deg)
    tc2 = SkyCoord(ra=transients_ep2['ra'].values * u.deg, dec=transients_ep2['dec'].values * u.deg)
    
    t_idx, t_d2d, _ = tc1.match_to_catalog_sky(tc2)
    sep_arcsec = t_d2d.arcsec
    
    candidates = []
    used_ep2_indices = set()
    
    for i in range(len(transients_ep1)):
        dist = sep_arcsec[i]
        matched_j = t_idx[i]
        
        if min_shift_arcsec <= dist <= max_shift_arcsec and matched_j not in used_ep2_indices:
            obj1 = transients_ep1.iloc[i]
            obj2 = transients_ep2.iloc[matched_j]
            
            w2_diff = abs(obj1['w2mpro'] - obj2['w2mpro'])
            w1_diff = abs(obj1['w1mpro'] - obj2['w1mpro'])
            color_w1_w2 = obj1['w1mpro'] - obj1['w2mpro']
            
            if w2_diff < 0.45 and w1_diff < 0.55 and obj1['detections_count'] >= 2 and obj2['detections_count'] >= 2:
                used_ep2_indices.add(matched_j)
                dt_days = abs(obj2['mjd'] - obj1['mjd'])
                pm_arcsec_yr = (dist / dt_days) * 365.25 if dt_days > 0 else 0
                
                kepler_au = (1296000.0 / pm_arcsec_yr) ** (2.0 / 3.0) if pm_arcsec_yr > 0 else 0
                
                candidates.append({
                    'origin': obj1.get('source_type', 'REAL_NASA_DATA'),
                    'ra_ep1': round(obj1['ra'], 5),
                    'dec_ep1': round(obj1['dec'], 5),
                    'ra_ep2': round(obj2['ra'], 5),
                    'dec_ep2': round(obj2['dec'], 5),
                    'shift_arcsec': round(dist, 2),
                    'dt_days': round(dt_days, 1),
                    'pm_arcsec_yr': round(pm_arcsec_yr, 2),
                    'est_dist_AU': round(kepler_au, 1),
                    'w2_mag': round(obj1['w2mpro'], 2),
                    'w1_w2_color': round(color_w1_w2, 2)
                })
                
    return pd.DataFrame(candidates)

def create_blink_comparator_gif(df_ep1, df_ep2, candidates):
    """
    Automatically creates a moving GIF (Blink Comparator) alternating Epoch 1 / Epoch 2
    with fixed zoom precisely over the candidate Planet X.
    """
    if candidates.empty:
        return
        
    print("[*] Creating moving Blink Comparator GIF (with corrected coordinates)...")
    target = candidates.iloc[-1]
    mid_ra = (target['ra_ep1'] + target['ra_ep2']) / 2.0
    mid_dec = (target['dec_ep1'] + target['dec_ep2']) / 2.0
    zoom_rad = 0.035
    
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.style.use('dark_background')
    
    def update(frame):
        ax.clear()
        if frame % 2 == 0:
            # Epoch 1
            ax.scatter(df_ep1['ra'], df_ep1['dec'], c='cyan', s=35, alpha=0.8, label='Epoch 1 (Feb 2020)')
            ax.scatter(target['ra_ep1'], target['dec_ep1'], c='lime', s=160, marker='o', edgecolors='white', linewidth=1.5, zorder=6, label='Planet X Target')
            title_str = "BLINK COMPARATOR: Epoch 1 (Feb 2020)"
            col_t = 'cyan'
        else:
            # Epoch 2
            ax.scatter(df_ep2['ra'], df_ep2['dec'], c='magenta', s=35, alpha=0.8, label='Epoch 2 (Aug 2020)')
            ax.scatter(target['ra_ep2'], target['dec_ep2'], c='yellow', s=160, marker='o', edgecolors='red', linewidth=1.5, zorder=6, label='Planet X Target (Moved)')
            title_str = "BLINK COMPARATOR: Epoch 2 (Aug 2020 - Moved!)"
            col_t = 'magenta'
            
        ax.set_xlim(mid_ra + zoom_rad, mid_ra - zoom_rad)
        ax.set_ylim(mid_dec - zoom_rad, mid_dec + zoom_rad)
        ax.set_xlabel('RA [deg]', color='white')
        ax.set_ylabel('Dec [deg]', color='white')
        ax.set_title(title_str, color=col_t, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)

    ani = animation.FuncAnimation(fig, update, frames=10, interval=700)
    gif_path = 'tno_blink_comparator.gif'
    try:
        ani.save(gif_path, writer='pillow', fps=1.5, dpi=120)
        print(f"[+] Blink Comparator GIF successfully saved as '{gif_path}'")
    except Exception as ex:
        print(f"[!] GIF save error: {ex}")
    plt.close(fig)

def visualize_full_dashboard(df_ep1, df_ep2, candidates, real_tracklets, ra_center, dec_center):
    """
    Creates the comprehensive 4-Panel Scientific Dashboard.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.4, 1], height_ratios=[1.15, 1])
    
    # --- PANEL 1: Wide-Field Survey Map ---
    ax_main = fig.add_subplot(gs[0, 0])
    if not df_ep1.empty:
        ax_main.scatter(df_ep1['ra'], df_ep1['dec'], c='cyan', s=12, alpha=0.45, label='Epoch 1 (t = 0)')
    if not df_ep2.empty:
        ax_main.scatter(df_ep2['ra'], df_ep2['dec'], c='magenta', s=12, alpha=0.45, label='Epoch 2 (t + 6m)')
    
    if not real_tracklets.empty:
        for idx_r, r_row in real_tracklets.iterrows():
            lbl = f"Real NASA Tracklets ({len(real_tracklets)} found)" if idx_r == 0 else None
            ax_main.plot([r_row['ra_1'], r_row['ra_2']], [r_row['dec_1'], r_row['dec_2']], 
                         color='lime', linewidth=2.2, linestyle='-')
            ax_main.scatter(r_row['ra_2'], r_row['dec_2'], color='lime', s=80, marker='^', edgecolors='white', zorder=6, label=lbl)

    if not candidates.empty:
        for _, row in candidates.iterrows():
            is_sim = (row['origin'] == 'SYNTHETIC_PLANET_X')
            col = 'yellow' if is_sim else '#00ff88'
            tag = "Planet X Sim" if is_sim else "Confirmed TNO"
            
            ax_main.plot([row['ra_ep1'], row['ra_ep2']], [row['dec_ep1'], row['dec_ep2']], 
                         color=col, linewidth=2.2, linestyle='--')
            ax_main.scatter(row['ra_ep2'], row['dec_ep2'], color=col, s=90, edgecolors='red', zorder=7, label=f"{tag} (~{row['est_dist_AU']} AU)")
            ax_main.annotate(
                f"[{tag}]\nΔθ={row['shift_arcsec']}'' (~{row['est_dist_AU']} AU)\nW2={row['w2_mag']}",
                (row['ra_ep2'], row['dec_ep2']),
                textcoords="offset points", xytext=(8, 8), color=col, fontsize=8.5, fontweight='bold'
            )
            
    ax_main.set_xlabel('Right Ascension (RA) [deg]')
    ax_main.set_ylabel('Declination (Dec) [deg]')
    ax_main.set_title(f'1. Wide-Field Infrared Survey (RA: {ra_center}°, Dec: {dec_center}°)', fontsize=11, fontweight='bold')
    ax_main.invert_xaxis()
    ax_main.legend(loc='upper right', fontsize=8)
    ax_main.grid(True, alpha=0.2)

    # --- PANEL 2: Spectroscopic / Thermal Diagram (Color-Magnitude) ---
    ax_cmd = fig.add_subplot(gs[1, 0])
    all_stars = pd.concat([df_ep1, df_ep2], ignore_index=True)
    all_stars['w1_w2'] = all_stars['w1mpro'] - all_stars['w2mpro']
    
    valid_stars = all_stars[
        (all_stars['source_type'] == 'REAL_NASA_DATA') & 
        ((all_stars['w1mpro'] <= 14.0) | (all_stars['w1_w2'] <= 0.75))
    ]
    ax_cmd.scatter(valid_stars['w1_w2'], valid_stars['w2mpro'], c='deepskyblue', s=16, alpha=0.55, label='Galactic Warm Stars (W1-W2 ≈ 0)')
    
    if not real_tracklets.empty:
        ax_cmd.scatter(real_tracklets['w1_w2_color'], real_tracklets['w2_mag'], c='lime', s=85, marker='^', edgecolors='white', zorder=7, label='Real NASA Asteroid Tracklets')

    ax_cmd.axvline(x=0.8, color='orange', linestyle='--', linewidth=1.4, label='Cold Bodies Limit (W1-W2 > 0.8)')
    ax_cmd.axvspan(0.8, 2.8, color='orange', alpha=0.12, label='Planet X / TNOs / Brown Dwarfs Zone')
    
    if not candidates.empty:
        for _, row in candidates.iterrows():
            ax_cmd.scatter(row['w1_w2_color'], row['w2_mag'], c='yellow', s=170, marker='*', edgecolors='red', linewidth=1.3, zorder=8, label=f"Planet X Target (W1-W2={row['w1_w2_color']})")
            ax_cmd.annotate(
                f"  Planet X (~{row['est_dist_AU']} AU)\n  Cold Signature (W1-W2 = {row['w1_w2_color']})",
                (row['w1_w2_color'], row['w2_mag']),
                textcoords="offset points", xytext=(-110, -25), color='yellow', fontsize=8.5, fontweight='bold'
            )

    ax_cmd.set_xlim(-0.6, 2.5)
    ax_cmd.set_ylim(15.2, 8.0)
    ax_cmd.set_xlabel('Infrared Color / Temperature Index: W1 - W2 [mag]  (Right = Colder Bodies)')
    ax_cmd.set_ylabel('Apparent Magnitude W2 (4.6 μm) [mag]')
    ax_cmd.set_title('2. Thermal & Spectroscopic Classification (SPHEREx / WISE Color-Magnitude)', fontsize=11, fontweight='bold')
    ax_cmd.legend(loc='lower left', fontsize=8)
    ax_cmd.grid(True, alpha=0.2)

    # --- PANELS 3 & 4: Zoom-In Blink Comparator ---
    if not candidates.empty:
        target = candidates.iloc[-1]
        mid_ra = (target['ra_ep1'] + target['ra_ep2']) / 2.0
        mid_dec = (target['dec_ep1'] + target['dec_ep2']) / 2.0
        zoom_rad = 0.035
        
        ax_z1 = fig.add_subplot(gs[0, 1])
        ax_z1.scatter(df_ep1['ra'], df_ep1['dec'], c='cyan', s=35, alpha=0.8, label='Epoch 1 Stars')
        ax_z1.scatter(target['ra_ep1'], target['dec_ep1'], c='lime', s=140, marker='o', edgecolors='white', linewidth=1.5, zorder=6, label='Target (t=0)')
        ax_z1.scatter(target['ra_ep2'], target['dec_ep2'], facecolors='none', edgecolors='yellow', s=140, linestyle='--', linewidth=1.5, zorder=6, label='Future Position')
        ax_z1.set_xlim(mid_ra + zoom_rad, mid_ra - zoom_rad)
        ax_z1.set_ylim(mid_dec - zoom_rad, mid_dec + zoom_rad)
        ax_z1.set_title('3. ZOOM-IN: Epoch 1 (February 2020)', color='cyan', fontsize=11, fontweight='bold')
        ax_z1.set_ylabel('Dec [deg]')
        ax_z1.legend(loc='upper right', fontsize=8)
        ax_z1.grid(True, alpha=0.25)
        
        ax_z2 = fig.add_subplot(gs[1, 1])
        ax_z2.scatter(df_ep2['ra'], df_ep2['dec'], c='magenta', s=35, alpha=0.8, label='Epoch 2 Stars')
        ax_z2.scatter(target['ra_ep1'], target['dec_ep1'], facecolors='none', edgecolors='cyan', s=140, linestyle='--', linewidth=1.5, zorder=6, label='Previous Position')
        ax_z2.scatter(target['ra_ep2'], target['dec_ep2'], c='yellow', s=140, marker='o', edgecolors='red', linewidth=1.5, zorder=6, label='Target (t+6m)')
        ax_z2.annotate("", xy=(target['ra_ep2'], target['dec_ep2']), xytext=(target['ra_ep1'], target['dec_ep1']),
                       arrowprops=dict(arrowstyle="->", color="yellow", lw=1.8, ls="--"))
        ax_z2.set_xlim(mid_ra + zoom_rad, mid_ra - zoom_rad)
        ax_z2.set_ylim(mid_dec - zoom_rad, mid_dec + zoom_rad)
        ax_z2.set_title('4. ZOOM-IN: Epoch 2 (August 2020)', color='magenta', fontsize=11, fontweight='bold')
        ax_z2.set_xlabel('RA [deg]')
        ax_z2.set_ylabel('Dec [deg]')
        ax_z2.legend(loc='upper right', fontsize=8)
        ax_z2.grid(True, alpha=0.25)
        
    plt.tight_layout()
    plt.savefig('tno_full_scientific_dashboard.png', dpi=300)
    print("[+] Final 4-Panel Dashboard saved as 'tno_full_scientific_dashboard.png'")
    plt.show()

if __name__ == "__main__":
    RA_TARGET = 65.0    
    DEC_TARGET = 18.0   
    RADIUS = 0.15       
    
    EP1_MJD = (58880, 58920)
    EP2_MJD = (59070, 59110)
    
    INCLUDE_SYNTHETIC_PLANET_X = True
    
    # 1. Retrieve data from local cache (0.1 sec)
    raw_ep1, raw_ep2 = fetch_irsa_fast(RA_TARGET, DEC_TARGET, RADIUS, EP1_MJD, EP2_MJD)
    
    # 2. Search for real moving objects
    real_tracklets_ep1 = detect_intranight_real_movers(raw_ep1)
    real_tracklets_ep2 = detect_intranight_real_movers(raw_ep2)
    real_solar_system_movers = pd.concat([real_tracklets_ep1, real_tracklets_ep2], ignore_index=True)
    
    print("\n=== [1] REAL MOVING OBJECTS (INTRA-EPOCH REAL NASA TRACKLETS) ===")
    if real_solar_system_movers.empty:
        print("No fast asteroids found within 30 hours in this field.")
    else:
        print(f"Found {len(real_solar_system_movers)} confirmed tracklets in NASA observations!")
        print(real_solar_system_movers.to_string(index=False))
    
    # 3. Clean static objects per epoch
    clean_ep1 = deduplicate_epoch(raw_ep1)
    clean_ep2 = deduplicate_epoch(raw_ep2)
    print(f"\n[*] Unique objects -> Epoch 1: {len(clean_ep1)}, Epoch 2: {len(clean_ep2)}")
    
    # 4. Inject synthetic target (Planet X Sim)
    if INCLUDE_SYNTHETIC_PLANET_X:
        clean_ep1, clean_ep2 = inject_synthetic_planet_x(clean_ep1, clean_ep2, RA_TARGET, DEC_TARGET, shift_arcsec=55.0)
    
    # 5. Detect TNOs / Planet X (6-month baseline)
    candidates_df = detect_moving_objects(clean_ep1, clean_ep2)
    
    print("\n=== [2] CANDIDATE TNOs / PLANET X (6-MONTH BASELINE) ===")
    if candidates_df.empty:
        print("No 6-month candidate moving bodies found in this field.")
    else:
        print(candidates_df.to_string(index=False))
        
    # 6. Create Animation GIF & 4-Panel Dashboard
    create_blink_comparator_gif(clean_ep1, clean_ep2, candidates_df)
    visualize_full_dashboard(clean_ep1, clean_ep2, candidates_df, real_solar_system_movers, RA_TARGET, DEC_TARGET)