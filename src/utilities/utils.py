# AP 2026

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patheffects as pe

from astropy import units as u
from astropy.coordinates import SkyCoord


def include_scale_bar(fig, ax, wcs, x_min, x_max, y_min, y_max, scale_bar_length_arcsec=2, color='white', fontsize=8):
    scale_bar_length_pixels = scale_bar_length_arcsec / wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
    crop_h, crop_w = (y_max - y_min), (x_max - x_min)
    margin = 0.05 * crop_h
    x0 = margin
    y0 = margin  # anchor near the bottom-left of the actual crop
    outline = [pe.withStroke(linewidth=4, foreground='black')]
    ax.plot([x0, x0 + scale_bar_length_pixels], [y0, y0], color=color, linewidth=2.5,
            solid_capstyle='butt', path_effects=outline)
    ax.text(x0, y0 + 0.02 * crop_h, f'{scale_bar_length_arcsec} arcsec', color=color,
            fontsize=fontsize, va='bottom', path_effects=outline)
    ax.axis('off')

def include_interface_box(fig, ax, x_pixel, y_pixel, wcs, x_min, y_min, box_size_arcsec=5.75, color='dodgerblue'):
    box_size_pixels = box_size_arcsec / wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
    rect = plt.Rectangle((x_pixel - x_min - box_size_pixels / 2, y_pixel - y_min - box_size_pixels / 2),
                            box_size_pixels, box_size_pixels, edgecolor=color, facecolor='none',
                            linestyle='--', label='Inspection equivalent')
    ax.add_patch(rect)

def include_marker_scaled(fig, ax, x_pixel, y_pixel, wcs, x_min, y_min, marker_size_arcsec=5.2, color='red', obj_id=None):
    pixscale_arcsec = wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
    radius_pix = marker_size_arcsec / pixscale_arcsec  # radius in image pixels

    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((radius_pix, 0))
    radius_pts = (p1[0] - p0[0]) * 72.0 / fig.dpi

    sizeOfMarker = np.pi * (radius_pts ** 2)  # scatter expects area in points^2
    ax.scatter(x_pixel - x_min, y_pixel - y_min, s=sizeOfMarker, edgecolor=color, facecolor='none',
                label=f'ID: {obj_id} Res @ {marker_size_arcsec}"')

def include_ellipse_scaled(fig, ax, x_pixel, y_pixel, obj_id, wcs, x_min, y_min, a_arcsec=5.2, b_arcsec=3.0, pa_deg=30, color='red'):
    # print(f"Including ellipse with a={a_arcsec}\" and b={b_arcsec}\" at PA={pa_deg} degrees for object ID: {obj_id}")
    pixscale_arcsec = wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
    a_pix = a_arcsec / pixscale_arcsec  # semi-major axis in image pixels
    b_pix = b_arcsec / pixscale_arcsec  # semi-minor axis in image pixels

    ellipse = plt.matplotlib.patches.Ellipse((x_pixel - x_min, y_pixel - y_min), width=2*a_pix, height=2*b_pix,
                    angle=pa_deg, edgecolor=color, facecolor='none', linestyle='--',
                    label=f'ID: {obj_id} Ellipse @ {a_arcsec:<.2f}"x{b_arcsec:<.2f}" PA={pa_deg:<.2f}°')
    ax.add_patch(ellipse)

def include_small_crosshair(fig, ax, x_pixel, y_pixel, x_min, y_min, size_pixels=1, color='red', alpha=0.8):
    ax.plot([x_pixel - x_min - size_pixels, x_pixel - x_min + size_pixels],
            [y_pixel - y_min, y_pixel - y_min], color=color, linewidth=1, alpha=alpha)
    ax.plot([x_pixel - x_min, x_pixel - x_min],
            [y_pixel - y_min - size_pixels, y_pixel - y_min + size_pixels], color=color, linewidth=1, alpha=alpha)

def plot_panel(fig, ax, image, wcs, hdr, ra, dec, obj_id, title_prefix, zoom_size=None, resolution_arcsec=None, statistic_resolution_arcsec=None, cmap='inferno'):
    """Crop `image` around the object's own pixel position (from `idic`), display it, and annotate."""
    # if obj_id not in idic:
    if not check_point_is_in_image(image, wcs, ra, dec):
        # print(f"Object ID: {obj_id} is out of image bounds for the given RA and Dec: RA={ra}, Dec={dec}. Where image shape is {image.shape} and WCS info is {wcs}")
        ax.text(0.5, 0.5, f'{title_prefix} data not available',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title_prefix)
        ax.axis('off')
        return
    # x_pixel, y_pixel, intensity_point, intensity_avg, intensity_peak = idic[obj_id]

    # Pixel scale and beam sizes are needed regardless of whether `resolution_arcsec`
    # is supplied, so compute them up-front (previously they were only defined inside
    # the `resolution_arcsec is None` branch, which raised NameError otherwise).
    pixel_scales = np.abs(wcs.celestial.proj_plane_pixel_scales()[0]).to(u.deg)
    if 'BMAJ' in hdr:
        beamsizeMaj = hdr['BMAJ'] * u.deg
    else:
        print(f"Warning: BMAJ keyword not found in header. Using default beamsize of {resolution_arcsec} arcsec.")
        beamsizeMaj = (resolution_arcsec if resolution_arcsec is not None else 1) * u.arcsec
    if 'BMIN' in hdr:
        beamsizeMin = hdr['BMIN'] * u.deg
    else:
        print(f"Warning: BMIN keyword not found in header. Using default beamsize of {resolution_arcsec} arcsec.")
        beamsizeMin = (resolution_arcsec if resolution_arcsec is not None else 1) * u.arcsec
        
    #beam angle
    if 'BPA' in hdr:
        beam_pa = hdr['BPA'] * u.deg
    else:
        print(f"Warning: BPA keyword not found in header. Using default beam position angle of 0 degrees.")
        beam_pa = 0 * u.deg
    beamsize_maj_arcsec = beamsizeMaj.to(u.arcsec).value
    beamsize_min_arcsec = beamsizeMin.to(u.arcsec).value

    if resolution_arcsec is None:
        # radius_for_peak_arcsec as 1*BMAJ in arcseconds for each image.
        radius = int(np.ceil((1 * beamsizeMaj / pixel_scales).decompose().value)) # in pixels, then convert to arcseconds for the function input
        radius_for_peak_arcsec = (radius * pixel_scales).to(u.arcsec).value #akin to resolution, since it's 1*BMAJ, and convert it to arcseconds
        # print(f"{title_prefix} beamsize: {beamsize_maj_arcsec:.2f} x {beamsize_min_arcsec:.2f} arcsec, pixel scale: {pixel_scales.to(u.arcsec)}, radius: {radius} pixels, radius for peak measurement: {radius_for_peak_arcsec:.2f} arcsec, radius for statistics: {radius_for_statistics_arcsec:.2f} arcsec")
    else:
        radius_for_peak_arcsec = resolution_arcsec
        # print(f"{title_prefix} using provided resolution for analysis: {resolution_arcsec} arcsec, radius for peak measurement: {radius_for_peak_arcsec:.2f} arcsec, radius for statistics: {radius_for_statistics_arcsec:.2f} arcsec")
        
    if statistic_resolution_arcsec is not None:
        radius_for_statistics_arcsec = statistic_resolution_arcsec

    elif resolution_arcsec is not None and statistic_resolution_arcsec is None:
        radius_for_statistics_arcsec = round(radius_for_peak_arcsec * 5)

    elif resolution_arcsec is None and statistic_resolution_arcsec is None:
        radius_for_statistics_arcsec = round(radius_for_peak_arcsec * 5)

        
    x_pixel, y_pixel, intensity_point, intensity_avg, intensity_peak = get_peak_within_radius(image, wcs, hdr, obj_id, ra, dec, radius_arcsec=radius_for_peak_arcsec)

    # Fixed SQUARE window centered on the object so every panel has identical 1:1 aspect
    # and the columns sit flush (no internal whitespace). Out-of-image area is padded with NaN.
    half = zoom_size
    x_min, x_max = x_pixel - half, x_pixel + half
    y_min, y_max = y_pixel - half, y_pixel + half

    crop = np.full((2 * half, 2 * half), np.nan, dtype=float)
    sx0, sx1 = max(0, x_min), min(image.shape[1], x_max)
    sy0, sy1 = max(0, y_min), min(image.shape[0], y_max)
    crop[sy0 - y_min:sy1 - y_min, sx0 - x_min:sx1 - x_min] = image[sy0:sy1, sx0:sx1]

    # Guard against an all-NaN crop (object sits in a fully-blank/edge region):
    # np.nanpercentile on an all-NaN array raises, so fall back to a default scale.
    if np.all(np.isnan(crop)):
        vmin, vmax = 0, 1
    else:
        vmin, vmax = np.nanpercentile(crop, 10), np.nanpercentile(crop, 99)
    ax.imshow(crop, origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    # include_marker_scaled(fig, ax, x_pixel, y_pixel, wcs, x_min, y_min, marker_size_arcsec=circle_size_arcsec, color='red')
    include_interface_box(fig, ax, x_pixel, y_pixel, wcs, x_min, y_min, box_size_arcsec=5.75, color='dodgerblue')
    include_scale_bar(fig, ax, wcs, x_min, x_max, y_min, y_max, scale_bar_length_arcsec=2, color='white')
    include_small_crosshair(fig, ax, x_pixel, y_pixel, x_min, y_min, size_pixels=0.5, color='red', alpha=0.5)
    include_ellipse_scaled(fig, ax, x_pixel, y_pixel, obj_id, wcs, x_min, y_min, a_arcsec=beamsize_maj_arcsec, b_arcsec=beamsize_min_arcsec, pa_deg=beam_pa.to(u.deg).value, color='red')


    stdev_value, rms_value = get_statistics_with_sigma_clipping(image, x_pixel, y_pixel, radius_arcsec=radius_for_statistics_arcsec, sigma_threshold=3, pixscale_arcsec=pixel_scales.to(u.arcsec).value)
    sigma3 = 3 * stdev_value

    ax.set_title(f"{title_prefix}: {radius_for_peak_arcsec:<.2f}\" radius point: {intensity_point:.3g}, \n avg: {intensity_avg:.3g}, peak: {intensity_peak:.3g} \n {radius_for_statistics_arcsec:<.2f}\" radius stdev: {stdev_value:.3g}, \n RMS: {rms_value:.3g}, 3-sigma: {sigma3:.3g}")

#first measure the stdev and rms in a 5 arcsec radius box around the object, and then remove any f_pixels/noise that are above 3 sigma in that box, and then re-measure the stdev and rms with
def get_statistics_with_sigma_clipping(image, x_pixel, y_pixel, sigma_threshold, radius_arcsec=None, pixscale_arcsec=None, wcs=None, hdr=None):
    if radius_arcsec is None or pixscale_arcsec is None:
        #make the radius for statistics as 5 times the radius for peak measurement, which is based on the resolution of the image (1*BMAJ), and convert it to pixels using the pixel scale from the WCS information in the header.
        if wcs is not None and hdr is not None:
            if 'BMAJ' in hdr:
                beamsizeMaj = hdr['BMAJ'] * u.deg
            else:
                print(f"Warning: BMAJ keyword not found in header. Using default beamsize of 1 arcsec.")
                beamsizeMaj = 1 * u.arcsec
            pixel_scales = np.abs(wcs.celestial.proj_plane_pixel_scales()[0]).to(u.deg)
            radius = int(np.ceil((1 * beamsizeMaj / pixel_scales).decompose().value)) # in pixels, then convert to arcseconds for the function input
            radius_arcsec = round(5 * (radius * pixel_scales).to(u.arcsec).value)
            pixscale_arcsec = pixel_scales.to(u.arcsec).value
            print(f"Using a radius of {radius_arcsec} arcsec for statistics based on 5 times BMAJ and pixel scale, since radius_arcsec or pixscale_arcsec was not provided. Computed radius for statistics is {radius_arcsec:.2f} arcsec with pixel scale of {pixscale_arcsec:.2f} arcsec/pixel.")
        else:
            print(f"Warning: radius_arcsec or pixscale_arcsec not provided, and WCS or header information not available to compute them. Defaulting to a radius of 10 arcsec for statistics and pixel scale of 1 arcsec/pixel.")
            radius_arcsec = 10  # default radius in arcseconds for statistics if not provided and cannot be computed
            pixscale_arcsec = 1  # default pixel scale in arcseconds/pixel if not provided and cannot be computed
            print(f"Using default radius of {radius_arcsec} arcsec for statistics and pixel scale of {pixscale_arcsec} arcsec/pixel.")
            
    else:
        print(f"Using provided radius of {radius_arcsec} arcsec for statistics and pixel scale of {pixscale_arcsec} arcsec/pixel.")
    
    # at least 1 pixel radius so the statistics box is never zero-size
    radius_pixels = max(1, int(radius_arcsec / pixscale_arcsec))
    x_min_statistic = max(0, x_pixel - radius_pixels)
    x_max_statistic = min(image.shape[1], x_pixel + radius_pixels + 1)
    y_min_statistic = max(0, y_pixel - radius_pixels)
    y_max_statistic = min(image.shape[0], y_pixel + radius_pixels + 1)
    sub_image = image[y_min_statistic:y_max_statistic, x_min_statistic:x_max_statistic]
    if sub_image.size == 0 or np.all(np.isnan(sub_image)):
        return np.nan, np.nan
    stdev_value = np.nanstd(sub_image)
    mean_value = np.nanmean(sub_image)
    sigma_threshold_value = mean_value + sigma_threshold * stdev_value
    clipped_sub_image = np.where(sub_image > sigma_threshold_value, np.nan, sub_image)
    if np.all(np.isnan(clipped_sub_image)):
        return np.nan, np.nan
    stdev_clipped = np.nanstd(clipped_sub_image)
    rms_clipped = np.sqrt(np.nanmean(clipped_sub_image**2))
    return stdev_clipped, rms_clipped

def get_peak_within_radius(image, wcs, hdr, id, ra, dec, radius_arcsec = None):
    if radius_arcsec is None:
        if 'BMAJ' in hdr:
            beamsizeMaj = hdr['BMAJ'] * u.deg
        else:
            print(f"Warning: BMAJ keyword not found in header. Using default beamsize of 1 arcsec.")
            beamsizeMaj = 1 * u.arcsec
        pixel_scales = np.abs(wcs.celestial.proj_plane_pixel_scales()[0]).to(u.deg)
        radius = int(np.ceil((1 * beamsizeMaj / pixel_scales).decompose().value)) # in pixels, then convert to arcseconds for the function input
        # radius_for_peak_arcsec = (radius * pixel_scales).to(u.arcsec).value #akin to resolution, since it's 1*BMAJ, and convert it to arcseconds
        radius_arcsec = (radius * pixel_scales).to(u.arcsec).value
        print(f"Using a radius of {radius_arcsec} arcsec for peak extraction based on BMAJ and pixel scale.")
    else:
        print(f"Using the given radius of {radius_arcsec} arcsec for peak extraction as provided.")

    pixscale_arcsec = wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value

    x_pixel, y_pixel = wcs.celestial.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    x_pixel, y_pixel = int(round(float(x_pixel))), int(round(float(y_pixel)))
    # at least 1 pixel radius so the extraction box is never zero-size (caused
    # "zero-size array to reduction operation fmax" when radius_arcsec < pixscale).
    radius_pixels = max(1, int(round(radius_arcsec / pixscale_arcsec)))
    x_min = max(0, x_pixel - radius_pixels)
    x_max = min(image.shape[1], x_pixel + radius_pixels + 1)
    y_min = max(0, y_pixel - radius_pixels)
    y_max = min(image.shape[0], y_pixel + radius_pixels + 1)
    sub_image = image[y_min:y_max, x_min:x_max]
    if sub_image.size == 0 or np.all(np.isnan(sub_image)):
        peak_value = np.nan
        avg_value = np.nan
    else:
        peak_value = np.nanmax(sub_image)
        avg_value = np.nanmean(sub_image)
    # peaks.append(peak_value)
    if (0 <= y_pixel < image.shape[0]) and (0 <= x_pixel < image.shape[1]):
        point_intensity = image[y_pixel, x_pixel]
    else:
        point_intensity = np.nan
    return x_pixel, y_pixel, point_intensity, avg_value, peak_value

def check_point_is_in_image(image, wcs, ra, dec):
    x_pixel, y_pixel = wcs.celestial.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    x_pixel, y_pixel = int(round(float(x_pixel))), int(round(float(y_pixel)))
    if (0 <= x_pixel < image.shape[1]) and (0 <= y_pixel < image.shape[0]):
        return True
    else:
        print(f"Point with RA={ra}, Dec={dec} is out of image bounds. Where image shape is {image.shape} and WCS info is {wcs} and pixel coordinates are x={x_pixel}, y={y_pixel}")
        return False

# plot the SNR for each frequency point in the spectra, as SNR = flux / flux_error, and plot it on a secondary y-axis on the same plot as the spectra, with a horizontal line at SNR=3 to indicate the detection threshold.
def plot_alma_spectra_with_snr(S2COSP0762_ALMA_SPECTRA, ALMA_ID, COSMOS_WEB_ID, show_plot=True):
    alma_spectra_fig, alma_spectra_ax, coeffs, alpha, A, fitted_flux, s2cos0762_freq, s2cos0762_flux, s2cos0762_flux_error, signal_mask, bin_size = plot_alma_spectra(S2COSP0762_ALMA_SPECTRA, ALMA_ID, COSMOS_WEB_ID, show_plot=False)
    
    # Calculate SNR
    snr = abs(s2cos0762_flux) / abs(s2cos0762_flux_error)
    
    # Create a secondary y-axis for SNR
    ax_snr = alma_spectra_ax.twinx()
    ax_snr.plot(s2cos0762_freq, snr, label='SNR', color='green', linestyle=':')
    ax_snr.axhline(y=3, color='red', linestyle='--', label='SNR=3 Threshold')
    ax_snr.set_ylabel('Signal-to-Noise Ratio (SNR)')
    ax_snr.set_ylim(bottom=0)
    
    # Add legends
    alma_spectra_ax.legend(loc='upper left')
    ax_snr.legend(loc='upper right')
    
    if show_plot:
        plt.show()
    
    return alma_spectra_fig, alma_spectra_ax, ax_snr

def plot_alma_spectra(S2COSP0762_ALMA_SPECTRA_file = None, spectraData = None, ALMA_ID = None, COSMOS_WEB_ID = None, show_plot=True, bin_factor=2, ylim=None, xlim = None):
	#load S2COSP0762_ALMA_SPECTRA from txt (space separated) into numpy array, each row is freq, flux, flux_error
	if S2COSP0762_ALMA_SPECTRA_file is not None:
		s2cos0762_spectra = np.loadtxt(S2COSP0762_ALMA_SPECTRA_file)
	elif spectraData is not None:
		s2cos0762_spectra = spectraData
	else:
		raise ValueError("Either S2COSP0762_ALMA_SPECTRA_file or spectraData must be provided")

	s2cos0762_freq = s2cos0762_spectra[:, 0]  # in GHz
	s2cos0762_flux = s2cos0762_spectra[:, 1]  # in mJy
	s2cos0762_flux_error = s2cos0762_spectra[:, 2]  # in mJy


	#bin every 2 values in the spectra to reduce noise, by averaging the flux and flux_error values in each bin, and taking the mean of the frequency values in each bin as the new frequency value.
	bin_size = bin_factor
	n_bins = len(s2cos0762_freq) // bin_size
	s2cos0762_freq = s2cos0762_freq[:n_bins * bin_size]
	s2cos0762_flux = s2cos0762_flux[:n_bins * bin_size]
	s2cos0762_flux_error = s2cos0762_flux_error[:n_bins * bin_size]
	s2cos0762_freq_binned = np.mean(s2cos0762_freq.reshape(-1, bin_size), axis=1)
	s2cos0762_flux_binned = np.mean(s2cos0762_flux.reshape(-1, bin_size), axis=1)
	s2cos0762_flux_error_binned = np.mean(s2cos0762_flux_error.reshape(-1, bin_size), axis=1)

	print(f"binned spectra: {len(s2cos0762_freq_binned)} points, original spectra: {len(s2cos0762_freq)} points, bin size: {bin_size}")

	#plot the spectra
	alma_spectra_fig, alma_spectra_ax = plt.subplots(figsize=(24, 6))
	alma_spectra_ax.plot(s2cos0762_freq_binned, s2cos0762_flux_binned, label=f'ALMA Spectra for {ALMA_ID} (COSMOS-Web ID: {COSMOS_WEB_ID})', color='blue')
	# alma_spectra_ax.fill_between(s2cos0762_freq_binned, s2cos0762_flux_binned - s2cos0762_flux_error_binned, s2cos0762_flux_binned + s2cos0762_flux_error_binned, color='blue', alpha=0.3, label='Flux Error')
	alma_spectra_ax.set_xlabel('Observed Frequency (GHz)')
	alma_spectra_ax.set_ylabel('Flux Density (mJy)')
	alma_spectra_ax.set_title(f'ALMA Spectra for {ALMA_ID} / COSMOS-Web ID {COSMOS_WEB_ID}')
	alma_spectra_ax.grid(True, which='both', linestyle='--', linewidth=0.5)
	alma_spectra_ax.minorticks_on()


	#try to find possible lines and mask out the lines from the spectra, by finding points where the flux is greater than 3 times the flux error, and masking those points out from the spectra before fitting the power law.
	signal_mask = s2cos0762_flux_binned > 3 * s2cos0762_flux_error_binned



	#include power law fit to the spectra, using np.polyfit on the log-log of the data, and plot the fit as a dashed line on the same plot. The power law is of the form S = A * nu^alpha, where S is the flux density, nu is the frequency, A is a normalization constant, and alpha is the spectral index. The fit should be done in log-log space, so we take the log of both frequency and flux density, and then use np.polyfit to fit a linear model to the log-log data. The slope of the fitted line will give us the spectral index alpha.
	#filter out any non-positive flux values before taking the log
	positive_flux_mask = s2cos0762_flux_binned > 0
	signal_mask = ~signal_mask & positive_flux_mask

	# Use the BUNDED (binned) arrays for masking/fit to avoid dimension mismatch
	if np.count_nonzero(signal_mask) > 1:
		log_freq = np.log10(s2cos0762_freq_binned[signal_mask])
		log_flux = np.log10(s2cos0762_flux_binned[signal_mask])
		#fit a linear model to the log-log data
		coeffs = np.polyfit(log_freq, log_flux, 1)
		alpha = coeffs[0]  # spectral index
		A = 10**coeffs[1]  # normalization constant
		#generate fitted flux values using the power law model evaluated on the original (unbinned) frequency grid for plotting
		fitted_flux = A * s2cos0762_freq**alpha
		#plot the fitted power law on the same plot as a dashed line
		# alma_spectra_ax.plot(s2cos0762_freq, fitted_flux, label=f'Power Law Fit: S = {A:.3e} * nu^{alpha:.3f}', color='red', linestyle='--')
		# alma_spectra_ax.legend()
	else:
		print("Not enough positive signal points in binned spectrum to perform power-law fit.")
		coeffs = np.array([np.nan, np.nan])
		alpha = np.nan
		A = np.nan
		fitted_flux = np.full_like(s2cos0762_freq, np.nan)

	alma_spectra_ax.plot(s2cos0762_freq, fitted_flux, label=f'Power Law Fit: S = {A:.3e} * nu^{alpha:.3f}', color='red', linestyle='--')
	alma_spectra_ax.legend()
	# include in top left the ALMA ID and cosmos Web ID and 
	
	z_redshift = np.inf

	alma_spectra_ax.text(0.02, 0.98, f'ALMA ID: {ALMA_ID}\nCOSMOS-Web ID: {COSMOS_WEB_ID}\n', transform=alma_spectra_ax.transAxes, verticalalignment='top')#, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
	# alma_spectra_ax.set_ylim(bottom=-0.75, top=1.25)

	if ylim is not None:
		alma_spectra_ax.set_ylim(ylim)
	else:
		# #set ylim to be 1.5 times the max flux value, and -0.5 times the min flux value, to give some space above and below the data.
		bottom = 1.3 * np.nanmin(s2cos0762_flux_binned) if np.nanmin(s2cos0762_flux_binned) < 0 else 0
		top = 1.8 * np.nanmax(s2cos0762_flux_binned)
		alma_spectra_ax.set_ylim(bottom=bottom, top=top)
  
	if xlim is not None:
		alma_spectra_ax.set_xlim(xlim)
  
	if show_plot:
		plt.show()
 
	return alma_spectra_fig, alma_spectra_ax, coeffs, alpha, A, fitted_flux, s2cos0762_freq, s2cos0762_flux, s2cos0762_flux_error, signal_mask, bin_size


#try and find the redshift of the object from the ALMA spectra, by looking for the CO(5-4) line at rest frequency 576.268 GHz, and calculating the redshift as z = (rest_freq / obs_freq) - 1, where obs_freq is the observed frequency of the line in GHz. The CO(5-4) line should appear as a peak in the spectra, so we can try and fit a double Gaussian to the spectra to find the peak frequency. We can use scipy.optimize.curve_fit to fit a double Gaussian model to the spectra, and then find the peak frequency from the fitted parameters. The double Gaussian model is of the form:
# S(nu) = A1 * exp(-((nu - mu1)^2) / (2 * sigma1^2)) + A2 * exp(-((nu - mu2)^2) / (2 * sigma2^2)) + C
# where A1, A2 are the amplitudes of the two Gaussians, mu1, mu2 are the means (peak frequencies) of the two Gaussians, sigma1, sigma2 are the standard deviations of the two Gaussians, and C is a constant offset. We can use the fitted parameters to find the peak frequency of the CO(5-4) line, and then calculate the redshift as z = (rest_freq / obs_freq) - 1. We can also plot the fitted double Gaussian model on top of the spectra to visualize the fit.

def double_gaussian(nu, A1, mu1, sigma1, A2, mu2, sigma2, C):
    return (A1 * np.exp(-((nu - mu1) ** 2) / (2 * sigma1 ** 2)) +
            A2 * np.exp(-((nu - mu2) ** 2) / (2 * sigma2 ** 2)) + C)
    
# try to fit the double Gaussian model to the spectra using scipy.optimize.curve_fit, with initial guesses for the parameters based on the data. We can use the np.max and np.argmax functions to find the peak flux and its corresponding frequency, and use that as an initial guess for mu1. We can also use the standard deviation of the flux values as an initial guess for sigma1. For A1, we can use the peak flux value as an initial guess. For A2, mu2, sigma2, and C, we can use 0 as initial guesses. We can also set bounds for the parameters to ensure they are physically reasonable.
from scipy.optimize import curve_fit
from IPython.display import display

def plot_alma_spectra_with_co_fit(S2COSP0762_ALMA_SPECTRA_file = None, spectraData = None, ALMA_ID = None, COSMOS_WEB_ID = None, show_plot=True, bin_factor=2):
    co_fit_fig, co_fit_ax, coeffs, alpha, A, fitted_flux, s2cos0762_freq, s2cos0762_flux, s2cos0762_flux_error, signal_mask, bin_size = plot_alma_spectra(S2COSP0762_ALMA_SPECTRA_file=S2COSP0762_ALMA_SPECTRA_file, spectraData=spectraData, ALMA_ID=ALMA_ID, COSMOS_WEB_ID=COSMOS_WEB_ID, show_plot=False, bin_factor=bin_factor)

    positive_flux_mask = s2cos0762_flux > 0
    try:
        popt, pcov = curve_fit(double_gaussian, s2cos0762_freq[positive_flux_mask], s2cos0762_flux[positive_flux_mask],
                            p0=[np.max(s2cos0762_flux), s2cos0762_freq[np.argmax(s2cos0762_flux)], np.std(s2cos0762_flux), 0, 0, 0, 0],
                            bounds=([0, 0, 0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))
        A1_fit, mu1_fit, sigma1_fit, A2_fit, mu2_fit, sigma2_fit, C_fit = popt
        print(f"Fitted parameters: A1={A1_fit}, mu1={mu1_fit}, sigma1={sigma1_fit}, A2={A2_fit}, mu2={mu2_fit}, sigma2={sigma2_fit}, C={C_fit}")
        
        print(f"Estimated peak frequency of CO(5-4) line: {mu1_fit:.4f} GHz")
        # Calculate the redshift from the fitted peak frequency of the CO(5-4) line
        rest_freq_CO54 = 576.268  # GHz
        z_redshift = (rest_freq_CO54 / mu1_fit) - 1
        print(f"Estimated redshift from CO(5-4) line: z = {z_redshift:.4f}")
        # Plot the fitted double Gaussian model on top of the spectra to visualize the fit
        
        # fitted_flux_double_gaussian = double_gaussian(s2cos0762_freq, *popt)
        # co_fit_ax.plot(s2cos0762_freq, fitted_flux_double_gaussian, label='Fitted Double Gaussian', color='green', linestyle='--')
        #instead of plotting the fitted gaussian, plot a shaded region of the spectra above the power law fit, to show where the line is detected. Use the fitted mu1_fit and sigma1_fit to define the region of the line, and shade the region where the flux is above the power law fit.
        power_law_fit_flux = fitted_flux #A * s2cos0762_freq**alpha
        s2cos0762_freq_binned = np.mean(s2cos0762_freq.reshape(-1, bin_size), axis=1)
        s2cos0762_flux_binned = np.mean(s2cos0762_flux.reshape(-1, bin_size), axis=1)
        power_law_fit_flux_binned = np.mean(power_law_fit_flux.reshape(-1, bin_size), axis=1)
        line_region_mask = (s2cos0762_freq_binned > (mu1_fit - 3 * sigma1_fit)) & (s2cos0762_freq_binned < (mu1_fit + 3 * sigma1_fit)) & (s2cos0762_flux_binned > power_law_fit_flux_binned)
        co_fit_ax.fill_between(s2cos0762_freq_binned[line_region_mask], s2cos0762_flux_binned[line_region_mask], power_law_fit_flux_binned[line_region_mask], color='green', alpha=0.3, label='Detected CO(5-4) Line Region')
        
        #add the double gaussian fit to the plot
        near = np.abs(s2cos0762_freq - mu1_fit) < 3 * sigma1_fit
        co_fit_ax.plot(s2cos0762_freq[near], double_gaussian(s2cos0762_freq, *popt)[near], label='Fitted Double Gaussian', color='orange', linestyle='--')
        
        #add redshift text to the plot
        co_fit_ax.text(0.02, 0.90, f'Estimated redshift from CO(5-4) line: z = {z_redshift:.4f}', transform=co_fit_ax.transAxes, verticalalignment='top')#, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        #add annotation to the plot showing the fitted peak frequency of the CO(5-4) line
        co_fit_ax.annotate(f'CO(5-4) Peak: {mu1_fit:.4f} GHz', xy=(mu1_fit, double_gaussian(mu1_fit, *popt)), xytext=(mu1_fit + 0.5, double_gaussian(mu1_fit, *popt) + 0.5), arrowprops=dict(arrowstyle='-', color='red', lw=1.5), fontsize=10, color='black')
        
        if show_plot:
            plt.show()
        # try:
        #     co_fit_fig.canvas.draw_idle()
        # except Exception:
        #     pass
        # display(co_fit_fig)
    except Exception as e:
        print(f"Could not fit double Gaussian to the spectra: {e}")
        z_redshift = np.inf
        
    return co_fit_fig, co_fit_ax, z_redshift, mu1_fit, sigma1_fit, coeffs, alpha, A, fitted_flux, s2cos0762_freq, s2cos0762_flux, s2cos0762_flux_error, signal_mask, bin_size



## Look for the H2O(1_{10}-1_{01}) line at rest 556.936 GHz. It is often a
## P-Cygni profile (emission on one side of systemic, absorption on the other),
## so we fit TWO Gaussians with opposite-sign amplitudes -- one emission
## (A>=0), one absorption (A<=0) -- and take the systemic (zero-crossing)
## frequency for the redshift.
rest_freq_H2O = 556.936  # GHz
rest_freq_CO54 = 576.2679305  # GHz, CO(5-4) rest


def plot_alma_spectra_with_h2o_fit(S2COSP0762_ALMA_SPECTRA_file = None, spectraData = None, ALMA_ID = None, COSMOS_WEB_ID = None, show_plot=True, bin_factor=2):
    h2o_co_fit_fig, h2o_co_fit_ax, co_z_redshift, mu1_fit, sigma1_fit, coeffs_h2o, alpha_h2o, A_h2o, fitted_flux_h2o, s2cos0762_freq, s2cos0762_flux, s2cos0762_flux_error, signal_mask_h2o, bin_size_h2o = plot_alma_spectra_with_co_fit(S2COSP0762_ALMA_SPECTRA_file, spectraData, ALMA_ID, COSMOS_WEB_ID, show_plot=False, bin_factor=bin_factor)

    # Initialise so the except/return path can never NameError.
    z_redshift_h2o = np.inf
    mu_h2o = sig_h2o = np.nan

    try:
        # Emission + absorption model: A_e >= 0, A_a <= 0 (enforced via bounds).
        def emit_abs(x, A_e, mu_e, sig_e, A_a, mu_a, sig_a, C):
            return (A_e * np.exp(-(x - mu_e)**2 / (2 * sig_e**2))
                  + A_a * np.exp(-(x - mu_a)**2 / (2 * sig_a**2)) + C)

        # Redshift from the CO fit -> where we expect H2O.
        z_CO = rest_freq_CO54 / mu1_fit - 1
        mu_H2O_expected = rest_freq_H2O / (1 + z_CO)
        print(f"z_CO = {z_CO:.4f}  ->  expect H2O at {mu_H2O_expected:.4f} GHz")

        # Window around the prediction; widen for a P-Cygni (both lobes must fit),
        # and exclude the CO core.
        half_win = 0.5  # GHz -- tune to cover emission AND absorption lobes
        win = (np.abs(s2cos0762_freq - mu_H2O_expected) < half_win) & \
            (np.abs(s2cos0762_freq - mu1_fit) > 6 * sigma1_fit)

        x, y = s2cos0762_freq[win], s2cos0762_flux[win]
        yerr = s2cos0762_flux_error[win]
        if len(x) < 8:
            raise RuntimeError(f"only {len(x)} channels in the H2O window; widen half_win or check masking")

        # Initial guesses: emission from the max, absorption from the min.
        C0 = np.median(y)
        imax, imin = np.argmax(y), np.argmin(y)
        p0 = [max(y.max() - C0, 1e-6), x[imax], sigma1_fit,
              min(y.min() - C0, -1e-6), x[imin], sigma1_fit, C0]
        bounds = ([0,      x.min(), 0,            -np.inf, x.min(), 0,            -np.inf],
                  [np.inf, x.max(), 5*sigma1_fit,  0,      x.max(), 5*sigma1_fit,  np.inf])

        popt_h2o, pcov_h2o = curve_fit(emit_abs, x, y, p0=p0, bounds=bounds,
                                       sigma=yerr, absolute_sigma=True)
        A_e, mu_e, sig_e, A_a, mu_a, sig_a, C_h2o = popt_h2o
        #  tighten sig_e/sig_a upper bounds or seed mu_e/mu_a from the expected blue/red offsets rather than from argmax/argmin

        # Systemic frequency = zero-crossing of (model - continuum) between the
        # two component centres; fall back to their midpoint if no crossing.
        lo, hi = sorted([mu_e, mu_a])
        xg = np.linspace(lo, hi, 1001)
        resid = emit_abs(xg, *popt_h2o) - C_h2o
        crossings = np.where(np.diff(np.sign(resid)))[0]
        mu_systemic = xg[crossings[0]] if crossings.size else 0.5 * (mu_e + mu_a)

        mu_h2o, sig_h2o = mu_systemic, 0.5 * (sig_e + sig_a)
        z_redshift_h2o = rest_freq_H2O / mu_systemic - 1
        c_kms = 299792.458
        dv = c_kms * (mu_e - mu_a) / mu_systemic  # emission-absorption velocity split

        print(f"Fitted H2O: emission A={A_e:.3e} @ {mu_e:.4f} GHz (sig {sig_e:.4f}); "
              f"absorption A={A_a:.3e} @ {mu_a:.4f} GHz (sig {sig_a:.4f}); C={C_h2o:.3e}")
        print(f"Systemic H2O frequency = {mu_systemic:.4f} GHz  ->  z = {z_redshift_h2o:.4f}")
        print(f"Emission-absorption velocity split = {dv:.1f} km/s (P-Cygni signature)")

        # Overplot the fitted profile and shade emission (above C) / absorption
        # (below C) relative to the fitted continuum.
        model_full = emit_abs(s2cos0762_freq, *popt_h2o)
        near = np.abs(s2cos0762_freq - mu_systemic) < 3 * (sig_e + sig_a)
        h2o_co_fit_ax.plot(s2cos0762_freq[near], model_full[near], color='orange',
                           ls='--', lw=1.5, label='H2O P-Cygni fit')
        # Shade emission / absorption on the SAME binned grid that is plotted,
        # so `where` matches the x/y length. C_h2o is the scalar fitted
        # continuum -- fill_between broadcasts it, so it must NOT be reshaped.
        nb_bins = len(s2cos0762_flux) // bin_factor
        freq_b = np.mean(s2cos0762_freq[:nb_bins * bin_factor].reshape(-1, bin_factor), axis=1)
        flux_b = np.mean(s2cos0762_flux[:nb_bins * bin_factor].reshape(-1, bin_factor), axis=1)

        near_b = np.abs(freq_b - mu_systemic) < 3 * (sig_e + sig_a)
        above = near_b & (flux_b > C_h2o)
        below = near_b & (flux_b < C_h2o)

        h2o_co_fit_ax.fill_between(freq_b, C_h2o, flux_b, where=above,
                                   color='orange', alpha=0.3, label='H2O emission')
        h2o_co_fit_ax.fill_between(freq_b, C_h2o, flux_b, where=below,
                                   color='purple', alpha=0.3, label='H2O absorption')

        h2o_co_fit_ax.annotate(f'H2O $(1_{{10}} - 1_{{01}})$: {mu_systemic:.4f} GHz',
                               xy=(mu_systemic, C_h2o),
                               xytext=(mu_systemic + 0.001, C_h2o + max(A_e, 1)),
                               arrowprops=dict(arrowstyle='-', color='red', lw=1.5),
                               fontsize=10, color='black')
        h2o_co_fit_ax.text(0.02, 0.85, f'Estimated redshift from H2O line: z = {z_redshift_h2o:.4f}',
                           transform=h2o_co_fit_ax.transAxes, verticalalignment='top')
        h2o_co_fit_ax.text(0.02, 0.80, f'Averaged redshift from CO and H2O: z = {(z_redshift_h2o + z_CO) / 2:.4f}',
                           transform=h2o_co_fit_ax.transAxes, verticalalignment='top')
        h2o_co_fit_ax.legend()
        if show_plot:
            plt.show()

    except Exception as e:
        print(f"Could not fit H2O P-Cygni profile to the spectra: {e}")
        z_redshift_h2o = np.inf

    return h2o_co_fit_fig, h2o_co_fit_ax, z_redshift_h2o, mu_h2o, sig_h2o, coeffs_h2o, alpha_h2o, A_h2o, fitted_flux_h2o, s2cos0762_freq, s2cos0762_flux, s2cos0762_flux_error, signal_mask_h2o, bin_size_h2o
