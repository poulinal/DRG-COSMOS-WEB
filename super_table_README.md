# Overview
Catalogs from: JWST: COSMOSWeb_mastercatalog_v1.1.fits, super-deblended: COSMOS_deblended_FIR_201821_photo_phys_2023June.fits, MIGHTEE: MIGHTEE_Continuum_DR1_COSMOS_5p2arcsec_I_v1.1.fits, ALMA: champs_blind_catalog.fits & champs_prior_catalog.fits

This is a "super-table" of the detections chosen from JWST COSMOS-Web DR1 Catalog (M. Shuntov+ 2025) which passed the criteria: (flux_model SNR < 3 for f115w & f150w & f227w) & (flux_model SNR > 5 for f44w and f770w). 18 detected from DR1 catalog, with visual inspection: 11 detected sample. 

Then cross-matched with Super-Deblended (S. Jin+ 2018), matching the COSMOS-Web ra,dec with Super-Deblended's RA,Dec for each object (match if within 1 arcsec tolerance). Result: 2 matching refs.

Cross-matched with MIGHTEE (C. L. Hale+ 2024), getting the pixel intensities of COSMOS-Web's ra,dec in the map. Result: 11 matching refs.

Finally cross-matched with ALMA in both CHAMPS_blind and CHAMPS_prior (A. Faisst ...), again matching the COSMOS-Web ra,dec with CHAMPS's RA,Dec for each object for both catalogs respectively(match if within 1 arcsec tolerance). Result: 2 matching refs in CHAMPS_prior and 2 matching refs in CHAMPS_blind. Note interestingly enough, 1 refs in each are not the same.

Note if any of the objects dont have a cross-match in a specific catalog, those fields for that catalog will be left np.nan.

See bottom of document for quick reference.




# Table Details
=====================================================================================================================================================================

ID -- id with respect to the detected sample (0-10 of the detected sample)

COSMOS-Web_id
COSMOS-Web_ra
COSMOS-Web_dec
    rest of COSMOS-Web columns...

has_cross_match_in_super_deblended -- 0 if no match, 1 if match
super_deblended_ID -- id from the super_blended catalog if there is a match, else np.nan
super_deblended_RA -- RA from the super_blended catalog if there is a match, else np.nan
super_deblended_Dec -- Dec from the super_blended catalog if there is a match, else np.nan
    rest of super_deblended_catalog's columns...

has_cross_match_in_mightee -- 0 if no match, 1 if match
mightee_intensity -- intensity of the pixel at the RA, Dec of MIGHTEE map if there is a match, else np.nan
mightee_avg -- average intensity of the pixels around a ~3.3" arcsec radius (3 pixel radius) around the RA, Dec point
mightee_peak -- peak intensity of the pixels within a ~3.3" arcsec radius (3 pixel radius) around the RA, Dec point

has_cross_match_in_champs_prior -- 0 if no match, 1 if match
champs_prior_cosmos_id -- id from champs prior catalog if there is a match, else np.nan
champs_prior_ra -- ra from champs prior catalog if there is a match, else np.nan
champs_prior_dec -- dec from champs prior catalog if there is a match, else np.nan
    rest of champ_prior_catalog's columns

has_cross_match_in_champs_blind -- 0 if no match, 1 if match
champs_blind_cosmos_id -- id from champs blind catalog if there is a match, else np.nan
champs_blind_ra -- ra from champs blind catalog if there is a match, else np.nan
champs_blind_dec -- dec from champs prior catalog if there is a match, else np.nan
    rest of champs_blind_catalog's columns







# For quick reference
=====================================================================================================================================================================
COSMOS-Web ID	has_cross_match_in_super_deblended	super_deblended_ID	has_cross_match_in_mightee	mightee_intensity_point	mightee_intensity_avg	mightee_intensity_peak	has_cross_match_in_champs_prior	champs_prior_cosmos_id	has_cross_match_in_champs_blind	champs_blind_id
9404	0	nan	1	-2.57E-06	-1.68E-06	-3.63E-07	0	nan	0	nan
171453	1	836111	1	1.07E-05	9.18E-06	1.33E-05	1	171453	1	164
177683	0	nan	1	1.07E-05	9.25E-06	1.25E-05	0	nan	1	229
184688	0	nan	1	-6.71E-07	-2.37E-07	3.50E-06	0	nan	0	nan
413568	0	nan	1	-2.27E-06	-1.74E-06	2.03E-07	0	nan	0	nan
593259	0	nan	1	4.22E-06	4.17E-06	8.57E-06	0	nan	0	nan
612754	0	nan	1	2.61E-06	1.75E-06	4.16E-06	0	nan	0	nan
710864	0	nan	1	2.72E-06	2.44E-06	4.78E-06	0	nan	0	nan
718984	1	39233	1	1.27E-05	1.14E-05	1.30E-05	1	718983	0	nan
722439	0	nan	1	3.35E-06	2.66E-06	3.45E-06	0	nan	0	nan
781174	0	nan	1	4.18E-06	3.21E-06	4.70E-06	0	nan	0	nan