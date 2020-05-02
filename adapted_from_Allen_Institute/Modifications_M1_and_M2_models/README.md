© 2019 Allen Institute for Brain Science. Models of the Mouse Primary Visual Cortex. 
Files are adapted from: https://www.dropbox.com/sh/w5u31m3hq6u2x5m/AACpYpeWnm6s_qJDpmgrYgP7a?dl=0.

This folder contains the files which were modified from the Allen LGN model
to create the M1 and M2 models for the thesis "Exploring a Model of the Mouse Primary Visual Cortex".


The modifications made to each file are presented below.

**flashing_spots.py:**

```diff
# Change spatial resolution to 1 pixel per degree:
- physical_spacing = 1. / (float(cpd) * 10)
+ physical_spacing = 1. / (float(0.1) * 10)  
```
```diff
# Set number of flashes and length of each flash:
+ nr_of_flashes = 20
+ seg_length = int((numberFramesNeeded-1)/(nr_of_flashes*2))
```
```diff
# Creating segments of gray and white full-field input:
+ gray_part = np.zeros((seg_length, self.row_size, self.col_size))
+ circle_part = np.ones((seg_length, self.row_size, self.col_size))
+ gray_partb = np.zeros((1, self.row_size, self.col_size))
```
```diff
# Apply mask to get spot of specified size:
+ circle = (xx - self.col_size / 2) ** 2 + (yy - self.row_size / 2) ** 2       
+ mask = (circle <= radius**2)
+ circle_part[~mask] = 0
```
```diff
# Create movie sequence:
+ segment = np.concatenate((circle_part, gray_part), axis=0)
+ data = np.tile(segment,reps =[nr_of_flashes,1,1])
+ data = np.concatenate((data, gray_partb), axis=0)
```

**moving_grating.py **
```diff
# Apply mask to get circular patches from the full-field grating:
+ circle = (xx - self.col_size / 2) ** 2 + (yy - self.row_size / 2) ** 2
+ mask = (circle <= radius**2)
+ data[~mask] = 0
```

**spatialfilter_surr.py**
```diff
+ scal_sig = 2.45
+ cent_amp = 1.73
+ scal_amp = 1*cent_amp   
+ filter_cen = ndimage.gaussian_filter(on_filter_spatial, (scaled_sigma_x, scaled_sigma_y), mode='nearest', cval=0)
+ filter_sur = ndimage.gaussian_filter(on_filter_spatial, (scal_sig*scaled_sigma_x, scal_sig*scaled_sigma_y), mode='nearest', cval=0)
+ on_filter_spatial = cent_amp*filter_cen - scal_amp*filter_sur
```
