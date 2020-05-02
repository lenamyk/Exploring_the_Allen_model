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

# Set number of flashes and length of each flash:
+ nr_of_flashes = 20
+ seg_length = int((numberFramesNeeded-1)/(nr_of_flashes*2))



```

