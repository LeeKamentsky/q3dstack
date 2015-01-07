Q3DStack - a program to quantify a 3D stack

This program takes a stack of 3-channel images, segments the red and blue
channel and then computes statistics for any object in the red channel
chosen by the user.

Preparation:
The images should have the nuclear stain in the third channel and the organelle
stain in the first channel. The user starts the program which shows a
file dialog from which they can select the images of the stack (the user can
select multiple images). Q3DStack then organizes these into Z planes by
alphabetical order and displays the color image of the stack.

Segmentation:
Initially, Q3DStack computes a threshold for each channel by taking the average
of the thresholds suggested by the Otsu and MCT methods. It displays these
as well as the standard deviation of the Gaussian used to smooth the image
before segmentation. Separate standard deviations are used for the X/Y and the
Z direction to count for any anisotropy of voxel size (the voxel's Z extent
is often larger than the X/Y extent). The user sets each of these and checks
the checkbox to the right of the channel to perform the segmentation. These
numbers can be adjusted, either by hitting the up and down arrows on each of
the parameter input boxes or by typing into the boxes, then hitting "Enter".

Measurements:
To measure quantities for an organelle, click on the organelle's segmentation.
The following will be displayed:
Green Normalization - Pixel values vary between 0 and 255. Q3DStack attempts to
                      take this into account by dividing pixel values by
                      the value at the 99.99999th percentile of intensity,
                      effectively a robust maximum. This is displayed as the
                      green normalization.
                      
Total green intensity - This is the sum of all pixel values for the green
                        channel within the chosen organelle, after normalization.
                        
Mean green intensity - This is the average intensity for the green channel
                       for the chosen organelle, after normalization.
                       The mean green intensity is calculated both overall
                       and per Z plane.

Pearson's correlation coefficient - For the pixels within the chosen organelle,
                      the red and green channels are normalized by subtracting
                      the mean values for each, then the normalized intensities
                      of the green and red channel are multiplied together,
                      then summed, then divided by the standard deviations of
                      each channel. Correlations are between 1 (correlated)
                      and -1 (anticorrelated).
                      
Manders' correlation coefficient - for the pixels within the chosen organelle,
                      the intensity of the red channel is multiplied by
                      the intensity of the green channel and then summed,
                      then divided by the square root of the sum of the
                      squares of the red channel and green channel.
                      
Anisotropy - The distance from each pixel in the chosen organelle to the
             nearest pixel in some nucleus is calculated. This distance
             is normalized by subtracting the mean distance for the organelle
             and by dividing by the standard deviation of the distance.
             The anisotropy is the mean value of this distance times the
             green intensity at each pixel. A negative anisotropy indicates
             that the green intensity is stronger for pixels nearer the nucleus
             whereas a positive anisotropy indicates that the intensity
             is stronger for pixels at a greater distance from the nucleus.