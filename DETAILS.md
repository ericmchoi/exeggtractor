# Image Processing Steps

## Overview

![Overview](https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/overview.png?raw=true)

## Criteria

Since photos can vary infinitely in angle, lighting, background, and more, it is useful to define a set of criteria for the kinds of images that a computer vision program is best equipped to handle. This becomes a guideline for users to follow so that they can provide an image that will produce the best results. The criteria for *Exeggtractor* are as follows:

 1. The entire screen is visible in the photo.
 2. The screen's width takes up at least 50% the width of the photo.
 3. The text in the photo is readable by a human.

## Image Type Recognition

The first thing *Exeggtractor* does is identify whether an image is a clean screenshot exported from a Nintendo Switch console, or a photo taken of the screen. It performs the following checks:

 1. Resolution is one of the two known screenshot sizes (1920x1080 or 1280x720)
 2. The two top pixels are a blue-green color and the two bottom pixels are black

If an image passes both tests, it is recognized as a screenshot and skips the following step.

## Screen Finding

If the image is a photo, *Exeggtractor* attempts to find the game screen within the image and crop it out. To find the screen in the photo, the program uses color quantization to group similar colors together and connected component analysis to determine the most "screen-like" area in the image.

Color quantization is done using [k-means clustering](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html) using a k value of 3. The aim of this quantization is to divide the image into three distinct parts: the screen, the console or screen edge, and the background. Since we know that the screen will be mostly a blue-green color that takes up a large area of the image, one of the 3 clusters determined should include this blue-green area of the image.

Next, Exeggtractor takes the binary mask of each cluster, and uses connected components analysis to divide the mask into connected parts. Each part is then filtered out if it fails any of the criteria listed in the beginning of this page, or if it's approximated shape is not a quadrilateral. For the example below, this results in two potential candidates for the screen.

Each of the potential candidates is compared to it's bounding rectangular box as a measure of skew, and the candidate with the least skew is chosen as the final screen area.

| Name | Image |
| -- | -- |
| Original Image |<img alt="original image" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/00-source.jpg?raw=true" height="320" />  |
| Color Quantized | <img alt="color quantized" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/01-source-quantized.jpg?raw=true" height="320" /> |
| Color Mask (of blue) | <img alt="color mask" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/03-color-mask-1.jpg?raw=true" height="320" /> |
| Connected Components (Bounding boxes in red) | <img alt="connected components" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/05-connected-components.jpg?raw=true" height="320" /> |
| Final Result (Screen is outlined red) | <img alt="final result" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/06-source-with-screen.jpg?raw=true" height="320" /> |

### Notes

A more conventional approach to this sort of computer vision problem is to use [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) to get the screen edges in the image. However, this technique is particularly susceptible to noise, and may yield widely varying results depending on the thresholds chosen for the edge detection algorithm. It is very difficult to determine values for these thresholds that would work well with every image, so this approach was avoided for this program.

## Screen Crop and Perspective Fix

Once the contour of the screen is found, Exeggtractor performs a perspective transform to straighten and crop out the screen into a separate image. At this point, the image should be very similar to a screenshot image.

<img alt="final result" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/07-screen.jpg?raw=true" height="320" />

### Notes

Since this image is so similar to a screenshot, we could take specific pixel regions of the image and extract the text from there. However, in order to handle more cases where the image maybe a few pixels misaligned, *Exeggtractor* uses a more involved process to find text in the image.

## White Region Finding

The next step of the process is to look for the rectangular white regions of the screen that contain some of the text we want to extract. Since they are large, distinct areas of the screen arranged in a grid, it is easier to identify and find them rather than look for text right away.

First, *Exeggtractor* performs a color quantization of the screen image. It also uses a k value of 3, this time to separate the image into blue-green, white, and black/gray areas. After the image has been quantized, we take the [delta-E](https://colour.readthedocs.io/en/latest/generated/colour.delta_E.html) of each color to white. Delta-E is a measure of color difference as perceptible to the human eye and helps identify the correct areas in scenarios where lightning and other photo conditions may affect colors in the image. The program takes the color with the smallest delta-E value and creates a mask from it.

Next, a [morphological opening](https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html) is performed twice in vertical and horizontal directions to clean the mask of any small artifacts that may disrupt edge detection. A Sobel filter is then used to get the edges in the mask. Using [Hough Line transforms](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html), we look for the longest horizontal and vertical edges, which should match up to the edges of the regions we want.

| Name | Image |
| -- | -- |
| Color Quantization | <img alt="screen quantized" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/08-screen-quantized.jpg?raw=true" height="320" /> |
| Mask of White Pixels | <img alt="Mask of White Pixels" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/09-white-mask.jpg?raw=true" height="320" /> |
| Opening (Vertical Direction) | <img alt="Vertical Opening" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/10-opening-x.jpg?raw=true" height="320" />  |
| Sobel Filter (Vertical) | <img alt="Sobel Filter X" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/11-sobel-x.jpg?raw=true" height="320" />  |
| Hough Lines (Vertical) | <img alt="Hough Lines Vertical" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/12-vertical-hough-lines.jpg?raw=true" height="320" />  |
| Opening (Horizontal) | <img alt="Horizontal Opening" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/13-opening-y.jpg?raw=true" height="320" />  |
| Sobel Filter (Horizontal) | <img alt="Sobel Filter Y" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/14-sobel-y.jpg?raw=true" height="320" />  |
| Hough Lines (Horizontal) | <img alt="Hough Lines Horizontal" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/15-horizontal-hough-lines.jpg?raw=true" height="320" />  |
| Final Result | <img alt="Final Result" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/16-merged-lines.jpg?raw=true" height="320" />  |

### Notes

There is a more straightforward approach that involves taking the mask of the white regions and then using [Contour Finding and Approximation](https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html) to look for rectangular areas in the image. But when this approach was used, bright, near-white screen animations and photo artifacts interfered with finding rectangular polygons with straight horizontal and vertical edges. Morphological transforms like the opening used above could be used to clean up the image, but would distort the image enough to interfere with contour finding.

## Text Extraction

Using the intersection of the lines found above, and with a little geometric extrapolation, each text line can be isolated and cropped.

<img alt="enter image description here" src="https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/72-text-regions.jpg?raw=true" height="320" />

These cropped lines are then preprocessed and given to [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) to read text. The extracted text is then matched against a list of known values to produce the final result. If it does not match well enough to any value, the image is preprocessed again with slightly different parameters to try and improve the result of the OCR.

The preprocess steps are as follows:

 1. Convert image into grayscale
 2. Use K-means clustering to threshold the image into a binary black and white image
 3. Use connected components analysis to remove any pixels connected to the edge of the image

Since the text we want is a single color against a single color background, we start with a k value of 2, and increase it with each retry for better resolution. Below is an example of the same text line, preprocessed with slightly different k values to try and get different OCR results.

![Text Example 1](https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/63-6-species-1.jpg?raw=true)
![Text Example 2](https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/64-6-species-2.jpg?raw=true)
![Text Example 3](https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/65-6-species-3.jpg?raw=true)
