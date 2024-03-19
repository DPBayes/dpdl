# Better from scratch training

## Motivation
The wishlist contains (1) finding out what are important characteristics of pre-training images and (2) make pre-training (e.g., using simulated data) better for DP fine-tuning because we know what matters thanks to (1)

## Methodology

### Model
Probably want to use FiLM to enable near full adaption possibility for fine-tuning. (For pre-training obviously tuning all parameters.)

### Datasets (pre-training)

- ImageNet-1k (in different variations)

### Datasets (fine-tuning)

- [VTAB](https://arxiv.org/pdf/1910.04867.pdf) (easy to setup and standard benchmark, 19 datasets)
    - from huggingface (2): cifar100 svhn
    - from Tensorflow datasets (easy to download, 8): dmlab dtd eurosat oxford_flowers102 oxford_iiit_pet patch_camelyon resisc45 sun397
    - from Tensorflow datasets (needs additional work):
        - different labels based on task (6 = 3x2): dsprites, clevr, smallnorb
        - manual download (1): diabetic_retinopathy_detection/btgraham-300
        - error (FileFormat.TFRECORD. Got FileFormat.ARRAY_RECORD, 2): caltech101 kitti
- Medical datasets
    - Borja Balle used [CheXpert](https://arxiv.org/pdf/1901.07031.pdf) which is [not on huggingface datasets](https://github.com/huggingface/datasets/issues/6382)
    - Borja Balle also used [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) which is not on huggingface datasets
    - Huggingface: alkzar90/NIH-Chest-X-ray-dataset [paper](https://arxiv.org/pdf/1705.02315.pdf)
    - Huggingface: marmal88/skin_cancer [paper](https://www.nature.com/articles/sdata2018161.pdf)
- Other datasets
    - Huggingface: HuggingFaceM4/Stanford-Cars [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yang_A_Large-Scale_Car_2015_CVPR_paper.pdf)

### List of Transformations

Table consists of transformations available in the [Albumentations](https://github.com/albumentations-team/albumentations?tab=readme-ov-file) library and ImageMagick operations.

| Transformation Name            | Description                                                                   | Source |
|--------------------------------|-------------------------------------------------------------------------------|--------|
| AdvancedBlur                   | Blur the image with an advanced algorithm to simulate out-of-focus blur.     | Albumentations |
| Blur                           | Soften the image by reducing detail and noise.                               | Albumentations |
| CLAHE                          | Apply Contrast Limited Adaptive Histogram Equalization to improve contrast.  | Albumentations |
| ChannelDropout                 | Randomly set channels to zero to simulate channel dropout.                   | Albumentations |
| ChannelShuffle                 | Randomly rearrange the channels of the image.                                | Albumentations |
| ColorJitter                    | Randomly change the brightness, contrast, saturation, and hue of the image.  | Albumentations |
| Defocus                        | Simulate defocusing of the image.                                            | Albumentations |
| Downscale                      | Downscale the image to simulate lower resolution capture.                    | Albumentations |
| Emboss                         | Apply an emboss effect to simulate raised edges around objects.              | Albumentations |
| Equalize                       | Apply histogram equalization to spread out the frequent intensity values.    | Albumentations |
| FDA                            | Frequency Domain Analysis for image transformation.                          | Albumentations |
| FancyPCA                       | Apply PCA color augmentation as described in the AlexNet paper.              | Albumentations |
| FromFloat                      | Convert image from float type to standard image format (0-255 range).        | Albumentations |
| GaussNoise                     | Add Gaussian noise to the image to simulate camera noise.                    | Albumentations |
| GaussianBlur                   | Apply Gaussian blur to the image; reduces high frequency noise.              | Albumentations |
| GlassBlur                      | Apply a glass-like distortion effect to the image.                           | Albumentations |
| HistogramMatching              | Match the histogram of the image to that of a target image.                  | Albumentations |
| HueSaturationValue             | Adjust the hue, saturation, and value of the image.                          | Albumentations |
| ISONoise                       | Simulate ISO noise which is common in digital cameras.                       | Albumentations |
| ImageCompression               | Compress the image to simulate the effect of saving as a JPEG.               | Albumentations |
| InvertImg                      | Invert the colors of the image.                                              | Albumentations |
| MedianBlur                     | Apply median blur; can remove salt and pepper noise from images.             | Albumentations |
| MotionBlur                     | Apply motion blur to simulate the effect of movement during capture.         | Albumentations |
| MultiplicativeNoise            | Multiply image pixels by noise generated from a random distribution.         | Albumentations |
| Normalize                      | Normalize the image by scaling pixel values.                                 | Albumentations |
| PixelDistributionAdaptation    | Adapt the pixel value distribution of an image.                              | Albumentations |
| Posterize                      | Reduce the number of bits for each color channel; simulates posterization.   | Albumentations |
| RGBShift                       | Randomly shift values for each channel of the RGB image.                     | Albumentations |
| RandomBrightnessContrast       | Adjust brightness and contrast randomly within a given range.                | Albumentations |
| RandomFog                      | Simulate the effect of fog for atmospheric augmentation.                     | Albumentations |
| RandomGamma                    | Apply gamma correction randomly within a given range.                        | Albumentations |
| RandomGravel                   | Add random gravel-like noise to the image.                                   | Albumentations |
| RandomRain                     | Simulate raindrops on the image for weather augmentation.                    | Albumentations |
| RandomShadow                   | Add random shadows to the image to simulate shadowing effects.               | Albumentations |
| RandomSnow                     | Simulate snowfall on the image for weather augmentation.                     | Albumentations |
| RandomSunFlare                 | Add a random sun flare effect to the image.                                  | Albumentations |
| RandomToneCurve                | Apply random tone curve adjustments to the image.                            | Albumentations |
| RingingOvershoot               | Simulate ringing overshoot artefacts around sharp edges in the image.        | Albumentations |
| Sharpen                        | Sharpen the image by enhancing edges.                                        | Albumentations |
| Solarize                       | Invert all pixel values above a threshold; simulates the solarization effect.| Albumentations |
| Spatter                        | Simulate paint spatter effect.                                               | Albumentations |
| Superpixels                    | Reduce the image to a limited number of color segments (superpixels).        | Albumentations |
| TemplateTransform             | Apply a template transformation.                                             | Albumentations |
| ToFloat                       | Convert image to float representation.                                       | Albumentations |
| ToGray                        | Convert the image to grayscale.                                              | Albumentations |
| ToRGB                         | Convert the image to RGB format.                                             | Albumentations |
| ToSepia                       | Apply a sepia tone to the image.                                             | Albumentations |
| UnsharpMask                   | Enhance edges by subtracting an unsharp (blurred) version of the image.      | Albumentations |
| ZoomBlur                      | Apply a zoom blur effect to simulate motion towards/away from the camera.    | Albumentations |
| Affine                        | Apply affine transformations including scaling, translation, rotation.       | Albumentations |
| BBoxSafeRandomCrop            | Crop randomly without cutting bounding boxes.                                | Albumentations |
| CenterCrop                    | Crop the center part of the image.                                           | Albumentations |
| CoarseDropout                 | Randomly mask square regions of the image.                                   | Albumentations |
| Crop                          | Crop a region from the image.                                                | Albumentations |
| CropAndPad                    | Crop and pad images.                                                         | Albumentations |
| CropNonEmptyMaskIfExists      | Crop around non-empty mask regions if they exist.                            | Albumentations |
| ElasticTransform              | Apply elastic transformations to distort images.                             | Albumentations |
| Flip                          | Flip the image horizontally or vertically.                                   | Albumentations |
| GridDistortion                | Apply a grid distortion effect to the image.                                 | Albumentations |
| GridDropout                   | Randomly drop regions of the grid.                                           | Albumentations |
| HorizontalFlip                | Flip the image horizontally.                                                 | Albumentations |
| Lambda                        | Apply a lambda function for custom transformations.                          | Albumentations |
| LongestMaxSize                | Resize the image so the maximum size is as specified.                        | Albumentations |
| MaskDropout                   | Randomly mask out regions within masks.                                      | Albumentations |
| NoOp                          | Perform no operation (useful as a placeholder).                              | Albumentations |
| OpticalDistortion             | Distort image using optical distortion.                                      | Albumentations |
| PadIfNeeded                   | Pad the image if needed to meet size requirements.                           | Albumentations |
| Perspective                   | Apply perspective transformation.                                            | Albumentations |
| PiecewiseAffine               | Apply piecewise affine transformation to simulate elastic deformations.      | Albumentations |
| PixelDropout                  | Randomly drop pixels to simulate dead pixels/sensor noise.                   | Albumentations |
| RandomCrop                    | Randomly crop the image.                                                     | Albumentations |
| RandomCropFromBorders         | Crop from the borders after scaling.                                         | Albumentations |
| RandomCropNearBBox            | Randomly crop near bounding boxes if present.                                | Albumentations |
| RandomGridShuffle             | Shuffle grid patches of the image randomly.                                  | Albumentations |
| RandomResizedCrop             | Randomly crop and resize the image.                                          | Albumentations |
| RandomRotate90                | Rotate the image randomly by 0, 90, 180, or 270 degrees.                     | Albumentations |
| RandomScale                   | Randomly scale the image.                                                    | Albumentations |
| RandomSizedBBoxSafeCrop       | Crop randomly without cutting bounding boxes, with resizing.                 | Albumentations |
| RandomSizedCrop               | Randomly crop and resize with varying aspect ratios and sizes.               | Albumentations |
| Resize                        | Resize the image to a specified size.                                        | Albumentations |
| Rotate                        | Rotate the image by a specified angle.                                       | Albumentations |
| SafeRotate                    | Rotate the image by a specified angle with safety checks.                    | Albumentations |
| ShiftScaleRotate              | Apply shifting, scaling, and rotation to the image.                          | Albumentations |
| SmallestMaxSize               | Resize the image so the smallest side is as specified.                       | Albumentations |
| Transpose                     | Transpose the image (rotate by 90 degrees and flip horizontally).            | Albumentations |
| VerticalFlip                  | Flip the image vertically.                                                   | Albumentations |
| XYMasking                     | Apply masking in the XY plane.                                               | Albumentations |
| adaptive-blur                 | Adaptively blur pixels with a decrease in effect near edges.                 | ImageMagick |
| adaptive-resize               | Resize image using data dependent triangulation.                             | ImageMagick |
| adaptive-sharpen              | Sharpen pixels adaptively, increasing effect near edges.                     | ImageMagick |
| annotate                      | Annotate the image with text.                                                | ImageMagick |
| auto-gamma                    | Automatically adjust the gamma level of the image.                           | ImageMagick |
| auto-level                    | Automatically adjust the color levels of the image.                          | ImageMagick |
| auto-orient                   | Automatically orient the image based on its metadata.                        | ImageMagick |
| adaptive-sharpen              | Adaptively sharpen pixels with an increase in effect near edges.             | ImageMagick |
| annotate                      | Annotate the image with text.                                                | ImageMagick |
| auto-gamma                    | Automatically adjust gamma level of the image.                               | ImageMagick |
| auto-level                    | Automatically adjust color levels of the image.                              | ImageMagick |
| auto-orient                   | Automatically orient the image based on EXIF data.                           | ImageMagick |
| bench                         | Measure the performance of operations.                                       | ImageMagick |
| black-threshold               | Force all pixels below the threshold into black.                             | ImageMagick |
| blue-shift                    | Simulate a scene at nighttime in moonlight.                                  | ImageMagick |
| blur                          | Reduce image noise and reduce detail levels with a Gaussian operator.        | ImageMagick |
| border                        | Surround image with a border of a specific color.                            | ImageMagick |
| brightness-contrast           | Improve brightness and contrast of the image.                                | ImageMagick |
| canny                         | Detect edges in the image using the Canny algorithm.                         | ImageMagick |
| charcoal                      | Simulate a charcoal drawing.                                                 | ImageMagick |
| chop                          | Remove a region of an image and collapse the image to occupy the removed portion. | ImageMagick |
| clamp                         | Restrict pixel values to within a specified range.                           | ImageMagick |
| clip                          | Clip along the first path from the 8BIM profile.                             | ImageMagick |
| clip-mask                     | Associate a clip mask with the image.                                        | ImageMagick |
| clip-path                     | Clip along a named path from the 8BIM profile.                               | ImageMagick |
| colorize                      | Colorize the image with the fill color.                                      | ImageMagick |
| color-matrix                  | Apply a color correction matrix to the image.                                | ImageMagick |
| connected-component           | Label connected components within an image.                                  | ImageMagick |
| contrast                      | Enhance or reduce the image contrast.                                        | ImageMagick |
| contrast-stretch              | Improve contrast by 'stretching' the intensity range.                        | ImageMagick |
| convolve                      | Apply a convolution kernel to the image.                                     | ImageMagick |
| cycle                         | Cycle the image's colormap.                                                  | ImageMagick |
| deskew                        | Straighten an image.                                                         | ImageMagick |
| despeckle                     | Reduce the speckles within an image.                                         | ImageMagick |
| distort                       | Distort images according to specified method and arguments.                  | ImageMagick |
| draw                          | Annotate the image with a graphic primitive.                                 | ImageMagick |
| edge                          | Apply a filter to detect edges in the image.                                 | ImageMagick |
| emboss                        | Emboss an image.                                                             | ImageMagick |
| enhance                       | Apply a digital filter to enhance a noisy image.                             | ImageMagick |
| equalize                      | Perform histogram equalization to an image.                                  | ImageMagick |
| evaluate                      | Evaluate an arithmetic, relational, or logical expression on an image.       | ImageMagick |
| extent                        | Set the image size.                                                          | ImageMagick |
| extract                       | Extract a region of the image.                                               | ImageMagick |
| fft                           | Implements the Discrete Fourier Transform (DFT).                             | ImageMagick |
| adaptive-sharpen              | Adaptively sharpen pixels with an increase in effect near edges.             | ImageMagick |
| annotate                      | Annotate the image with text.                                                | ImageMagick |
| auto-gamma                    | Automatically adjust gamma level of the image.                               | ImageMagick |
| auto-level                    | Automatically adjust color levels of the image.                              | ImageMagick |
| auto-orient                   | Automatically orient the image based on EXIF data.                           | ImageMagick |
| bench                         | Measure the performance of operations.                                       | ImageMagick |
| black-threshold               | Force all pixels below the threshold into black.                             | ImageMagick |
| blue-shift                    | Simulate a scene at nighttime in moonlight.                                  | ImageMagick |
| blur                          | Reduce image noise and reduce detail levels with a Gaussian operator.        | ImageMagick |
| border                        | Surround image with a border of a specific color.                            | ImageMagick |
| brightness-contrast           | Improve brightness and contrast of the image.                                | ImageMagick |
| canny                         | Detect edges in the image using the Canny algorithm.                         | ImageMagick |
| charcoal                      | Simulate a charcoal drawing.                                                 | ImageMagick |
| chop                          | Remove a region of an image and collapse the image to occupy the removed portion. | ImageMagick |
| clamp                         | Restrict pixel values to within a specified range.                           | ImageMagick |
| clip                          | Clip along the first path from the 8BIM profile.                             | ImageMagick |
| clip-mask                     | Associate a clip mask with the image.                                        | ImageMagick |
| clip-path                     | Clip along a named path from the 8BIM profile.                               | ImageMagick |
| colorize                      | Colorize the image with the fill color.                                      | ImageMagick |
| color-matrix                  | Apply a color correction matrix to the image.                                | ImageMagick |
| connected-component           | Label connected components within an image.                                  | ImageMagick |
| contrast                      | Enhance or reduce the image contrast.                                        | ImageMagick |
| contrast-stretch              | Improve contrast by 'stretching' the intensity range.                        | ImageMagick |
| convolve                      | Apply a convolution kernel to the image.                                     | ImageMagick |
| cycle                         | Cycle the image's colormap.                                                  | ImageMagick |
| deskew                        | Straighten an image.                                                         | ImageMagick |
| despeckle                     | Reduce the speckles within an image.                                         | ImageMagick |
| distort                       | Distort images according to specified method and arguments.                  | ImageMagick |
| draw                          | Annotate the image with a graphic primitive.                                 | ImageMagick |
| edge                          | Apply a filter to detect edges in the image.                                 | ImageMagick |
| emboss                        | Emboss an image.                                                             | ImageMagick |
| enhance                       | Apply a digital filter to enhance a noisy image.                             | ImageMagick |
| equalize                      | Perform histogram equalization to an image.                                  | ImageMagick |
| evaluate                      | Evaluate an arithmetic, relational, or logical expression on an image.       | ImageMagick |
| extent                        | Set the image size.                                                          | ImageMagick |
| extract                       | Extract a region of the image.                                               | ImageMagick |
| fft                           | Implements the Discrete Fourier Transform (DFT).                             | ImageMagick |
