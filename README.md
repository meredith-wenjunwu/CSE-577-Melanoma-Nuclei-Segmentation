# CSE-577-Melanoma-Nuclei-Segmentation
Class project for CSE 577: Nuclei Segmentation in Skin histopathological images. 

## Author Info

**Shima Nofallah**: shima@cs.washington.edu

**Meredith Wu**: wenjunw@cs.washington.edu

## Usage

`main.py` For data loading and main pipeline. Refer to code block for function of code.

 - `computeFeatureInDirectory(inputDirectory, outputDirectory)` computes features of images in a directory and write as
 - `trainAdaboostwithDirectory()` uses online-boosting to train with features in a directory. 

`feature.py` has the supporting feature computation for `main.py`

`cellBodyFinder.py` uses watershed to segment cell-bodies given a mask of labeled nuclei. The centers of the nuclei will be marked as '+'