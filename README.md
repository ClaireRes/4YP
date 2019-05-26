# 4th Year Project Repository
Compiled program listing &amp; model training results for the 4th Year Project: Regional Respiratory Analysis With Non-invasive Imaging and 1D-CNNs. This repository is an appendix to the given report and is not intended as a standalone repo to reproduce the work within it. Actual data used in this report is omitted due to data sharing restrictions.

## Getting started
You will require a version of Python 3.4 or above to run the Python scripts in this repo, and the Jupyter Notebook App with the Python 3 kernel to run the notebook files.
A suggested installation package to setup the above is the latest Python 3 version of the Anaconda Package, available from:
https://www.anaconda.com/distribution/

## Contents

### Example data files

The ```slp_demo_file.pn3``` is an example of the raw file format collected by the [PneumaCare Thora-3di](https://www.pneumacare.com/thora-3di) system. This is separately processed into two CSV files: 

```slp_demo_file.csv``` contains a concatenated array of the X-Y-Z coordinates for each grid point from a SLP recording for each frame. 
The array has dimensions ```[grid_size_rows x (3*grid_size_cols*num_frames)]```.

```slp_demo_file_gs.csv``` contains the dimensions of the projected grid used in the SLP recording as a 1x2 array of
```[grid_size_rows x grid_size_cols]```, used for frame-by-frame processing of the grid data.


### Data processing scripts

```visualise.py``` performs the reconstruction of the 3D surface data for a particular sample from its two CSV files, as well as normalisation, segmentation and volume curve extraction for this surface. The results are plotted as a dynamic reconstruction of the SLP recording. The entire recording can be reconstructed, or a subset of ```sequence_length``` frames from the first frame can be analysed. CSV file names and sequence length to analyses can be passed on command line using:

```visualise.py --file.csv --file_gs.csv --length ```

or can be changed in the main script for ```f_in```, ```f_in_gs``` and ```sequence_length```.

```extract_clf_inputs.py``` performs bulk processing using the same techniques as ```visualise.py``` of files of the same class into the format required for use in classifier training notebooks. The ```file_path``` set in main is the directory containing the .pn3 file and .CSV files for each sample to be processed.

The ```results``` directory created in this folder contains the following files after running:

```results_q1.csv``` = row-wise storing of relative volume sequence for right chest segment of each sample processed

```results_q2.csv``` = as above but for left chest segment

```results_q3.csv``` = as above but for right abdominal segment

```results_q4.csv``` = as above but for left abdominal segment

```results_names.csv``` = row-wise storing of sample names

```results_labels.csv``` = row-wise storing of sample class labels for classification use

The ```split_samples``` binary parameter in main is used to select between extracting the first slice of ```slice_length``` from each sample (0), or extracting as many slices as possible of this length from each sample (1).


### Classifier Training Notebooks

```cnn-asthma-classifier``` uses input data in form from data processed by ```extract_clf_inputs.py``` to train a 1D CNN binary classifier for paediatric asthma classification. Model implemented using TensorFlow.


```cnn-COPD-classifier``` as above but for the training of a 1D CNN binary classifier for COPD classification.

### Analysis of Optimal Classifier results

Using predicted labels from the optimal classifiers found in the notebooks above, the ```Analyse_predictions``` notebook was used to carry out ROC-curve analysis and estimate probability density functions for each classifier's predictions on its associated test dataset.


### Compiled training results

The final round of training results and summary details of the data cohorts are compiled in the ```compiled_training_results``` spreadsheet. 



