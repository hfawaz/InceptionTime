# InceptionTime: Finding AlexNet for Time Series Classification
This is the companion repository for our paper titled [InceptionTime: Finding AlexNet for Time Series Classification](https://link.springer.com/article/10.1007/s10618-020-00710-y) published in [Data Mining and Knowledge Discovery](https://www.springer.com/journal/10618) and also available on [ArXiv](https://arxiv.org/pdf/1909.04939.pdf). 

## Inception module
![inception module](https://github.com/hfawaz/InceptionTime/blob/master/pngs/inception-module.png)

## Data
The data used in this project comes from the [UCR/UEA archive](http://timeseriesclassification.com/TSC.zip). 
We used the 85 datasets listed [here](https://www.cs.ucr.edu/~eamonn/time_series_data/).  

## Requirements
You will need to install the following packages present in the [requirements.txt](https://github.com/hfawaz/InceptionTime/blob/master/requirements.txt) file. 

## Code
The code is divided as follows: 
* The [main.py](https://github.com/hfawaz/InceptionTime/blob/master/main.py) python file contains the necessary code to run an experiement. 
* The [utils](https://github.com/hfawaz/InceptionTime/tree/master/utils) folder contains the necessary functions to read the datasets and visualize the plots.
* The [classifiers](https://github.com/hfawaz/InceptionTime/tree/master/classifiers) folder contains two python files: (1) [inception.py](https://github.com/hfawaz/InceptionTime/tree/master/classifiers/inception.py) contains the inception network; (2) [nne.py](https://github.com/hfawaz/InceptionTime/tree/master/classifiers/nne.py) contains the code that ensembles a set of Inception networks. 

### Adapt the code for your PC
You should first consider changing the following [line](https://github.com/hfawaz/InceptionTime/blob/c9a323c789984e3fb56e82ebb4eea6438611e59c/main.py#L83). 
This is the root file of everything (data and results) let's call it ```root_dir```. 

After that you should create a folder called ```archives``` inside your ```root_dir```, which should contain the folder ```UCR_TS_Archive_2015```. 
The latter will contain a folder for each dataset called ```dataset_name```, which can be downloaded from this [website](https://www.cs.ucr.edu/~eamonn/time_series_data/).

The names of the datasets are present [here](https://github.com/hfawaz/InceptionTime/blob/c9a323c789984e3fb56e82ebb4eea6438611e59c/utils/constants.py#L1). 
You can comment [this line](https://github.com/hfawaz/InceptionTime/blob/c9a323c789984e3fb56e82ebb4eea6438611e59c/utils/constants.py#L19) to run the experiments on all datasets. 

Once you have done all that, you can proceed to run on a single archive. 

### Run InceptionTime on a single Archive
You should issue the following command ```python3 main.py InceptionTime```. 

### Run the hyperparameter search for InceptionTime on a single Archive
You should issue the following command ```python3 main.py InceptionTime_xp```. 

### Run the length experiment on the InlineSkate dataset
You should first issue the following command ```python3 main.py run_length_xps``` to generate the resamples.
Then you should issue the following command ```python3 main.py InceptionTime``` but make sure that the ```InlineSkateXPs``` is chosen [here](https://github.com/hfawaz/InceptionTime/blob/690aa776081e77214db95ddd5c53c7ec3ac79d61/utils/constants.py#L22). 

### Receptive field
To run the experiments on the synthetic dataset, you should issue the following command ```python3 receptive.py```. 

## Results
The result (i.e. accuracy) for each dataset will be present in ```root_dir/results/nne/incepton-0-1-2-4-/UCR_TS_Archive_2015/dataset_name/df_metrics.csv```.

The raw results can be found [here](https://github.com/hfawaz/InceptionTime/blob/master/results-InceptionTime-85.csv) and generated using the following command ```python3 main.py generate_results_csv```.

We added the full results for the 128 datasets from the UCR archive, they can be found [here](https://github.com/hfawaz/InceptionTime/blob/master/results-InceptionTime-128.csv). 

<!-- We have added the full results for the 30 datasets from the [MTS UEA archive](http://www.timeseriesclassification.com/), they can be found [here](https://github.com/hfawaz/InceptionTime/blob/master/results-mts.csv). 
 -->

The [results-inception-128.csv](https://github.com/hfawaz/InceptionTime/blob/master/results-inception-128.csv) file contains five individual runs of the Inception model over the 128 datasets from the UCR 2018 archive. 

### Critical difference diagrams
If you would like to generate such a diagram, take a look at [this code](https://github.com/hfawaz/cd-diagram)!

![cd diagram](https://github.com/hfawaz/InceptionTime/blob/master/pngs/cd-diagram.png)

### Training time plots
These plots were generated using the [matplotlib](https://matplotlib.org/) library. 

Accuracy vs train size             |  Accuracy vs series length
:-------------------------:|:-------------------------:
![training time size](https://github.com/hfawaz/InceptionTime/blob/master/pngs/train-time-size.png) | ![training time length](https://github.com/hfawaz/InceptionTime/blob/master/pngs/train-time-length.png)

### Receptive field
This plot was generated by issuing this command ```python3 receptive.py plot_results```.


Receptive field effect             |  Depth effect
:-------------------------:|:-------------------------:
![receptive field](https://github.com/hfawaz/InceptionTime/blob/master/pngs/plot-receptive-field.png) | ![training time length](https://github.com/hfawaz/InceptionTime/blob/master/pngs/depth-vs-length.png)

## Reference

If you re-use this work, please cite:

```
@article{IsmailFawaz2020inceptionTime,
  Title                    = {InceptionTime: Finding AlexNet for Time Series Classification},
  Author                   = {Ismail Fawaz, Hassan and Lucas, Benjamin and Forestier, Germain and Pelletier, Charlotte and Schmidt, Daniel F. and Weber, Jonathan and Webb, Geoffrey I. and Idoumghar, Lhassane and Muller, Pierre-Alain and Petitjean, François},
  journal                  = {Data Mining and Knowledge Discovery},
  Year                     = {2020}
}
```

## Acknowledgement

We would like to thank the providers of the [UCR/UEA archive](http://timeseriesclassification.com/TSC.zip). 
We would also like to thank NVIDIA Corporation for the Quadro P6000 grant and the Mésocentre of Strasbourg for providing access to the cluster.
