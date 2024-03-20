# **CSLP: Collaborative Solution to Long-tailed Problem and Popularity Bias in Sequential Recommendation**



## **Overview**  

### Overall Framework  
<img src="figure/Main.png" width="300">

### Abstract   

Sequential recommender systems, leveraging the temporal information from users' behaviors, have noticeably improved user experience against traditional systems. 
However, these behaviors often follow some long-tailed distribution, making the systems biased towards popular users or items (i.e., popularity bias). Moreover, popularity bias would amplify the neglect of long-tailed recommendations, thereby sharpening the long-tailed problem.
Previous research addresses these challenges separately, focusing on reducing the over-recommendation of popular items or enhancing the quality of long-tailed representation.
We can incorporate their merits to achieve the best of both worlds. Thus, we propose a novel and unified framework, named Collaborative Solution to Long-tailed problem and Popularity bias (CSLP), to tackle both the long-tailed problem and popularity bias simultaneously. 
To achieve this, we first introduce a representation enhancement module featuring dual generators to enhance user and item representations, particularly for those in the tail.
On the other hand, a debiasing module incorporating an Inverse Propensity Score (IPS) with a clipping strategy is introduced to alleviate the popularity bias further. 
Specifically, this clipping strategy demonstrates a clear decrease in the original IPS method's variance, effectively improving the recommendation for stability and accuracy.


## **Data Preprocess**  

### **1. Download the raw datasets (i.e., ratings only) in the following links**  

* [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html): Download the raw datasets instead of "5-core" datasets.

 
### **2. Then, put the raw datasets in the *raw_dataset* directory** 

### **3. Preprocess the datasets using **preprocess.ipynb** file**

We follow the same-preprocessing strategy with [SASRec](https://github.com/kang205/SASRec/blob/master/data/DataProcessing.py).

## **Library Versions**

* Python: 3.9.12  
* Pytorch : 1.10  
* Numpy: 1.21.2  
* Pandas: 1.3.4  

We upload the **environment.yaml** file to directly install the required packages.

``` python  
conda env create --file environment.yaml
``` 

## **Training**



### 1. Prepare the trained backbone (e.g., SASRec, FMLP) model.

We share the pretrained backbone encoder.  
You can donwload the pretrained model and put it on *save_model/{DATA_NAME}* directory.

* [SASRec](https://drive.google.com/drive/folders/1SKpdN_mAyMJgLTLSbqJOi3C9b8zm9Gbp?usp=sharing)

* [FMLP](https://drive.google.com/drive/folders/1D-dWuWKQB1VOwC91w26jjD1CvXqs2qx9?usp=sharing)

Or you can explicitly train the SASRec or FMLP model with the following commands.


#### SASRec  

``` python  
# In the shell code, please change the 'data' variable 
python main.py --inference false --model SASRec --e_max $e_max --batch_size $batch_size --dataset $data --gpu $gpu --pareto_rule $pareto_rule
```  

#### FMLP 

``` python  
# In the shell code, please change the 'data' variable 
python main.py --inference false --model FMLP --e_max $e_max --batch_size $batch_size --dataset $data --gpu $gpu --pareto_rule $pareto_rule
```  



### 2. Train the CSLP framework.


#### CSLP+SASRec  

``` python  
# In the shell code, please change the 'data' variable 
python main.py --inference false --dataset $data --gpu $gpu --model CSLP_SASRec --lamb_u $lamb_u --lamb_i $lamb_i --e_max $e_max --pareto_rule $pareto_rule --batch_size $batch_size
```  

#### CSLP+FMLP  

``` python  
# In the shell code, please change the 'data' variable 
python main.py --inference false --dataset $data --gpu $gpu --model CSLP_FMLP --lamb_u $lamb_u --lamb_i $lamb_i --e_max $e_max --pareto_rule $pareto_rule --batch_size $batch_size
```  


When you download the CSLP's pre-trained model, put it on *save_model/{DATA_NAME}* directory.  


To train the model on **Behance** or **Foursquare** datasets, please run the **train_others.sh** shell code.



## **Hyperparameter**  

* **CSLP+SASRec**


| Data   | lamb_U | lamb_I | e_max | Pareto(%) - a | gamma |
|--------|--------|--------|-------|---------------|-----|
| Music  | 0.4    | 0.9    | 180   | 20            | 0.1 |
| Beauty | 0.4    | 0.3    | 180   | 20            | 0.1 |
| Sports | 0.1    | 0.3    | 200   | 20            | 0.1 |








