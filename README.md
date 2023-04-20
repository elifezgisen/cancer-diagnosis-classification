# Cancer Diagnosis Classification

## Diagnosis of Cancer Using Blood Microbiome Data

Recently, it has been discovered that DNA samples of microbes in human blood can be symptoms of various types of cancer in the human body.

### Data

We have blood sample data from 355 people with the 4 most common types of cancer: colon cancer, breast cancer, lung cancer and prostate cancer. labels.csv is a label file that shows the sample names and the disease type of each person with the corresponding sample name. The data is stored in the data.csv file. Again, each row contains the sample name of the corresponding person, and the rest is the number of DNA fragments for each type of microorganism (virus or bacteria). 1836 different microorganisms appear as features.

### Model

With this project, it is expected to have the highest possible correct classification scores. There will be 4 different classifications. (Colon Cancer, Breast Bancer, Lung Cancer and Prostate Cancer)

### Classification Algorithms

Classifications were made with two different algorithms, Random Forest and Gradient Boosted Trees. As a result, the performances of these two algorithms were compared.

### Performance Measures

Sensitivity and Specificity are requested as performance output from the program.
