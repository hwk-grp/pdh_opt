# Artificial-intelligence-guided discovery and optimization of heterogeneous catalysts for propane dehydrogenation 

This repository reported sample Python codes used to optimize catalyst composition and reaction conditions. We executed these codes sequentially for two different methods: XGBoost (XGB) and fully connected neural network (FCNN). We performed the optimization for two sequential rounds. Codes for the first round stored in directories whose names start with 01 and 02, and codes for the second round were in directories whose names start with 03 and 04. We saved the codes for constructing ML models separately. These ML models were utilized for the optimization using the artificial bee colony (ABC) algorithm. These optimization codes were stored in directories with names starting with 02 and 04. Each directory contained commands actually used in the command.csh script file.

Installation
---------

1. Clone the package.
```
git clone https://github.com/hwk-grp/pdh_opt.git
```
2. Create conda environment using environment.yml file. 
```
# Tested in Linux 
conda env create -f environment.yml
```

Usage
---------

To run the FCNN code, follow these steps:
1. Go to the 01.FCNN.v1 directory and run command.csh. (Note: command.csh is a C-shell script file.). You can build FCNN models.
2. Go to the 02.ABC_FCNN.v1 directory and run command.csh. It provides optimized features with the highest score among the features that have been tried. This is what we did in our first round of the optimization.
3. In the second round, the FCNN model can be built using the code from 03.FCNN.v2, and then the catalyst score was optimized using the code from 04.ABC_FCNN.v2.

To run the XGB (XGBoost version 1.3.x) code, follow these steps:
1. Go to the 01.XGB.v1 directory and run command.csh. (Note: command.csh is a C-shell script file.). You can build XGB models. It should be noted that the code has been tested with XGBoost version 1.3.x. 
2. Go to the 02.ABC_XGB.v1 directory and run command.csh. It provides optimized features with the highest score among the features that have been tried. This is what we did in our first round of the optimization.
3. In the second round, the XGB model can be built using the code from 03.XGB.v2, and then the catalyst score was optimized using the code from 04.ABC_XGB.v2.

