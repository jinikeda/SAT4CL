# SAT4CL
Classify &amp; Label land types using Satellite data with DL model

This package has been developed further based on the WatNet2 Deep Learning model (Added multiple ML architecture). 

## §1. Install virtual conda env and activation
Type: ***conda env create -f env.yml*** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; autopep8 is a gentle Python code formatter, flake8 is a linter to check code, and pytest and mock are code testing tools. These packages are not mandatory for running the model(future development of pytest).  

## §2. Activate virtual env
Type: ***conda activate SAT_env***

## §3. Train DL model (optional)
Run src/DL_processing_train.py (you need datasets/s2/tra_xxx etc now)

## §4. Validate data
Run src/val_check.py or val_check.ipynb (you need src/model/trained_model/***pth)

# Development note

## Contributors
* Jin Ikeda (LSU|Center for Computation and Technology)

## Acknowledgments
* [xinluo2018](https://github.com/xinluo2018/WatNetv2)
* [rishikksh20](https://github.com/rishikksh20/ResUnet)
