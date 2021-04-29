# Code for Thesis

This repository is used to generate all figures and tables in the thesis.

Note: TensorFlow version is 2.1. To update TensorFlow from an older version, you can run

        pip install tensorflow --upgrade

- Chapter 6

    - Way I:
    
        python gendata.py      #generate all needed data sequentially
        
        python Chapter6.py
        
    - Way II:
    
        python gendata.py    or     run gendata_parallel.ipynb 
        
        run Chapter6.ipynb
        
   Note: gendata_parallel.ipynb generates data in parallel with 10 clusters.
