# DDN
Differential Dependency Network

DDN 2.1 Manual

## Data Format

* The input file:
	* Gene name: filename_genename.txt
	* Expression data in condition 1 (case): filename_case.txt
	* Expression data in condition 2 (control) : filename_control.txt

For both expression data, row represents gene, column represents sample, e.g., 
    #   | Sample 1 | Sample 2 | Sample 3 
 ------ | -------- | -------- | -------- 
 Gene 1 |    #     |    #     |    #     
 Gene 2 |    #     |    #     |    #     
 Gene 3 |    #     |    #     |    #     

* Input file format: *.txt
* The output file: *.csv

The first column and third column represent the nodes with differential edges, the second column stands for the differential edges detected under which condition
Output file will be saved as *.csv format

* Differential network will be plotted in color-coded graph

## Code File:

* General file: DDN 2.1 package example
* Main function: DDN.py

Please use Jupyter Notebook to open the .ipynb file

## Demo

* Simulated data by SynTreN

For demo, please see DDN 2.1 package example.ipynb
