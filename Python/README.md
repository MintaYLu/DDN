# Differential Dependency Network

DDN 2.1 Manual

## DDN Data
* Data Format:
	* Input file format: *.txt
	* The output file: *.csv

* Code File:
	* General file: DDN 2.1 package example
	* Main function: DDN.py

Please use Jupyter Notebook to open the *.ipynb file

* Input file:
	* Gene name: filename_genename.txt
	* Expression data in condition 1 (case): filename_case.txt
	* Expression data in condition 2 (control) : filename_control.txt

For both expression data, row represents gene, column represents sample, e.g., 
    #   | Sample 1 | Sample 2 | Sample 3 
 ------ | -------- | -------- | -------- 
 Gene 1 |    #     |    #     |    #     
 Gene 2 |    #     |    #     |    #     
 Gene 3 |    #     |    #     |    #     

* Output file:
	* Differential edges: filename_diffedges.csv

The first column and second column represent the nodes with differential edges, the third column stands for the differential edges detected under which condition, the fourth columns shows the strength of the differential edges, which is the nontrival linear regression coefficient between the two nodes.

* Graphic output:
	* Differential network: will be plotted in color-coded graph

## DDN Demo

* Simulated data by SynTreN

For demo, please see DDN 2.1 package example.ipynb
