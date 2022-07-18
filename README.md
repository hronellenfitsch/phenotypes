# phenotypes
Code used to generate the data from H. Ronellenfitsch and E. Katifori. “Phenotypes of Vascular Flow Networks”. Physical Review Letters 123.24 (Dec. 2019), 248101. doi: 10.1103/PhysRevLett.123.248101.

The code is written in Julia 1.0.
You can run the code by typing

  `julia run_parallel_gamma.jl 0.5`

To run simulations with gamma=0.5. You may edit the file to change the networks that are generated.
The resultung .bson file can be turned into easily readable text files using the ExtractJuliaData.ipynb notebook the text files may be plotted using the LoadData.ipynb notebook (in Python).
