### Synthetic, depth-indexed porosity/permeability data per Petrophysical Rock Type (PRT)

This program produces a synthetic k-phi dataset (vs depth) from a user-specified Petrophysical Rock Types (PRT) table with statistics per rock type. An incidental lithology column based on the PRT is produced and subsequently porosity/permeability data is generated - all depth indexed.

The code works such that a random (but more structured than complete random) ordering of the PRTs is generated: a synthetic lithology column. Due to this incidental structuring, this column resembles more "true geology" than what would be generated from a complete random state. Once the "stacking" of PRTs is in place, the random variables are transformed according to user-specified statistics for each of the PRTs.

![screenshot result](https://github.com/harrybrandsen/synthetic_k_phi/blob/master/screenshot_result_synthetic_k_phi.png)

### Known Issues
- The lithology/PRT column isn't drawn entirely correct. Especially when more than 2 PRTs are used, or when the column is relatively short, the colors visible do not seem to correspond to the occurrence of each of the PRTs (see percentages top of plot). The data (stored in the pandas DataFrame) are correct though (i.e. it is a graphics or plotting issue)
       
- The correlation metrics in the "litholib"/PRT dictionary need to be quite high, even for PRTs that will ultimately be poorly correlated.
This is unfortunate, as it is somewhat trial and error to get a "good-looking" dataset.



### Acknowledgements
The function "create_approximately_structured_single_variable" was written
by Co Stuifbergen in JavaScript and translated to Python by HARBR.


