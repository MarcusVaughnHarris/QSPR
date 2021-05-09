# QSPR

### examples
The given example file (Bio_Surfactant_Target_Generation.ipynb) generates 10,000+ non-ionic surfactant candidates via common bio-derived reactants and simple 1-4 step synthetic routes. QSPR models are trained and used to predict their critical micelle concentrations (CMC). Finally, the candidates and their predicted CMC's are subset via outlier detection and their molecular distance to the training/test model. 

### Model Helper Functions
"GCNNhelperFun.py" contains a set of custom defined functions for quickly building, training, and testing Graph CNN's, while "QsarHelperFun.py" quickly builds, trains, and tests sequential models.

### Candidate Generation
"Rxn.Fun.py" contains a set of custom defined functions for generating molecules. The functions work to generate molecular candidates by reacting biomass-derived precursors with various cheap/easily-accessible reactants via 1-4 step synthetic routes. Reaction functions were created using custom-defined reaction SMARTS and functions from the AllChem package.

### Applicability Domain
"SimAD.py" has custom defined functions for estimating the reliability of predictions for molecules in a dataset.
