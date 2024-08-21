# BioInformatics_Proj
##ML: Random Forest and Linear Regression

Training/Testing: CCLE/Lumin (928 samples/1065 samples)

Feature Selection:
The features chosen are genes. The idea was to find the genes with highest correlation to doubling time.

The method presented here included incorporated data inferred from PDXs taken from ‘Lumin’ website.
After removing ‘NA’ values, and cross- referencing the genes with our CCLE training set, we were left with 916 genes to be considered as features, and a Lumin testing set containing labeled data for 1065 patients.
