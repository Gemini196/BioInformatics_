# BioInformatics_Proj
##ML: Random Forest and Linear Regression

Training/Testing: CCLE data (928 samples)

Feature Selection:
The features chosen are genes. The idea was to find the genes with highest correlation to doubling time.

The method presented here included running Pearson correlation tests between each gene expression column and the doubling time array. Genes that resulted in Rho value of 0.37 and higher, and p-value of 0.05 and lower were chosen as features. In addition, due the high influence that cancer type has on doubling time, the following cancer marker genes were also included as features: RB1, PIK3CA, BRAF, CTNNB1, CDKN2A, MLH1, APC, BRCA2, BRCA1, FGFR3, KRAS, EGFR, ATM, VHL, MET, PTEN, TP53, ERBB2. Totalling in 30 Genes to be considered as features.
