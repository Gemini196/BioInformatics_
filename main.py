# Data Processing
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# PCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# import plotly.graph_objs as go
# import plotly.offline as offline
# from plotly.offline import plot


# Modelling
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
#from IPython.display import Image
#import graphviz


if __name__ == '__main__':
    #############################################
    ########## Load and Clean Data ##############
    #############################################
    # Load clinical
    df = pd.read_csv('./ccle_broad_2019_clinical_data.csv')
    df_cleaned = df.dropna(subset=['Doubling Time (hrs)']) # Clean 'NA' labels
    patient_IDs = df_cleaned["Patient ID"].tolist() # who was left after cleaning NA

    # Load RNA expression data (use Reads Per Kilobase data)
    expression_df = pd.read_csv('./ccle_broad_2019/data_mrna_seq_rpkm.txt',  sep='\t')
    # Remove patients with NA Doubling Time
    selected_columns = ['Hugo_Symbol'] + [col for col in patient_IDs if col in expression_df.columns]
    expression_df = expression_df[selected_columns]
    print(expression_df.head())

    # sort by patient ID
    expression_df_sorted = expression_df[expression_df.columns[0]].to_frame().join(expression_df[expression_df.columns[1:]].sort_index(axis=1))
    df_cleaned_sorted = df_cleaned.sort_values(by='Patient ID')

    # Get all column names except for the first one and store them in a list
    patient_ID_list = expression_df_sorted.columns[1:].tolist()

    # Filter out rows where 'Patient ID' is not in the allowed list
    df_cleaned_sorted_filtered = df_cleaned_sorted[df_cleaned_sorted['Patient ID'].isin(patient_ID_list)]

    num_rows, num_columns = df_cleaned_sorted_filtered.shape
    print(f'Successfully loaded data.\nNumber of rows:{num_rows}\nNumber of columns:{num_columns-1}')
    print(df_cleaned_sorted_filtered.head())

    ## At this point we've got sorted expression data from 529 patients ##

    # Prepare counts
    counts = expression_df_sorted
    counts = counts.set_index('Hugo_Symbol')
    counts = counts[counts.sum(axis=1) > 0]   #remove rows containing only 0

    # Transpose the DataFrame to have samples as rows and genes as columns
    counts = counts.T #(=x.train)

    # prepare metadata (=y.train)
    metadata = pd.DataFrame(zip(counts.index, df_cleaned_sorted_filtered['Doubling Time (hrs)']),
                            columns=['Sample', 'Doubling Time (hrs)'])
    metadata = metadata.set_index('Sample')

    genes, rhos, pvals = [], [], []
    # Loop through each gene column
    for gene_idx in range(counts.shape[1]):
        # Extract the gene expression levels for the current gene
        gene_expression = counts.iloc[:, gene_idx].tolist()

        # Calculate correlation and p-value
        correlation_coefficient, p_value = pearsonr(gene_expression, metadata['Doubling Time (hrs)'].tolist())

        if abs(correlation_coefficient) >= 0.4 and p_value <= 0.05:
            genes.append(counts.columns[gene_idx])
            pvals.append(p_value)
            rhos.append(correlation_coefficient)
            #print(f"Gene: {counts.columns[gene_idx]}")
            #print(f"Correlation coefficient (rho): {correlation_coefficient}")
            #print(f"P-value: {p_value}")

    chosen_genes_expression = counts[counts.columns[counts.columns.isin(genes)]]

    # add the doubling time as a column in the df
    #chosen_genes_expression['Doubling Time (hrs)'] = metadata['Doubling Time (hrs)']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(chosen_genes_expression, metadata['Doubling Time (hrs)'], test_size=0.2, random_state=42)

    # Create a random forest regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = rf_model.predict(X_test)

    # Evaluate the model
    #mse = mean_squared_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    print('Mean Squared Error (MSE):', mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, predictions)))
    mape = np.mean(np.abs((y_test - predictions) / np.abs(y_test)))
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(100 * (1 - mape), 2))

    #print(f'Mean Squared Error: {mse}') # not very close to 0 - that's not a very
    #print(f'R-squared: {r2}')





