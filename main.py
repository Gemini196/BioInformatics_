# Data Processing
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def train_and_predict(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train) # Train the model
    predictions = model.predict(X_test) # Make predictions on the test set
    #predictions[predictions <= 0] = 1e-10  # Handling zero values in predictions
    predictions[predictions < 0] = 0  # Handling negative values in predictions
    r2 = r2_score(y_test, predictions)
    return predictions


def print_stats(y_test, predictions):
    print('Mean Squared Error (MSE):', mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, predictions)))
    mape = np.mean(np.abs((y_test - predictions) / np.abs(y_test)))
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(100 * (1 - mape), 2))

    correlation_coefficient, p_value = spearmanr(y_test, predictions)
    print(f"\nSpearman Rho:{correlation_coefficient}\np-value:{p_value}")


if __name__ == '__main__':
    #############################################
    ########## Load and Clean Data ##############
    #############################################
    # Load clinical data file
    df = pd.read_csv('./ccle_broad_2019_clinical_data.csv')
    df_cleaned = df.dropna(subset=['Doubling Time (hrs)']) # Clean 'NA' labels
    patient_IDs = df_cleaned["Patient ID"].tolist() # who was left after cleaning NA

    # Load second clinical data file
    df2 = pd.read_excel('./merged_doubling_time_with_site.xlsx')

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

    #############################################
    ########## Get Features (Genes) #############
    #############################################
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

    chosen_genes_expression = counts[counts.columns[counts.columns.isin(genes)]]

    # add the doubling time as a column in the df
    #chosen_genes_expression['Doubling Time (hrs)'] = metadata['Doubling Time (hrs)']

    #############################################
    ######## Train Models and Predict ###########
    #############################################

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(chosen_genes_expression, metadata['Doubling Time (hrs)'], test_size=0.2, random_state=42)

    ############## Random Forest ##############
    print("\n--- Random Forest ---\n")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    predictions = train_and_predict(rf_model, X_train, X_test, y_train, y_test)

    print_stats(np.log10(y_test), np.log10(predictions))

    # Correlation Figure
    plt.scatter(np.log10(predictions), np.log10(y_test), color='blue', label='Data points')
    b, m = polyfit(np.log10(predictions), np.log10(y_test), 1) # Fit with polyfit
    plt.plot(np.log10(predictions), b + m * np.log10(predictions), '.')
    plt.xlabel('log(Predictions)')
    plt.ylabel('log(y_test) values')
    plt.title('Correlation Figure')
    plt.legend()
    plt.show()

    ############## Linear Regression ##############
    print("\n--- Linear Regression ---\n")
    lr_model = LinearRegression()
    predictions = train_and_predict(lr_model, X_train, X_test, y_train, y_test)

    print_stats(y_test, predictions)

    # Correlation Figure
    plt.scatter(predictions, y_test, color='blue', label='Data points')
    b, m = polyfit(predictions, y_test, 1) # Fit with polyfit
    plt.plot(predictions, b + m * predictions, '.')
    plt.xlabel('Predictions')
    plt.ylabel('y_test values')
    plt.title('Correlation Figure')
    plt.legend()
    plt.show()



