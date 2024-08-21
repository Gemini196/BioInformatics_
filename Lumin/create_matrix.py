import os
import pandas as pd

# Directory containing the CSV files
csv_folder = 'csv_files'  # Replace with the path to your folder containing the CSV files

# Initialize an empty dictionary to store the data
data_dict = {}

# Loop through each CSV file in the directory
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv'):
        # Read the CSV file, skipping the first two lines and using the third line as the header
        file_path = os.path.join(csv_folder, filename)
        df = pd.read_csv(file_path, skiprows=2)

        # Extract the relevant columns: "Model", "Gene", and "Log (RPKM+1)"
        for _, row in df.iterrows():
            model = row['Model']
            gene = row['Gene']
            log_rpkm = row['Log (RPKM+1)']

            if model not in data_dict:
                data_dict[model] = {}
            data_dict[model][gene] = log_rpkm

# Convert the dictionary to a DataFrame
final_df = pd.DataFrame.from_dict(data_dict, orient='index').fillna(0)

#--------------------------------------------------------------

dt = pd.read_csv('DT.csv')   # read doubling time matrix
dt = dt.dropna(subset=['DT']) # Remove rows where 'DT' column has missing values

# FILTER EXPRESSION
patient_ID_list = dt['Model'].tolist()   # Get all Model names that have DT value to them
filtered_df = final_df[final_df.index.isin(patient_ID_list)] # Filter out rows where 'Model' is not in the list

filtered_df = filtered_df.sort_index()  # sort using index
filtered_df = filtered_df[~filtered_df.index.duplicated(keep='first')]  # Unique

# FILTER DOUBLING TIME
model_names = filtered_df.index.tolist()
filtered_dt = dt[dt['Model'].isin(model_names)]

print(filtered_df.head())

# Save the final DataFrame to a CSV file
filtered_df.to_csv('combined_log_rpkm.csv')

filtered_dt = filtered_dt[['Model', 'DT']]
filtered_dt.set_index('Model', inplace=True) # Set 'Model' as the index
filtered_dt.to_csv('filtered_DT.csv')