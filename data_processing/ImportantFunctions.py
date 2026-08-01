import os
import pandas as pd
from collections import defaultdict
import warnings

def group_files_by_keyword(file_list, keywords):
    """
    Group files based on shared keywords found in their filenames.

    Parameters
    ----------
    file_list : list of str
        List of filenames or file paths to be grouped.
    keywords : list of str
        List of keywords that define the grouping criteria.

    Returns
    -------
    dict of str : list of str
        A dictionary where each key is a keyword and the corresponding value
        is a list of files containing that keyword in their name.

    Notes
    -----
    - Matching is case-sensitive by default. Use `str.lower()` on filenames and
      keywords before calling the function to make it case-insensitive.
    - A file will be added to the first group whose keyword it matches.
      Remove the `break` statement if you want files to belong to multiple groups.
    """
    groups = defaultdict(list)

    for file in file_list:
        isFound = False
        name = os.path.basename(file)
        for keyword in keywords:
            if keyword in name:
                groups[keyword].append(file)
                isFound = True
                break
        if not isFound:
            warnings.warn(f"{file} doesn't match with any keyword from {keywords}")

    return dict(groups)


def sort_function(string):
    return int(string.split("_")[-1].split(".")[0])

def sort_function_Pickle(string):
    split_string = string.split("_")
    return int(split_string[-2]) + int(split_string[-1].split(".")[0])

def CSV_merge(folder_path, exp_id):
    """This function merges the CSV files found in folder_path and saves the merged file in the same directory."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not files:
        return
 
    files.sort(key=sort_function)

    # Create an empty DataFrame
    combined_DataFrame = pd.DataFrame()

    # Iterate the CSV files found in the folder path
    print("\nMerging...")
    for file in files:
        # Read CSV
        df = pd.read_csv(os.path.join(folder_path, file), header=0, 
                         index_col=False, delimiter=';', decimal='.')

        print(file)

        # Concatenate CSV file
        combined_DataFrame = pd.concat([combined_DataFrame, df], ignore_index=True)
    
    # Delete individual CSV files after successful merge
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            os.remove(file_path)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Warning: Could not delete {file}: {e}")

    # Save concatenated DataFrame
    filename = f'Motor-{exp_id}.csv'
    combined_DataFrame.to_csv(os.path.join(folder_path, filename), index=False)

    print("Data saved to location:", os.path.join(folder_path, filename), "\n")

    return filename


def Pickle_merge(folder_path, exp_id, groupby=None):
    """This function merges the Pickle files found in folder_path and saves the merged file in the same directory."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    if not files:
        return
    
    files.sort(key=sort_function_Pickle)

    if groupby:

        groups = group_files_by_keyword(files, groupby)
        filenames = []

        for n, group in enumerate(groups.keys()):

            # Create an empty DataFrame
            combined_DataFrame = pd.DataFrame()

            # Iterate the Excel files found in the folder path
            print(f"\nMerging grouped by keyword {group}...")
            for file in groups[group]:
                # Read Excel
                df = pd.read_pickle(os.path.join(folder_path, file))

                print(file)

                # Concatenate CSV file
                combined_DataFrame = pd.concat([combined_DataFrame, df], ignore_index=True)

            # Save concatenated DataFrame
            filenames.append(f'DAQ-{group}-{exp_id}.pkl')
            combined_DataFrame.to_pickle(os.path.join(folder_path, filenames[n]))

            print("Data saved to location:", os.path.join(folder_path, filenames[n]))

        print("")

        return filenames

    else:

        # Create an empty DataFrame
        combined_DataFrame = pd.DataFrame()

        # Iterate the Excel files found in the folder path
        print("\nMerging...")
        for file in files:
            # Read Excel
            df = pd.read_pickle(os.path.join(folder_path, file))

            print(file)

            # Concatenate CSV file
            combined_DataFrame = pd.concat([combined_DataFrame, df], ignore_index=True)

        # Save concatenated DataFrame
        filename = f'DAQ-{exp_id}.pkl'
        combined_DataFrame.to_pickle(os.path.join(folder_path, filename))

        print("Data saved to location:", os.path.join(folder_path, filename))

        return filename

def validate_binary_column(df, column):
    # Check that all te values are {0, 1}
    if not df[column].isin([0, 1]).all():
        # Identify which are the values that are not 0 or 1
        wrong_values = df[~df[column].isin([0, 1])][column].unique()
        raise ValueError(f"Error: The column '{column}' contains not binary values: {wrong_values}")


def synchronize_dataframes(dataframes_list, time_col='Time (s)', filter_time=True, binary_cols=("LinMot_Enable", "LinMot_Up_Down")):
    """
    Synchronizes a list of DataFrames to the highest sampling rate found among them.
    It does a temporal boundary alignment as well (make all data have the same physical duration).
    IMPORTANT: All dataframes have to start at the same timestep (for example, time = 0) and have different columns

    Args:
        dataframes_list (list): List of Pandas DataFrames.
        time_col (str): The name of the time column (must be present in all DFs).

    Returns:
        List of Pandas DataFrames having the same time column as an index
    """

    if not dataframes_list:
        return []

    # =================================================================
    # STEP 0: Temporal Boundary Alignment: all data represent the same physical duration of the experiment
    # =================================================================

    # Check that timestamps are correct:
    for df in dataframes_list:
        if not (df[time_col].is_monotonic_increasing and df[time_col].is_unique):
            raise ValueError("The time column must be strictly increasing for interpolation.")

    # Check that all dataframes start at the same time
    valor_ref = dataframes_list[0].iloc[0][time_col]
    all_equal = all(df.iloc[0][time_col] == valor_ref for df in dataframes_list)
    if not all_equal:
        raise Exception("The dataframes don't start at the same time")

    if filter_time:
        # Find the elapsed time for each dataframe and select the smallest one
        print("Analysing time duration...")
        min_end_time = float('inf')
        for i, df in enumerate(dataframes_list):
            current_time = df[time_col].iloc[-1]
            print(f"  - DataFrame {i}: Time duration = {current_time:.6f} s")
            if current_time < min_end_time:
                min_end_time = current_time
        print(f"The experiment ended at time: {min_end_time:.4f} seconds.")

        # Filter the dataframes to make them have the same time duration
        for i in range(len(dataframes_list)):
            # Filter the dataframes to the min_end_time timestamp (ignore old index [drop=True] and create a new one)
            dataframes_list[i] =\
                dataframes_list[i][dataframes_list[i][time_col] <= min_end_time].copy().reset_index(drop=True)
        print("All dataframes now have the same time duration")

    # =================================================================
    # STEP 1: Find the Highest Sampling Rate DataFrame
    # =================================================================
    min_time_step = float('inf')
    master_time_index = None

    print("Analyzing sampling rates...")

    for i, df in enumerate(dataframes_list):

        current_step = df[time_col].diff().mean()
        print(f"  - DataFrame {i}: Average timestep = {current_step:.6f} s")

        if current_step < min_time_step:
            # Save the timestamp of the highest sampling rate DataFrame
            master_time_index = df[time_col].values
            min_time_step = current_step

    print(f"Target sampling step: {min_time_step:.6f} s")

    # =================================================================
    # STEP 2: Resample and Interpolate all DataFrames
    # =================================================================
    synced_dataframes = []

    for df in dataframes_list:

        # 1. Prepare the DF: remove duplicates and set the time column as the index
        df_temp = df.drop_duplicates(subset=[time_col]).set_index(time_col)

        # 2. Reindex and Interpolate
        # - union(): Merges the dataframe's timestamp with the higher sampling rate timestamp
        # - interpolate(method='index'): Uses the actual time values to calculate data in the new timesteps
        # - loc[master_time_index]: Keeps only the points belonging to the master timestamp
        df_sync = (
            df_temp
            .reindex(df_temp.index.union(master_time_index))
            .interpolate(method='index')
            .loc[master_time_index]
        )

        # 3. Ensure binary columns to be int data type
        found_binary_columns = [col for col in binary_cols if col in df_sync.columns]
        if found_binary_columns:
            # Apply rounding and cast to integer for all matching columns simultaneously (Vectorization)
            df_sync[found_binary_columns] = df_sync[found_binary_columns].round().astype(int)

        synced_dataframes.append(df_sync)

    return synced_dataframes

def merge_DAQ_data(folder_path, time_col='Time (s)'):
    files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    dataframes = []

    for file in files:
        try:
            df = pd.read_pickle(os.path.join(folder_path, file))
            dataframes.append(df)
        except Exception as e:
            raise Exception(f'Error reading DAQ file {file}: {e}.')

    # Synchronize dataframes
    synced_dataframes = synchronize_dataframes(dataframes, time_col=time_col)

    # Concatenate directly (since all DFs share the same master time index)
    df = pd.concat(synced_dataframes, axis=1)

    # Restore the time index as a standard column
    df.index.name = time_col
    df = df.reset_index()

    # Check synchronization columns:
    if not ("LinMot_Enable" in df.columns and "LinMot_Up_Down" in df.columns):
        raise Exception("Error, LinMot_Up_Down and LinMot_Enable are not present in the dataframes")

    # Validate that they have valid data
    for col in df.columns:
        if col in ["LinMot_Enable", "LinMot_Up_Down"]:
            validate_binary_column(df, col)

    # Calculate the differences between adjacent values
    diff = df["LinMot_Enable"].diff()

    # Search indices
    up_index = diff.index[diff == 1].tolist()
    down_index = diff.index[diff == -1].tolist()

    if len(up_index) != 1 and len(down_index) != 1:
        raise Exception("Error, LinMot_Enable start and end position not found")
    else:
        up_index = up_index[0] - 1
        down_index = down_index[0]

    print("Found rising edge and falling edge positions in LinMot_Enable")
    print(f"Rising edge (from 0 to 1): {up_index}")
    print(f"Falling edge (from 1 to 0): {down_index}")

    # Filter dataframes using the calculated indices and take the first row as the time reference value
    df_final = df.loc[up_index : down_index].reset_index(drop=True)
    df_final[time_col] -= df_final[time_col].iloc[0]
    return df_final

if __name__ == "__main__":
    df_final = merge_DAQ_data(r"C:\Users\rpieres\Desktop\RogerTest\RawData\RawData")
    print("DONE")

        
        
        
        
        
        
        
        
        
        