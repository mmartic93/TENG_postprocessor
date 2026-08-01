import os
import re
import logging

import numpy as np
import pandas as pd
from data_processing.ImportantFunctions import merge_DAQ_data, synchronize_dataframes


# %% --------------------------------------------------------------------------
# INITIAL CONFIGURATION
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.ERROR,
                    format="%(levelname)s: %(message)s"
)

REQ_LEVEL = 25
logging.addLevelName(REQ_LEVEL, "REQ")

def req(self, message, *args, **kwargs):
    if self.isEnabledFor(REQ_LEVEL):
        self._log(REQ_LEVEL, message, args, **kwargs)

logging.Logger.req = req
logger0 = logging.getLogger(__name__)

if __name__ == "__main__":
    logger0.setLevel(logging.INFO)


# %% --------------------------------------------------------------------------
# PARSE CUSTOM TIME TO SECONDS
# -----------------------------------------------------------------------------

def LTIME_to_seconds(LTIME):
    
    conversor = {"d":86400,
                 "h": 3600,
                 "m": 60,
                 "s": 1,
                 "ms": 1e-3,
                 "us": 1e-6,
                 "ns": 1e-9}

    units = re.split(r'\d+', LTIME)[1:]
    numbers_str = re.findall(r'\d+', LTIME)
    numbers = [int(number) for number in numbers_str]
    
    total_time = 0

    for number, unit in zip(numbers, units):
        total_time += number * conversor[unit]
    
    return total_time


# %% --------------------------------------------------------------------------
# RENAME RAWDATA DICTIONARIES
# -----------------------------------------------------------------------------

MotColumnsRenames = {
    'Time(s)': 'Time',
    'MC SW Overview - Actual Position(mm)': 'Position',
    'MC SW Force Control - Measured Force(N)': 'Force',
    'MC SW Force Control - Target Force(N)': 'TargetForce',
    'LINMOT_MOVING_BOOL': 'Bool1',
    'LINMOT_UP_AND_DOWN_BOOL': 'Bool2'
}

DaqColumnsRenames = {
    'Time (s)': 'Time',
    'Voltage': 'Voltage',
    'LinMot_Enable': 'Bool1',
    'LinMot_Up_Down': 'Bool2'
}


# %% --------------------------------------------------------------------------
# LOAD MOTOR RAWDATA
# -----------------------------------------------------------------------------

def LoadMotorFile(ExpPath):
    '''
    Loads and processes a motor CSV file.
    
    Parameters
    ----------
    ExpPath : str
        Path to the experiment folder.
    
    Returns
    -------
    pd.DataFrame or None
        Processed motor data or None if there is an error.
    '''

    # Find files with CSV extension
    files = [f for f in os.listdir(ExpPath) if f.endswith('.csv')]
    if len(files) == 0:
        logger0.error(f'No motor data file found in {ExpPath}.')
        return None
    elif len(files) > 1:
        logger0.error(f'More than one motor data file found in {ExpPath}, the program cannot proceed.')
        return None

    # Define the MotorFile path
    MotorFile = os.path.join(ExpPath, files[0])
    MotorFile = os.path.normpath(MotorFile)

    try:
        dfMot = pd.read_csv(MotorFile, header=0, index_col=False,
                            delimiter=',', decimal='.')
    except Exception as e:
        logger0.error(f'Error reading Motor file {MotorFile}: {e}.')
        return None
    
    # Drop non-defined columns
    dropcols = [col for col in dfMot.columns if col not in MotColumnsRenames]
    dfMot = dfMot.drop(columns=dropcols)
    
    # Check if all defined columns exist
    for col in MotColumnsRenames.keys():
        if col not in dfMot.columns:
            logger0.error(f'Column {col} not found in {MotorFile}.')
            return None
    
    # Rename defined columns
    dfMot = dfMot.rename(columns=MotColumnsRenames)
    
    # Ensure columns have the correct data types
    dfMot = dfMot.astype({
        'Position': float,
        'Force': float,
        'TargetForce': float,
        'Bool1': int,
        'Bool2': int
    })
    
    # Corrections Time
    dfMot['Time'] = dfMot['Time'].apply(LTIME_to_seconds).astype(float)
    dfMot['Time'] -= dfMot['Time'].iloc[0]
    
    # Corrections Position
    dfMot['Position'] -= dfMot['Position'].min()
    
    # Corrections Force and TargetForce
    dfMot['Force'] = - dfMot['Force']
    dfMot['TargetForce'] = - dfMot['TargetForce']
    
    return dfMot


# %% --------------------------------------------------------------------------
# LOAD DAQ RAWDATA
# -----------------------------------------------------------------------------

def LoadDAQData(ExpPath):
    '''
    Loads and processes a DAQ pickle file.
    
    Parameters
    ----------
    ExpPath : str
        Path to the experiment folder.
    
    Returns
    -------
    pd.DataFrame or None
        Processed DAQ data or None if there is an error.
    '''

    # Read the DAQ Data
    dfDaq = merge_DAQ_data(ExpPath)
    
    # Drop non-defined columns:
    dropcols = [col for col in dfDaq.columns if col not in DaqColumnsRenames]
    dfDaq = dfDaq.drop(columns=dropcols)
    
    # Check if all defined columns exist
    for col in DaqColumnsRenames.keys():
        if col not in dfDaq.columns:
            logger0.error(f'Column {col} not found in the experiment folder {ExpPath}.')
            return
    
    # Rename defined columns
    dfDaq = dfDaq.rename(columns=DaqColumnsRenames)
    
    # Ensure columns have the correct data types
    dfDaq = dfDaq.astype({
        'Time': float,
        'Voltage': float,
        'Bool1': int,
        'Bool2': int
    })

    # Corrections Voltage (find the starting point of motor movement and reference voltage from this point)
    j = dfDaq['Bool1'].eq(1).idxmax()
    dfDaq['Voltage'] -= dfDaq['Voltage'][:j].mean()
    
    return dfDaq


# %% --------------------------------------------------------------------------
# FIND CYCLES FUNCTION
# -----------------------------------------------------------------------------

def FindCycles(df):
    '''
    Identifies start and end indices of operational cycles based on state changes.
    
    Parameters
    ----------
    dataframe : Pandas DataFrame
        Dataframe with two binary synchronization columns (Enable/Disable, Move_Up/Move_Down)
    
    Returns
    -------
    list of [start, end]
        List of cycles represented by their star and end indices.
    '''
    state_series = df['Bool1'] + df['Bool2']
    cycles = []
    prev_state = state_series.iloc[0]
    start = None
    
    for i, s in enumerate(state_series[1:], start=1):
        if s != prev_state:
            if s == 2:
                if start is not None:
                    cycles.append([start, i - 1])
                start = i
            elif s == 0 and start is not None:
                cycles.append([start, i - 1])
                start = None
            prev_state = s
        
    if start is not None:
        cycles.append([start, len(state_series) - 1])
    
    return cycles


# %% --------------------------------------------------------------------------
# LOAD AND SYNCHRONIZE RAWDATA FILES
# -----------------------------------------------------------------------------

def ExtractCycles(ExpPath):
    '''
    Loads and synchronizes Motor and DAQ data files, returning combined cycles data.
    
    Parameters
    ----------
    ExpPath : str
        Path to the experiment directory containing the data files.
    
    Returns
    -------
    Cycles : list[pd.DataFrame]
        List of DataFrames, each containing data for a single cycle.
    '''
    # Load Motor data
    dfMot = LoadMotorFile(ExpPath)
    if dfMot is None:
        return []
    
    # Load DAQ data
    dfDaq = LoadDAQData(ExpPath)
    if dfDaq is None:
        return []
    
    # Motor sampling rate
    MotFs = 1 / dfMot['Time'].diff().mean()
    logger0.info(f'Motor sampling rate: {MotFs}.')
    
    # DAQ sampling rate
    DaqFs = 1 / dfDaq['Time'].diff().mean()
    logger0.info(f'DAQ sampling rate: {DaqFs}.')

    ### Synchronize motor and DAQ data ###

    # Calculate the differences between adjacent values
    diff = dfMot['Bool1'].diff()

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

    # Filter Motor dataframe using the calculated indices and reset time reference
    dfMot = dfMot.loc[up_index : down_index].copy().reset_index(drop=True)
    dfMot['Time'] -= dfMot['Time'].iloc[0]

    # Adjust the motor timestamp
    dfMot_time = dfMot['Time'].iloc[-1] - dfMot['Time'].iloc[0]
    dfDaq_time = dfDaq['Time'].iloc[-1] - dfDaq['Time'].iloc[0]
    print("Adjusting DAQ and Motor timestamps")
    print("Motor time: ", dfMot_time)
    print("DAQ time: ", dfDaq_time)
    print("Difference: ", abs(dfMot_time - dfDaq_time), "seconds")
    dfMot['Time'] = dfMot['Time'] * (dfDaq_time / dfMot_time)

    # Synchronize dataframes
    [dfDaq, dfMot] = synchronize_dataframes([dfDaq, dfMot],
                                            time_col='Time',
                                            filter_time=False,
                                            binary_cols=['Bool1', 'Bool2'])

    # Restore the time index as a standard column
    for df in [dfMot, dfDaq]:
        df.index.name = 'Time'
        df.reset_index(inplace=True)

    # Estimate Velocity and Acceleration
    dfMot['Velocity'] = (dfMot['Position'].diff()/1000) / dfMot['Time'].diff()
    dfMot['Acceleration'] = dfMot['Velocity'].diff() / dfMot['Time'].diff()
                    
    # Finding cycles
    MotCycles = FindCycles(dfMot)
    DaqCycles = FindCycles(dfDaq)
    if len(MotCycles) != len(DaqCycles):
        logger0.info(f'Different number of cycles: Motor={len(MotCycles)}, DAQ={len(DaqCycles)}. Using minimum.')
    nCycles = min(len(MotCycles), len(DaqCycles))
    
    Cycles = []
    for idx in range(nCycles):

        # Obtain data from the idx cycle interval (we are iterating for each cycle interval)
        dfcyM = dfMot.iloc[MotCycles[idx][0] : MotCycles[idx][1] + 1].reset_index(drop=True)
        dfcyD = dfDaq.iloc[DaqCycles[idx][0] : DaqCycles[idx][1] + 1].reset_index(drop=True)

        # Find the first index where (State == 1) in the cycle interval for the Motor
        mask = dfcyM['Bool2'].eq(1)
        if mask.any():
            i = mask.idxmax()
        else:
            # Incomplete cycle
            continue

        # Find the first index where (State == 1) in the cycle interval for the DAQ
        mask = dfcyD['Bool2'].eq(1)
        if mask.any():
            j = mask.idxmax()
        else:
            # Incomplete cycle
            continue

        # Interpolate the data columns using DAQ sampling rate as the reference
        for col in dfcyM.columns:
            if col == 'Time' or col == 'State':
                continue

            # Create in the DAQ a new column to store the motor data
            dfcyD[col] = np.zeros(len(dfcyD), dtype=float)

            # Down-phase interpolation
            valid_mask_down = dfcyM[col].iloc[:i].notna()
            xp = dfcyM['Time'].iloc[:i][valid_mask_down].values
            fp = dfcyM[col].iloc[:i][valid_mask_down].values
            x = dfcyD['Time'].iloc[:j].values

            if np.all(np.diff(xp) > 0) and len(xp) > 1:
                dfcyD.loc[dfcyD.index[:j], col] = np.interp(x, xp, fp)

            # Up-phase interpolation
            valid_mask_up = dfcyM[col].iloc[i:].notna()
            xp = dfcyM['Time'].iloc[i:][valid_mask_up].values
            fp = dfcyM[col].iloc[i:][valid_mask_up].values
            x = dfcyD['Time'].iloc[j:].values

            if np.all(np.diff(xp) > 0) and len(xp) > 1:
                dfcyD.loc[dfcyD.index[j:], col] = np.interp(x, xp, fp)
        
        # Keep the resulting cycle
        Cycles.append(dfcyD)
    
    return Cycles
