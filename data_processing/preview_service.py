import pandas as pd
import json
import numpy as np
import scipy

try:
    from nptdms import TdmsFile

    HAS_NPTDMS = True
except Exception:
    HAS_NPTDMS = False

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from scipy.signal import find_peaks

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# --- LOADERS ---

def csv_to_dataframe(path: str) -> pd.DataFrame:
    """Load CSV file as pandas DataFrame."""
    return pd.read_csv(path)


def tdms_to_dataframe(path: str) -> pd.DataFrame:
    """Load TDMS file as pandas DataFrame, looking for 'Input 0' channel."""
    if not HAS_NPTDMS:
        raise RuntimeError('nptdms library is not installed')
    tdms = TdmsFile.read(path)

    target_channel = None
    for group in tdms.groups():
        for channel in group.channels():
            if channel.name == 'Input 0':
                target_channel = channel
                break
        if target_channel:
            break

    if not target_channel:
        raise ValueError('TDMS file does not contain "Input 0" channel')

    data = target_channel[:]
    return pd.DataFrame({'Input 0': data, 'index': range(len(data))})


# --- MATH HELPERS ---

def apply_gain_to_dataframe(df: pd.DataFrame, gain: float) -> pd.DataFrame:
    """Multiply numeric columns by gain, excluding the index column."""
    if gain is None:
        return df
    result = df.copy()
    for col in result.columns:
        if col.lower() == 'index':
            continue
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].astype(float) * gain
    return result


def calculate_power_dataframe(df: pd.DataFrame, req: float) -> pd.DataFrame:
    """Convert the primary data column into power values using P = V^2 / R."""
    if req is None or req == 0:
        raise ValueError('Invalid Req value for power calculation')
    plot_columns = [col for col in df.columns if col.lower() != 'index' and pd.api.types.is_numeric_dtype(df[col])]
    if not plot_columns:
        raise ValueError('No numeric voltage column found for power calculation')

    primary = plot_columns[0]
    power_series = df[primary].astype(float) ** 2 / req
    return pd.DataFrame({'Power': power_series})


# --- CALCULATION LOGIC (VPP & POWER) ---

def calculate_mean_vpp(df: pd.DataFrame, gain: float) -> float:
    """Core logic to detect peaks and calculate mean Vpp."""
    if not HAS_SCIPY:
        return 0.0

    df_gain = apply_gain_to_dataframe(df, gain)

    # Identify data column
    time_col = 'Time(s)' if 'Time(s)' in df_gain.columns else None
    plot_columns = [col for col in df_gain.columns if col.lower() != 'index' and col != time_col]

    if not plot_columns:
        return 0.0

    y_data = df_gain[plot_columns[0]].values

    # Dynamic peak detection
    dynamic_prominence = np.std(y_data) * 0.5
    peaks_idx, _ = find_peaks(y_data, prominence=dynamic_prominence, height=0.1)
    troughs_idx, _ = find_peaks(-y_data, prominence=dynamic_prominence, height=0.1)

    if len(peaks_idx) > 0 and len(troughs_idx) > 0:
        mean_max = np.mean(y_data[peaks_idx])
        mean_min = np.mean(y_data[troughs_idx])
        return float(mean_max - mean_min)
    return 0.0


def calculate_mean_power(df: pd.DataFrame, gain: float, req: float) -> float:
    """Return the mean power value of the primary numeric voltage column."""
    if gain is None or req is None or req == 0:
        return 0.0
    df_gain = apply_gain_to_dataframe(df, gain)
    power_df = calculate_power_dataframe(df_gain, req)
    return float(power_df['Power'].mean())


# --- FILE WRAPPERS ---

def calculate_mean_power_from_file(path: str, ext: str, gain: float, req: float) -> float:
    try:
        df = csv_to_dataframe(path) if ext == '.csv' else tdms_to_dataframe(path)
        return calculate_mean_power(df, gain, req)
    except Exception:
        return 0.0


def calculate_mean_vpp_from_file(path: str, ext: str, gain: float) -> float:
    try:
        df = csv_to_dataframe(path) if ext == '.csv' else tdms_to_dataframe(path)
        return calculate_mean_vpp(df, gain)
    except Exception:
        return 0.0


# --- PLOTTING ---

def create_plot_html(df: pd.DataFrame, title: str = 'Data Plot', downsample_percent: int = 80, gain: float = None,
                     plot_mode: str = 'voltage', req: float = None) -> str:
    """Main signal plot (Voltage or Power) with peak markers."""
    if not HAS_PLOTLY:
        raise RuntimeError('plotly library is not installed')

    if gain is not None:
        df = apply_gain_to_dataframe(df, gain)

    if plot_mode == 'power':
        if req is None:
            raise ValueError('Req value is required for power plot')
        df = calculate_power_dataframe(df, req)

    time_col = 'Time(s)' if 'Time(s)' in df.columns else None
    plot_columns = [col for col in df.columns if col.lower() != 'index' and col != time_col]

    if not plot_columns:
        raise ValueError('No data columns to plot')

    vpp_info = None
    if plot_mode == 'voltage' and HAS_SCIPY:
        primary_col = plot_columns[0]
        y_data = df[primary_col].values
        dinamic_prominence = np.std(y_data) * 0.5
        peaks_idx, _ = find_peaks(y_data, prominence=dinamic_prominence, height=0.1)
        troughs_idx, _ = find_peaks(-y_data, prominence=dinamic_prominence, height=0.1)

        if len(peaks_idx) > 0 and len(troughs_idx) > 0:
            mean_max = np.mean(y_data[peaks_idx])
            mean_min = np.mean(y_data[troughs_idx])
            vpp_info = {
                'x_peaks': df.loc[peaks_idx, time_col] if time_col else peaks_idx,
                'y_peaks': y_data[peaks_idx],
                'x_troughs': df.loc[troughs_idx, time_col] if time_col else troughs_idx,
                'y_troughs': y_data[troughs_idx],
                'mean_max': mean_max, 'mean_min': mean_min, 'vpp': mean_max - mean_min
            }

    # Downsampling
    original_length = len(df)
    if downsample_percent < 100:
        target_size = max(1, int(original_length * (downsample_percent / 100.0)))
        indices = np.linspace(0, original_length - 1, target_size, dtype=int)
        df = df.iloc[indices].copy()

    x_values = df[time_col] if time_col else None
    fig = go.Figure()

    for col in plot_columns:
        fig.add_trace(go.Scatter(x=x_values, y=df[col], mode='lines', name=col))

    if vpp_info:
        fig.add_trace(go.Scatter(x=vpp_info['x_peaks'], y=vpp_info['y_peaks'], mode='markers', name='Max',
                                 marker=dict(color='green')))
        fig.add_trace(go.Scatter(x=vpp_info['x_troughs'], y=vpp_info['y_troughs'], mode='markers', name='Min',
                                 marker=dict(color='red')))
        fig.add_hline(y=vpp_info['mean_max'], line_dash="dash", line_color="green")
        fig.add_hline(y=vpp_info['mean_min'], line_dash="dash", line_color="red")
        title += f' | Mean Vpp: {vpp_info["vpp"]:.3f}V'

    fig.update_layout(title=title, xaxis_title='Time/Index', yaxis_title=plot_mode.capitalize(), height=600)
    return fig.to_html(include_plotlyjs='cdn', div_id='plot')


def create_mean_power_vs_req_plot(grouped_data: dict, title: str = 'Mean Power vs Resistance') -> str:
    """Scatter plot for Power vs Resistance grouped by TribuId."""
    if not HAS_PLOTLY or not grouped_data:
        return '<p>No data available</p>'

    fig = go.Figure()

    for tribu_id, data_points in grouped_data.items():
        if not data_points:
            continue
        # Sort points by X-axis (Req) so the lines connect properly left-to-right
        data_points.sort(key=lambda x: x[0])
        reqs, powers = zip(*data_points)
        fig.add_trace(go.Scatter(x=reqs, y=powers, mode='markers+lines', name=f'{tribu_id}'))

    fig.update_layout(title=title, xaxis_title='Resistance (Req) [ohms]', yaxis_title='Mean Power [W]', height=400)
    return fig.to_html(include_plotlyjs='cdn', div_id='mean_power_plot')


def create_mean_vpp_vs_req_plot(grouped_data: dict, title: str = 'Mean Vpp vs Resistance') -> str:
    """Scatter plot for Vpp vs Resistance grouped by TribuId."""
    if not HAS_PLOTLY or not grouped_data:
        return '<p>No data available</p>'

    fig = go.Figure()

    for tribu_id, data_points in grouped_data.items():
        if not data_points:
            continue
        data_points.sort(key=lambda x: x[0])
        reqs, vpps = zip(*data_points)
        fig.add_trace(go.Scatter(x=reqs, y=vpps, mode='markers+lines', name=f'{tribu_id}'))

    fig.update_layout(title=title, xaxis_title='Resistance (Req) [ohms]', yaxis_title='Mean Vpp [V]', height=400)
    return fig.to_html(include_plotlyjs='cdn', div_id='mean_vpp_plot')


def has_tdms_support() -> bool: return HAS_NPTDMS


def has_plotly_support() -> bool: return HAS_PLOTLY