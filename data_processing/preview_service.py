import pandas as pd
import json
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
def apply_lowpass_filter(data: np.ndarray, cutoff: float = 0.9, order: int = 2) -> np.ndarray:
    """
    Applies a Butterworth lowpass filter to smooth out high-frequency noise.
    'cutoff' is relative to Nyquist frequency.
    """
    try:
        b, a = butter(order, cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    except Exception:
        return data


def csv_to_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def tdms_to_dataframe(path: str) -> pd.DataFrame:
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
    # FIX: Calculate actual time in seconds using the channel properties
    # Defaulting to 1000Hz (0.001s) if sampling information is missing
    dt = target_channel.properties.get('wf_increment')
    if dt is None:
        # Si no existe wf_increment, buscamos sampling_rate
        fs = target_channel.properties.get('sampling_rate', 1000.0)  # 1000Hz por defecto
        dt = 1.0 / fs
    length = len(data)
    # Creamos el tiempo
    time_s = np.arange(length) * dt
    df = pd.DataFrame({'Input 0': data, 'Time(s)': time_s})
    return df


# --- MATH HELPERS ---

def apply_gain_to_dataframe(df: pd.DataFrame, gain: float) -> pd.DataFrame:
    if gain is None:
        return df
    result = df.copy()
    for col in result.columns:
        # EXCLUIR TIEMPO Y CUALQUIER VARIANTE DE INDEX
        if col.lower() in ['index', 'time(s)', 'time']:
            continue
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].astype(float) / gain
    return result


def calculate_power_dataframe(df: pd.DataFrame, req: float) -> pd.DataFrame:
    if req is None or req == 0:
        raise ValueError('Invalid Req value for power calculation')
    plot_columns = [col for col in df.columns if col.lower() != 'index' and pd.api.types.is_numeric_dtype(df[col])]
    if not plot_columns:
        raise ValueError('No numeric voltage column found for power calculation')

    primary = plot_columns[0]
    power_series = df[primary].astype(float) ** 2 / req
    # MANTENER EL TIEMPO SI EXISTE
    new_df = pd.DataFrame({'Power': power_series})
    if 'Time(s)' in df.columns:
        new_df['Time(s)'] = df['Time(s)'].values
    return new_df


# --- CENTRALIZED DETECTION LOGIC ---

def get_signal_peaks(y_raw: np.ndarray):
    """
    Shared logic for peak detection to ensure consistency between Vpp calculation and Plotly markers.
    """
    if not HAS_SCIPY:
        return None, None, 0.0, 0.0, 0.0

    # 1. Uniform Smoothing
    y_smooth = apply_lowpass_filter(y_raw, cutoff=0.1) #cutoff=1 no filter. cutoff=0 filters all
    std_val = np.std(y_smooth)

    # 2. Unified Parameters (Sigma-based thresholding + Minimum distance)
    # distance=300 helps avoid multiple detections in noisy wave cycles
    params = {
        'height': np.percentile(y_smooth, 95),  # Focus on the top 5% of signal values
        'prominence': np.std(y_smooth) * 2,  # Higher multiplier ignores noise
        'distance': 100  # Prevent 'double-counting' a single spike
    }

    peaks_idx, _ = find_peaks(y_smooth, **params)
    troughs_idx, _ = find_peaks(-y_smooth, **params)

    if len(peaks_idx) > 0 and len(troughs_idx) > 0:
        # We find indices on smoothed signal but average raw values for accuracy
        mean_max = np.mean(y_raw[peaks_idx])
        mean_min = np.mean(y_raw[troughs_idx])
        return peaks_idx, troughs_idx, mean_max, mean_min, float(mean_max - mean_min)

    return None, None, 0.0, 0.0, 0.0


# --- CALCULATION WRAPPERS ---

def calculate_mean_vpp(df: pd.DataFrame, gain: float) -> float:
    df_gain = apply_gain_to_dataframe(df, gain)
    time_col = 'Time(s)' if 'Time(s)' in df_gain.columns else None
    plot_columns = [col for col in df_gain.columns if col.lower() != 'index' and col != time_col]

    if not plot_columns:
        return 0.0

    raw_y = df_gain[plot_columns[0]].values
    _, _, _, _, vpp = get_signal_peaks(raw_y)
    return vpp


def calculate_mean_power(df: pd.DataFrame, gain: float, req: float) -> float:
    if gain is None or req is None or req == 0:
        return 0.0
    df_gain = apply_gain_to_dataframe(df, gain)
    power_df = calculate_power_dataframe(df_gain, req)
    return float(power_df['Power'].mean())


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
        raw_y = df[primary_col].values

        # Use centralized logic
        p_idx, t_idx, m_max, m_min, vpp = get_signal_peaks(raw_y)

        if p_idx is not None:
            vpp_info = {
                'x_peaks': df.loc[p_idx, time_col] if time_col else p_idx,
                'y_peaks': raw_y[p_idx],
                'x_troughs': df.loc[t_idx, time_col] if time_col else t_idx,
                'y_troughs': raw_y[t_idx],
                'mean_max': m_max, 'mean_min': m_min, 'vpp': vpp
            }

    # Downsampling for visualization (done AFTER analysis)
    original_length = len(df)
    if downsample_percent < 100:
        target_size = max(1, int(original_length * (downsample_percent / 100.0)))
        indices = np.linspace(0, original_length - 1, target_size, dtype=int)
        df_plot = df.iloc[indices].copy()
    else:
        df_plot = df

    x_values = df_plot[time_col] if time_col else None
    fig = go.Figure()

    for col in plot_columns:
        fig.add_trace(go.Scatter(x=x_values, y=df_plot[col], mode='lines', name=col))

    if vpp_info:
        fig.add_trace(go.Scatter(x=vpp_info['x_peaks'], y=vpp_info['y_peaks'], mode='markers', name='Max',
                                 marker=dict(color='green', size=8)))
        fig.add_trace(go.Scatter(x=vpp_info['x_troughs'], y=vpp_info['y_troughs'], mode='markers', name='Min',
                                 marker=dict(color='red', size=8)))
        fig.add_hline(y=vpp_info['mean_max'], line_dash="dash", line_color="green")
        fig.add_hline(y=vpp_info['mean_min'], line_dash="dash", line_color="red")
        title += f' | Mean Vpp: {vpp_info["vpp"]:.3f}V'

    # Usar el nombre de la columna si existe, de lo contrario poner 'Index'
    x_label = time_col if time_col else 'Index'
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Voltage(V)", height=600)
    return fig.to_html(include_plotlyjs='cdn', div_id='plot')


# --- Multiple Y Axis plotting (V,Pos,F) ---
def create_combined_motor_daq_plot(daq_df, motor_df, title, downsample_percent=100, gain=None):
    if not HAS_PLOTLY:
        raise RuntimeError('plotly library is not installed')

    if gain is not None:
        daq_df = apply_gain_to_dataframe(daq_df, gain)

    # --- Helper to find columns by partial name ---
    def find_col(df, keywords):
        for col in df.columns:
            if any(k.lower() in col.lower() for k in keywords) and "unnamed" not in col.lower():
                return col
        return None

    # 1. Identify Columns
    v_col = find_col(daq_df, ['input 0', 'voltage'])
    p_col = find_col(motor_df, ['position', 'actual position'])
    f_col = find_col(motor_df, ['force', 'measured force'])

    # Identify Time columns or create them if missing
    d_time = daq_df['Time(s)'] if 'Time(s)' in daq_df.columns else np.arange(len(daq_df))
    m_time = motor_df['Time(s)'] if 'Time(s)' in motor_df.columns else None

    # If Motor CSV has no time, but recording duration is same as DAQ:
    if m_time is None:
        duration = d_time.max() if len(d_time) > 0 else 1
        m_time = np.linspace(0, duration, len(motor_df))

    # 2. Crear Subplots: 3 filas, 1 columna
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Voltage (V)", "Position (mm)", "Force (N)")
    )

    # 3. Añadir Voltaje (Fila 1)
    if v_col:
        fig.add_trace(
            go.Scatter(x=d_time, y=daq_df[v_col], name="Voltage", line=dict(color='blue')),
            row=1, col=1
        )

    # 4. Añadir Posición (Fila 2)
    if p_col:
        fig.add_trace(
            go.Scatter(x=m_time, y=motor_df[p_col], name="Position", line=dict(color='orange')),
            row=2, col=1
        )

    # 5. Añadir Fuerza (Fila 3)
    if f_col:
        fig.add_trace(
            go.Scatter(x=m_time, y=motor_df[f_col], name="Force", line=dict(color='green')),
            row=3, col=1
        )

    # 6. Configuración de Layout
    fig.update_layout(
        title=title,
        height=900,  # Aumentamos la altura para que los 3 se vean bien
        template="plotly_white",
        showlegend=False,
        hovermode="x unified"
    )

    # Etiquetas de ejes Y individuales
    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(title_text="mm", row=2, col=1)
    fig.update_yaxes(title_text="N", row=3, col=1)

    # Etiqueta de eje X solo al final
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)

    return fig.to_html(include_plotlyjs='cdn', div_id='plot')

def create_mean_power_vs_req_plot(grouped_data: dict, title: str = 'Mean Power vs Resistance') -> str:
    if not HAS_PLOTLY or not grouped_data:
        return '<p>No data available</p>'
    fig = go.Figure()
    for tribu_id, data_points in grouped_data.items():
        if not data_points: continue
        data_points.sort(key=lambda x: x[0])
        reqs, powers = zip(*data_points)
        fig.add_trace(go.Scatter(x=reqs, y=powers, mode='markers+lines', name=f'{tribu_id}'))
    fig.update_layout(title=title, xaxis_title='Resistance (Req) [ohms]', yaxis_title='Mean Power [W]', height=400)
    return fig.to_html(include_plotlyjs='cdn', div_id='mean_power_plot')


def create_mean_vpp_vs_req_plot(grouped_data: dict, title: str = 'Mean Vpp vs Resistance') -> str:
    if not HAS_PLOTLY or not grouped_data:
        return '<p>No data available</p>'
    fig = go.Figure()
    for tribu_id, data_points in grouped_data.items():
        if not data_points: continue
        data_points.sort(key=lambda x: x[0])
        reqs, vpps = zip(*data_points)
        fig.add_trace(go.Scatter(x=reqs, y=vpps, mode='markers+lines', name=f'{tribu_id}'))
    fig.update_layout(title=title, xaxis_title='Resistance (Req) [ohms]', yaxis_title='Mean Vpp [V]', height=400)
    return fig.to_html(include_plotlyjs='cdn', div_id='mean_vpp_plot')


def has_tdms_support() -> bool: return HAS_NPTDMS


def has_plotly_support() -> bool: return HAS_PLOTLY