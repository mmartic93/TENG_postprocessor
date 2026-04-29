import pandas as pd
import numpy as np
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
    dt = target_channel.properties.get('wf_increment')
    if dt is None:
        fs = target_channel.properties.get('sampling_rate', 1000.0)
        dt = 1.0 / fs

    length = len(data)
    time_s = np.arange(length) * dt
    df = pd.DataFrame({'Input 0': data, 'Time(s)': time_s})
    return df


# --- MATH HELPERS ---

def apply_gain_to_dataframe(df: pd.DataFrame, gain: float) -> pd.DataFrame:
    if gain is None:
        return df
    result = df.copy()
    for col in result.columns:
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
    new_df = pd.DataFrame({'Power': power_series})
    if 'Time(s)' in df.columns:
        new_df['Time(s)'] = df['Time(s)'].values
    return new_df


# --- CENTRALIZED DETECTION LOGIC ---

def get_signal_peaks(y_raw: np.ndarray):
    """Shared logic for voltage peak detection."""
    if not HAS_SCIPY:
        return None, None, 0.0, 0.0, 0.0

    y_smooth = apply_lowpass_filter(y_raw, cutoff=0.1)
    params = {
        'height': np.percentile(y_smooth, 95),
        'prominence': np.std(y_smooth) * 2,
        'distance': 100
    }

    peaks_idx, _ = find_peaks(y_smooth, **params)
    troughs_idx, _ = find_peaks(-y_smooth, **params)

    if len(peaks_idx) > 0 and len(troughs_idx) > 0:
        mean_max = np.mean(y_raw[peaks_idx])
        mean_min = np.mean(y_raw[troughs_idx])
        return peaks_idx, troughs_idx, mean_max, mean_min, float(mean_max - mean_min)

    return None, None, 0.0, 0.0, 0.0


def get_power_peaks(power_raw: np.ndarray):
    """Detect peaks specifically for power signals (unipolar) and average the last 10 cycles."""
    if not HAS_SCIPY:
        return None, 0.0

    y_smooth = apply_lowpass_filter(power_raw, cutoff=0.1)
    params = {
        'height': np.percentile(y_smooth, 90),
        'prominence': np.std(y_smooth) * 1.5,
        'distance': 50
    }
    peaks_idx, _ = find_peaks(y_smooth, **params)

    if len(peaks_idx) > 0:
        # Calculate mean of LAST 10 cycles (peaks)
        last_peaks = peaks_idx[-10:]
        mean_peak_power = np.mean(power_raw[last_peaks])
        return peaks_idx, float(mean_peak_power)

    return None, 0.0


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


def calculate_peak_power(df: pd.DataFrame, gain: float, req: float) -> float:
    """Calculates average peak power over the last 10 cycles."""
    if gain is None or req is None or req == 0:
        return 0.0
    df_gain = apply_gain_to_dataframe(df, gain)
    power_df = calculate_power_dataframe(df_gain, req)
    _, mean_peak = get_power_peaks(power_df['Power'].values)
    return mean_peak


def calculate_mean_power_from_file(path: str, ext: str, gain: float, req: float) -> float:
    try:
        df = csv_to_dataframe(path) if ext == '.csv' else tdms_to_dataframe(path)
        return calculate_mean_power(df, gain, req)
    except Exception:
        return 0.0


def calculate_peak_power_from_file(path: str, ext: str, gain: float, req: float) -> float:
    try:
        df = csv_to_dataframe(path) if ext == '.csv' else tdms_to_dataframe(path)
        return calculate_peak_power(df, gain, req)
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

    primary_col = plot_columns[0]
    raw_y = df[primary_col].values
    analysis_info = None

    if plot_mode == 'voltage' and HAS_SCIPY:
        p_idx, t_idx, m_max, m_min, vpp = get_signal_peaks(raw_y)
        if p_idx is not None:
            analysis_info = {
                'x_peaks': df.loc[p_idx, time_col] if time_col else p_idx,
                'y_peaks': raw_y[p_idx],
                'x_troughs': df.loc[t_idx, time_col] if time_col else t_idx,
                'y_troughs': raw_y[t_idx],
                'lines': [('Max', m_max, 'green'), ('Min', m_min, 'red')],
                'label': f' | Mean Vpp: {vpp:.3f}V'
            }
    elif plot_mode == 'power' and HAS_SCIPY:
        p_idx, mean_peak = get_power_peaks(raw_y)
        if p_idx is not None:
            analysis_info = {
                'x_peaks': df.loc[p_idx, time_col] if time_col else p_idx,
                'y_peaks': raw_y[p_idx],
                'lines': [('Mean Peak (Last 10)', mean_peak, 'orange')],
                'label': f' | Avg Peak Power (last 10): {mean_peak:.4g} W'
            }

    # Downsampling for visualization
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

    if analysis_info:
        fig.add_trace(go.Scatter(x=analysis_info['x_peaks'], y=analysis_info['y_peaks'], mode='markers', name='Peaks',
                                 marker=dict(color='green', size=8)))
        if 'x_troughs' in analysis_info:
            fig.add_trace(
                go.Scatter(x=analysis_info['x_troughs'], y=analysis_info['y_troughs'], mode='markers', name='Troughs',
                           marker=dict(color='red', size=8)))

        for name, val, color in analysis_info['lines']:
            fig.add_hline(y=val, line_dash="dash", line_color=color, annotation_text=name)

        title += analysis_info['label']

    x_label = time_col if time_col else 'Index'
    y_label = "Power (W)" if plot_mode == 'power' else "Voltage (V)"
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=600)

    return fig.to_html(include_plotlyjs='cdn', div_id='plot')


def create_combined_motor_daq_plot(daq_df, motor_df, title, downsample_percent=100, gain=None):
    if not HAS_PLOTLY:
        raise RuntimeError('plotly library is not installed')

    if gain is not None:
        daq_df = apply_gain_to_dataframe(daq_df, gain)

    def find_col(df, keywords):
        for col in df.columns:
            if any(k.lower() in col.lower() for k in keywords) and "unnamed" not in col.lower():
                return col
        return None

    v_col = find_col(daq_df, ['input 0', 'voltage'])
    p_col = find_col(motor_df, ['position', 'actual position'])
    f_col = find_col(motor_df, ['force', 'measured force'])

    d_time = daq_df['Time(s)'] if 'Time(s)' in daq_df.columns else np.arange(len(daq_df))

    if 'Time(s)' in motor_df.columns:
        m_time = motor_df['Time(s)']
    else:
        duration = d_time.max() if len(d_time) > 0 else 1
        m_time = np.linspace(0, duration, len(motor_df))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Voltage (V)", "Position (mm)", "Force (N)")
    )

    if v_col:
        fig.add_trace(go.Scatter(x=d_time, y=daq_df[v_col], name="Voltage", line=dict(color='blue')), row=1, col=1)
    if p_col:
        fig.add_trace(go.Scatter(x=m_time, y=motor_df[p_col], name="Position", line=dict(color='orange')), row=2, col=1)
    if f_col:
        fig.add_trace(go.Scatter(x=m_time, y=motor_df[f_col], name="Force", line=dict(color='green')), row=3, col=1)

    fig.update_layout(
        title=title, height=900, template="plotly_white",
        showlegend=False, hovermode="x unified"
    )

    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(title_text="mm", row=2, col=1)
    fig.update_yaxes(title_text="N", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)

    return fig.to_html(include_plotlyjs='cdn', div_id='plot')


def create_mean_power_vs_req_plot(grouped_power: dict, grouped_peak_power: dict = None,
                                  title: str = 'Power Analysis vs Resistance') -> str:
    if not HAS_PLOTLY or not grouped_power:
        return '<p>No data available</p>'

    # Creating subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for tribu_id, points in grouped_power.items():
        if not points: continue
        points.sort(key=lambda x: x[0])
        reqs, powers = zip(*points)

        # Primary axis (Mean Power) - Solid Line
        fig.add_trace(go.Scatter(x=reqs, y=powers, mode='markers+lines', name=f'{tribu_id} (Mean)'), secondary_y=False)

        # Secondary axis (Peak Power) - Dashed Line
        if grouped_peak_power and tribu_id in grouped_peak_power:
            peak_points = grouped_peak_power[tribu_id]
            peak_points.sort(key=lambda x: x[0])
            p_reqs, p_peaks = zip(*peak_points)
            fig.add_trace(go.Scatter(x=p_reqs, y=p_peaks, mode='markers+lines', line=dict(dash='dash'),
                                     name=f'{tribu_id} (Peak)'), secondary_y=True)

    fig.update_layout(title=title, xaxis_title='Resistance (Req) [ohms]', height=500)
    fig.update_yaxes(title_text="Mean Power [W]", secondary_y=False)
    fig.update_yaxes(title_text="Avg Peak Power (Last 10 cycles) [W]", secondary_y=True)

    return fig.to_html(include_plotlyjs='cdn', div_id='mean_power_plot')


def create_mean_vpp_vs_req_plot(grouped_data: dict, title: str = 'Mean Vpp vs Resistance') -> str:
    """This function remains untouched, takes a single dict of grouped_data"""
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


def create_optimal_power_plot(optimal_points: list, title: str = 'Optimal Power per TribuId') -> str:
    """Creates a bar chart comparing the maximum mean power achieved by each TribuId."""
    if not HAS_PLOTLY or not optimal_points:
        return ''

    optimal_points.sort(key=lambda x: str(x['tribu_id']))
    tribu_ids = [str(p['tribu_id']) for p in optimal_points]
    max_powers = [p['max_power'] for p in optimal_points]
    req_labels = [f"Req: {p['req']} Ω" for p in optimal_points]

    fig = go.Figure(data=[
        go.Bar(
            x=tribu_ids,
            y=max_powers,
            text=req_labels,
            textposition='auto',
            marker=dict(color='royalblue'),
            hovertemplate="<b>TribuId: %{x}</b><br>Max Power: %{y:.4g} W<br>%{text}<extra></extra>"
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title='TribuId',
        yaxis_title='Max Mean Power [W]',
        height=450,
        template="plotly_white",
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig.to_html(include_plotlyjs='cdn', div_id='optimal_power_plot')


def has_tdms_support() -> bool: return HAS_NPTDMS


def has_plotly_support() -> bool: return HAS_PLOTLY