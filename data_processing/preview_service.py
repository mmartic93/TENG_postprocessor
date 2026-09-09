import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from plotly.subplots import make_subplots
from typing import Union, List
import json, os
import plotly.graph_objects as go
from scipy.signal import find_peaks, iirnotch

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


def infer_sampling_rate(time_values: np.ndarray) -> float:
    if time_values is None or len(time_values) < 2:
        raise ValueError('At least two time samples are required to infer sampling rate')
    dt = np.diff(time_values.astype(float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError('Could not infer a valid positive sampling period from time axis')
    dt_median = float(np.median(dt))
    if dt_median <= 0:
        raise ValueError('Invalid sampling period inferred from time axis')
    return 1.0 / dt_median


def apply_notch_filter(data: np.ndarray, fs: float, notch_freq: float = 50.0, quality_factor: float = 30.0) -> np.ndarray:
    if fs <= 0:
        raise ValueError('Sampling frequency must be > 0')
    if notch_freq <= 0:
        raise ValueError('Notch frequency must be > 0')
    if quality_factor <= 0:
        raise ValueError('Notch quality factor must be > 0')

    nyquist = fs / 2.0
    normalized_w0 = notch_freq / nyquist
    if normalized_w0 <= 0 or normalized_w0 >= 1:
        raise ValueError(f'Notch frequency must be between 0 and Nyquist ({nyquist:.4g} Hz)')

    b, a = iirnotch(normalized_w0, quality_factor)
    return filtfilt(b, a, data.astype(float))


def _parse_notch_list(value) -> List[float]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(',') if part.strip()]
    elif isinstance(value, (list, tuple, np.ndarray)):
        raw_values = [str(part).strip() for part in value if str(part).strip()]
    else:
        raw_values = [str(value).strip()] if str(value).strip() else []

    parsed: List[float] = []
    for raw in raw_values:
        parsed.append(float(raw))
    return parsed


def resolve_notch_parameters(notch_params: dict = None) -> tuple[List[float], List[float]]:
    frequencies = _parse_notch_list(notch_params.get('frequency', 50.0)) if notch_params else [50.0]
    q_values = _parse_notch_list(notch_params.get('q', 30.0)) if notch_params else [30.0]

    if not frequencies:
        frequencies = [50.0]
    if not q_values:
        q_values = [30.0]

    if len(q_values) == 1 and len(frequencies) > 1:
        q_values = q_values * len(frequencies)
    elif len(q_values) != len(frequencies):
        raise ValueError('Q factor count must be 1 or equal to frequency count')

    for freq in frequencies:
        if freq <= 0:
            raise ValueError('Notch frequency must be > 0')
    for q in q_values:
        if q <= 0:
            raise ValueError('Notch quality factor must be > 0')
    return frequencies, q_values


def apply_notch_filter_chain(data: np.ndarray, fs: float, notch_params: dict = None) -> tuple[np.ndarray, List[float], List[float]]:
    frequencies, q_values = resolve_notch_parameters(notch_params)
    filtered = data.astype(float)
    for freq, q in zip(frequencies, q_values):
        filtered = apply_notch_filter(filtered, fs, freq, q)
    return filtered, frequencies, q_values


# --- MATH HELPERS ---

def apply_gain_to_dataframe(df: pd.DataFrame, exp_path, gain: float) -> pd.DataFrame:
    result = df.copy()

    # Apply voltage divisor gain
    if gain:
        if 'Voltage' in result.columns and pd.api.types.is_numeric_dtype(result['Voltage']):
            result['Voltage'] = result['Voltage'].astype(float) / gain

    # Apply Analog 2V range conversion factor from Keithley
    json_path = os.path.join(exp_path, "experiment_metadata.json")
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
    for task in json_dict["DAQTasks"]:
        for channel in task["DAQ_CHANNELS"]:
            conversion_factor = task["DAQ_CHANNELS"][channel]["conversion_factor"]
            if conversion_factor:
                result[channel] *= conversion_factor

    return result


def calculate_power_dataframe(df: pd.DataFrame, req: float) -> pd.DataFrame:
    if req is None or req == 0:
        raise ValueError('Invalid Req value for power calculation')
    plot_columns = [col for col in df.columns if col.lower() != 'index' and pd.api.types.is_numeric_dtype(df[col])]
    if not plot_columns:
        raise ValueError('No numeric columns found for power calculation')

    primary = None
    for column in plot_columns:
        if column.lower() == "voltage":
            primary = column
            break
    if not primary:
        raise Exception("Voltage column not found")

    power_series = df[primary].astype(float) ** 2 / req
    new_df = pd.DataFrame({'Power': power_series})
    if 'Time' in df.columns:
        new_df['Time'] = df['Time'].values
    return new_df


def find_primary_signal_column(df: pd.DataFrame, mode: str = 'voltage') -> Union[str, None]:
    if mode == 'current':
        preferred = ['current', 'input 1', 'isc']
    else:
        preferred = ['voltage', 'input 0', 'voc']

    for col in df.columns:
        low = str(col).lower()
        if any(key in low for key in preferred):
            return col

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and str(col).lower() != 'time']
    return numeric_cols[0] if numeric_cols else None


# --- CENTRALIZED DETECTION LOGIC ---

def get_signal_peaks(y_raw: np.ndarray, custom_params: dict = None, cutoff: float = 0.1):
    """Shared logic for voltage peak detection with optional custom tuning."""

    # 1. CRITICAL: Extract cutoff from custom_params BEFORE filtering
    if custom_params and custom_params.get('cutoff') is not None:
        cutoff = float(custom_params['cutoff'])

    y_smooth = apply_lowpass_filter(y_raw, cutoff=cutoff)

    # 2. Set dynamic defaults
    params = {
        'height': np.percentile(y_smooth, 95),
        'prominence': np.std(y_smooth) * 2,
        'distance': 100
    }

    # 3. Override with custom values (filtering out Nones)
    if custom_params:
        params.update({k: v for k, v in custom_params.items() if v is not None})

    # 4. Prepare separate params for Troughs to avoid the 'height' conflict
    # We remove height for troughs unless specifically handled
    search_params_peaks = {k: v for k, v in params.items() if k != 'cutoff'}
    search_params_troughs = {k: v for k, v in search_params_peaks.items() if k != 'height'}

    peaks_idx, _ = find_peaks(y_smooth, **search_params_peaks)
    troughs_idx, _ = find_peaks(-y_smooth, **search_params_troughs)

    if len(peaks_idx) > 0 and len(troughs_idx) > 0:
        mean_max = np.mean(y_raw[peaks_idx])
        mean_min = np.mean(y_raw[troughs_idx])
        return peaks_idx, troughs_idx, mean_max, mean_min, float(mean_max - mean_min)

    return None, None, 0.0, 0.0, 0.0


def get_power_peaks(power_raw: np.ndarray, custom_params: dict = None, cutoff: float = 0.1):
    """Detect peaks specifically for power signals with optional custom tuning."""

    y_smooth = apply_lowpass_filter(power_raw, cutoff=cutoff)

    # Default parameters
    params = {
        'height': np.percentile(y_smooth, 90),
        'prominence': np.std(y_smooth) * 1.5,
        'distance': 50
    }

    if custom_params:
        # Filter out Nones and update
        params.update({k: v for k, v in custom_params.items() if v is not None})

    # Remove 'cutoff' from params because find_peaks does not accept it
    search_params = {k: v for k, v in params.items() if k != 'cutoff'}

    peaks_idx, _ = find_peaks(y_smooth, **search_params)

    if len(peaks_idx) > 0:
        last_peaks = peaks_idx[-10:]
        mean_peak_power = np.mean(power_raw[last_peaks])
        return peaks_idx, float(mean_peak_power)

    return None, 0.0


def get_plateau_peaks(y_vals, threshold_percentile=80, cutoff=0.05):
    if len(y_vals) == 0:
        return None, None, 0, 0, 0

    # 1. Determine if the signal is primarily positive or negative
    # We compare the max absolute value to the raw max to find polarity
    raw_max = np.max(y_vals)
    raw_min = np.min(y_vals)

    if abs(raw_min) > abs(raw_max):
        # Negative plateau logic
        threshold = np.percentile(y_vals, 100 - threshold_percentile)
        plateau_indices = np.where(y_vals <= threshold)[0]
    else:
        # Positive plateau logic
        threshold = np.percentile(y_vals, threshold_percentile)
        plateau_indices = np.where(y_vals >= threshold)[0]

    if len(plateau_indices) == 0:
        return None, None, 0, 0, 0

    # 2. Filter out values too close to zero (the baseline)
    # This prevents the baseline from being detected as a plateau
    abs_max = max(abs(raw_max), abs(raw_min))
    significant_indices = [i for i in plateau_indices if abs(y_vals[i]) > (abs_max * cutoff)]

    if not significant_indices:
        return None, None, 0, 0, 0

    significant_indices = np.array(significant_indices)
    mean_plateau_value = np.mean(y_vals[significant_indices])

    # For Voc, we return the mean of the plateau as the "max"
    return significant_indices, None, mean_plateau_value, 0, 0

# --- CALCULATION WRAPPERS ---

def calculate_mean_vpp(df: pd.DataFrame, exp_path: str, gain: float, peak_params: dict = None) -> float:
    df_gain = apply_gain_to_dataframe(df, exp_path, gain)
    time_col = 'Time' if 'Time' in df_gain.columns else None
    plot_columns = [col for col in df_gain.columns if col.lower() != 'index' and col != time_col]

    if not plot_columns:
        return 0.0

    primary = None
    for column in plot_columns:
        if column.lower() == "voltage":
            primary = column
            break
    if not primary:
        raise Exception("Voltage column not found")

    raw_y = df_gain[primary].values
    _, _, _, _, vpp = get_signal_peaks(raw_y)
    return vpp


def calculate_mean_power(df: pd.DataFrame, exp_path: str, gain: float, req: float, peak_params: dict = None) -> float:
    if gain is None or req is None or req == 0:
        return 0.0
    df_gain = apply_gain_to_dataframe(df, exp_path, gain)
    power_df = calculate_power_dataframe(df_gain, req)
    return float(power_df['Power'].mean())


def calculate_peak_power(df: pd.DataFrame, exp_path: str, gain: float, req: float, peak_params: dict = None) -> float:
    """Calculates average peak power over the last 10 cycles."""
    if gain is None or req is None or req == 0:
        return 0.0
    df_gain = apply_gain_to_dataframe(df, exp_path, gain)
    power_df = calculate_power_dataframe(df_gain, req)
    _, mean_peak = get_power_peaks(power_df['Power'].values,custom_params=peak_params)
    return mean_peak


# --- PLOTTING ---

def create_plot_html(df: pd.DataFrame, exp_path: str, title: str = 'Data Plot', downsample_percent: int = 80,
                     gain: float = None, plot_mode: str = 'voltage', req: float = None,
                     peak_params: dict = None, include_graphs: list = None, notch_params: dict = None,
                     signal_mode: str = 'voltage', cycle_markers: list = None,
                     use_converted_signal: bool = False) -> str:

    if use_converted_signal:
        df = apply_gain_to_dataframe(df, exp_path, gain)

    if plot_mode == 'power':
        if req is None:
            raise ValueError('Req value is required for power plot')
        df = calculate_power_dataframe(df, req)

    if 'Time (s)' in df.columns:
        time_col = 'Time (s)'
    elif 'Time' in df.columns:
        time_col = 'Time'
    else:
        time_col = None
    plot_columns = [col for col in df.columns if col.lower() != 'index' and col != time_col]
    signal_col = find_primary_signal_column(df, signal_mode)
    if signal_col is None:
        raise ValueError('No numeric signal columns found for plotting')
    primary_col = signal_col
    raw_y = df[primary_col].values
    analysis_info = None
    signal_label = 'Current' if signal_mode == 'current' else 'Voltage'
    signal_unit = 'A' if signal_mode == 'current' else 'V'

    # Extraer cutoff si existe en peak_params
    cutoff_val = peak_params.get('cutoff', 0.1) if peak_params else 0.1
    notch_enabled = bool(notch_params and notch_params.get('enabled'))
    show_raw_signal = bool(notch_params.get('show_raw_signal', True)) if notch_params else True
    show_filtered_signal = bool(notch_params.get('show_filtered_signal', True)) if notch_params else True

    if plot_mode == 'voltage':
        p_idx, t_idx, m_max, m_min, vpp = get_signal_peaks(raw_y, custom_params=peak_params, cutoff=cutoff_val)
        if p_idx is not None:
            analysis_info = {
                'x_peaks': df.loc[p_idx, time_col] if time_col else p_idx,
                'y_peaks': raw_y[p_idx],
                'x_troughs': df.loc[t_idx, time_col] if time_col else t_idx,
                'y_troughs': raw_y[t_idx],
                'lines': [('Max', m_max, 'green'), ('Min', m_min, 'red')],
                'label': f' | Mean Vpp: {vpp:.3f}V'
            }
    elif plot_mode == 'power':
        p_idx, mean_peak = get_power_peaks(raw_y, custom_params=peak_params, cutoff=cutoff_val)
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

    if plot_mode == 'both':
        selected = set(include_graphs or ['voltage', 'power'])
        show_voltage = 'voltage' in selected
        show_power = 'power' in selected

        if downsample_percent < 100:
            target_size = max(1, int(len(df) * (downsample_percent / 100.0)))
            indices = np.linspace(0, len(df) - 1, target_size, dtype=int)
            x_plot = df.iloc[indices][time_col] if time_col else indices
        else:
            x_plot = x_values if x_values is not None else np.arange(len(df))

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if signal_col and show_voltage:
            raw_signal = df[signal_col].astype(float).values
            signal_trace_count = 0
            if show_raw_signal:
                y_signal = raw_signal[indices] if downsample_percent < 100 else raw_signal
                fig.add_trace(
                    go.Scatter(x=x_plot, y=y_signal, mode='lines', name=signal_label, line=dict(color='#1f77b4')),
                    secondary_y=False
                )
                signal_trace_count += 1
            if notch_enabled and show_filtered_signal and time_col:
                try:
                    fs_notch = infer_sampling_rate(df[time_col].values)
                    filtered_signal, notch_freqs, notch_qs = apply_notch_filter_chain(raw_signal, fs_notch, notch_params)
                    y_signal_filtered = filtered_signal[indices] if downsample_percent < 100 else filtered_signal
                    filtered_line = dict(color='#6f42c1')
                    if show_raw_signal:
                        filtered_line['dash'] = 'dash'
                    fig.add_trace(
                        go.Scatter(
                            x=x_plot, y=y_signal_filtered, mode='lines',
                            name=f'{signal_label} Notch ({len(notch_freqs)} filter(s))',
                            line=filtered_line
                        ),
                        secondary_y=False
                    )
                    signal_trace_count += 1
                except (ValueError, RuntimeError) as notch_error:
                    fig.add_annotation(
                        x=0.5,
                        y=0.9,
                        xref="x domain",
                        yref="y domain",
                        text=f"Notch filter unavailable ({notch_error})",
                        showarrow=False,
                        font=dict(color="gray")
                    )
            elif notch_enabled and show_filtered_signal and not time_col:
                fig.add_annotation(
                    x=0.5,
                    y=0.9,
                    xref="x domain",
                    yref="y domain",
                    text="Notch filter unavailable (missing time axis)",
                    showarrow=False,
                    font=dict(color="gray")
                )
            if show_raw_signal:
                v_peak_idx, _, _, _, _ = get_signal_peaks(raw_signal, custom_params=peak_params, cutoff=cutoff_val)
                if v_peak_idx is not None:
                    x_v_peaks = df.loc[v_peak_idx, time_col] if time_col else v_peak_idx
                    fig.add_trace(
                        go.Scatter(
                            x=x_v_peaks,
                            y=raw_signal[v_peak_idx],
                            mode='markers',
                            name=f'{signal_label} Peaks',
                            marker=dict(color='#1f77b4', size=7, symbol='circle')
                        ),
                        secondary_y=False
                    )
                    signal_trace_count += 1
            if signal_trace_count == 0:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="x domain",
                    yref="y domain",
                    text="Signal hidden (enable non-filtered and/or filtered signal)",
                    showarrow=False,
                    font=dict(color="gray")
                )
            if show_power and signal_mode != 'current' and req is not None and req != 0:
                power_series = (raw_signal ** 2) / req
                y_power = power_series[indices] if downsample_percent < 100 else power_series
                fig.add_trace(
                    go.Scatter(x=x_plot, y=y_power, mode='lines', name='Power', line=dict(color='#d62728')),
                    secondary_y=True
                )
                p_peak_idx, _ = get_power_peaks(power_series, custom_params=peak_params, cutoff=cutoff_val)
                if p_peak_idx is not None:
                    x_p_peaks = df.loc[p_peak_idx, time_col] if time_col else p_peak_idx
                    fig.add_trace(
                        go.Scatter(
                            x=x_p_peaks,
                            y=power_series[p_peak_idx],
                            mode='markers',
                            name='Power Peaks',
                            marker=dict(color='#d62728', size=7, symbol='diamond')
                        ),
                        secondary_y=True
                    )
            elif show_power and signal_mode == 'current':
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="x domain",
                    yref="y domain",
                    text="Power disabled in SC mode",
                    showarrow=False,
                    font=dict(color="gray")
                )
            elif show_power:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="x domain",
                    yref="y domain",
                    text="Power unavailable (missing Req)",
                    showarrow=False,
                    font=dict(color="gray")
                )
            elif not show_voltage:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="x domain",
                    yref="y domain",
                    text="Voltage curve hidden by selection",
                    showarrow=False,
                    font=dict(color="gray")
                )
        elif plot_columns and show_voltage:
            primary_fallback = signal_col
            raw_primary = df[primary_fallback].astype(float).values
            y_primary = raw_primary[indices] if downsample_percent < 100 else raw_primary
            fig.add_trace(
                go.Scatter(x=x_plot, y=y_primary, mode='lines', name=primary_fallback, line=dict(color='#1f77b4')),
                secondary_y=False
            )
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="x domain",
                yref="y domain",
                text="Voltage column not found; power unavailable",
                showarrow=False,
                font=dict(color="gray")
            )
        else:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="x domain",
                yref="y domain",
                text="No compatible graph selected for this view",
                showarrow=False,
                font=dict(color="gray")
            )

        fig.update_layout(title=title, xaxis_title=time_col if time_col else 'Index', height=600)
        fig.update_yaxes(title_text=f'{signal_label} ({signal_unit})', secondary_y=False)
        fig.update_yaxes(title_text='Power (W)', secondary_y=True)
        for marker_x in (cycle_markers or []):
            fig.add_vline(x=marker_x, line_dash='dot', line_color='gray', opacity=0.5)
        return fig.to_html(include_plotlyjs='cdn', div_id='plot')

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
    for marker_x in (cycle_markers or []):
        fig.add_vline(x=marker_x, line_dash='dot', line_color='gray', opacity=0.5)

    return fig.to_html(include_plotlyjs='cdn', div_id='plot')


def create_combined_motor_daq_plot(exp_df, exp_path, title, downsample_percent=100, gain=None, req=None, peak_params=None,
                                   selected_graphs=None, notch_params=None, signal_mode='voltage',
                                   cycle_markers=None, use_converted_signal: bool = False):

    if use_converted_signal:
        exp_df = apply_gain_to_dataframe(exp_df, exp_path, gain)

    def find_col(df, keywords):
        for col in df.columns:
            if any(k.lower() in col.lower() for k in keywords) and "unnamed" not in col.lower():
                return col
        return None

    v_col = find_col(exp_df, ['input 0', 'voltage'])
    c_col = find_col(exp_df, ['input 1', 'current', 'isc'])
    p_col = find_col(exp_df, ['position', 'actual position'])
    f_col = find_col(exp_df, ['force', 'measured force'])

    d_time = exp_df['Time'] if 'Time' in exp_df.columns else np.arange(len(exp_df))

    if 'Time' in exp_df.columns:
        m_time = exp_df['Time']
    else:
        duration = d_time.max() if len(d_time) > 0 else 1
        m_time = np.linspace(0, duration, len(exp_df))
    cutoff_val = peak_params.get('cutoff', 0.1) if peak_params else 0.1
    notch_enabled = bool(notch_params and notch_params.get('enabled'))
    show_raw_signal = bool(notch_params.get('show_raw_signal', True)) if notch_params else True
    show_filtered_signal = bool(notch_params.get('show_filtered_signal', True)) if notch_params else True
    selected = set(selected_graphs or ['voltage', 'power', 'position', 'force', 'is_moving', 'up_down'])
    primary_col = c_col if signal_mode == 'current' else v_col
    primary_label = 'Current' if signal_mode == 'current' else 'Voltage'
    primary_unit = 'A' if signal_mode == 'current' else 'V'
    moving_col = 'IsMoving_Bool' if 'IsMoving_Bool' in exp_df.columns else None
    up_down_col = 'Motor_Up_Down_Bool' if 'Motor_Up_Down_Bool' in exp_df.columns else None
    row_specs = []
    if 'voltage' in selected:
        row_specs.append(('voltage', f'{primary_label} ({primary_unit})'))
    if 'power' in selected:
        row_specs.append(('power', 'Power (W)'))
    if 'position' in selected:
        row_specs.append(('position', 'Position (mm)'))
    if 'force' in selected:
        row_specs.append(('force', 'Force (N)'))
    if 'is_moving' in selected:
        row_specs.append(('is_moving', 'IsMoving_Bool'))
    if 'up_down' in selected:
        row_specs.append(('up_down', 'Motor_Up_Down_Bool'))
    if not row_specs:
        raise ValueError('No graphs selected')

    fig = make_subplots(
        rows=len(row_specs), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=tuple(label for _, label in row_specs)
    )

    primary_signal = exp_df[primary_col].astype(float).values if primary_col else None
    voltage_signal = exp_df[v_col].astype(float).values if v_col else None

    for row_index, (graph_type, _) in enumerate(row_specs, start=1):
        axis_suffix = '' if row_index == 1 else str(row_index)
        if graph_type == 'voltage':
            if primary_col:
                traces_added = 0
                if show_raw_signal:
                    fig.add_trace(
                        go.Scatter(x=d_time, y=primary_signal, name=primary_label, line=dict(color='blue')),
                        row=row_index, col=1
                    )
                    traces_added += 1

                if notch_enabled and show_filtered_signal and 'Time' in exp_df.columns:
                    try:
                        fs_notch = infer_sampling_rate(exp_df['Time'].values)
                        filtered_voltage, notch_freqs, notch_qs = apply_notch_filter_chain(primary_signal, fs_notch, notch_params)
                        filtered_line = dict(color='#6f42c1')
                        if show_raw_signal:
                            filtered_line['dash'] = 'dash'
                        fig.add_trace(
                            go.Scatter(
                                x=d_time,
                                y=filtered_voltage,
                                name=f'{primary_label} Notch ({len(notch_freqs)} filter(s))',
                                line=filtered_line
                            ),
                            row=row_index, col=1
                        )
                        traces_added += 1
                    except (ValueError, RuntimeError) as notch_error:
                        fig.add_annotation(
                            x=0.5, y=0.9,
                            xref=f"x{axis_suffix} domain",
                            yref=f"y{axis_suffix} domain",
                            text=f"Notch filter unavailable ({notch_error})",
                            showarrow=False,
                            font=dict(color="gray")
                        )
                elif notch_enabled and show_filtered_signal and 'Time' not in exp_df.columns:
                    fig.add_annotation(
                        x=0.5, y=0.9,
                        xref=f"x{axis_suffix} domain",
                        yref=f"y{axis_suffix} domain",
                        text="Notch filter unavailable (missing time axis)",
                        showarrow=False,
                        font=dict(color="gray")
                    )
                if show_raw_signal:
                    v_peak_idx, _, _, _, _ = get_signal_peaks(primary_signal, custom_params=peak_params, cutoff=cutoff_val)
                    if v_peak_idx is not None:
                        x_v_peaks = exp_df.loc[v_peak_idx, 'Time'] if 'Time' in exp_df.columns else v_peak_idx
                        fig.add_trace(
                            go.Scatter(
                                x=x_v_peaks,
                                y=primary_signal[v_peak_idx],
                                mode='markers',
                                name=f'{primary_label} Peaks',
                                marker=dict(color='#1f77b4', size=7, symbol='circle')
                            ),
                            row=row_index, col=1
                        )
                        traces_added += 1
                if traces_added == 0:
                    fig.add_annotation(
                        x=0.5, y=0.5,
                        xref=f"x{axis_suffix} domain",
                        yref=f"y{axis_suffix} domain",
                        text="Signal hidden (enable non-filtered and/or filtered signal)",
                        showarrow=False,
                        font=dict(color="gray")
                    )
            else:
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref=f"x{axis_suffix} domain",
                    yref=f"y{axis_suffix} domain",
                    text=f"{primary_label} unavailable",
                    showarrow=False,
                    font=dict(color="gray")
                )
            fig.update_yaxes(title_text=primary_unit, row=row_index, col=1)

        elif graph_type == 'power':
            if signal_mode != 'current' and req is not None and req != 0 and voltage_signal is not None:
                power_signal = (voltage_signal ** 2) / req
                fig.add_trace(
                    go.Scatter(x=d_time, y=power_signal, name="Power", line=dict(color='red')),
                    row=row_index, col=1
                )
                p_peak_idx, _ = get_power_peaks(power_signal, custom_params=peak_params, cutoff=cutoff_val)
                if p_peak_idx is not None:
                    x_p_peaks = exp_df.loc[p_peak_idx, 'Time'] if 'Time' in exp_df.columns else p_peak_idx
                    fig.add_trace(
                        go.Scatter(
                            x=x_p_peaks,
                            y=power_signal[p_peak_idx],
                            mode='markers',
                            name='Power Peaks',
                            marker=dict(color='#d62728', size=7, symbol='diamond')
                        ),
                        row=row_index, col=1
                    )
            else:
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref=f"x{axis_suffix} domain",
                    yref=f"y{axis_suffix} domain",
                    text="Power unavailable (missing Req or Voltage)",
                    showarrow=False,
                    font=dict(color="gray")
                )
            fig.update_yaxes(title_text="W", row=row_index, col=1)

        elif graph_type == 'position':
            if p_col:
                fig.add_trace(
                    go.Scatter(x=m_time, y=exp_df[p_col], name="Position", line=dict(color='orange')),
                    row=row_index, col=1
                )
            else:
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref=f"x{axis_suffix} domain",
                    yref=f"y{axis_suffix} domain",
                    text="Position unavailable",
                    showarrow=False,
                    font=dict(color="gray")
                )
            fig.update_yaxes(title_text="mm", row=row_index, col=1)

        elif graph_type == 'force':
            if f_col:
                fig.add_trace(
                    go.Scatter(x=m_time, y=exp_df[f_col], name="Force", line=dict(color='green')),
                    row=row_index, col=1
                )
            else:
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref=f"x{axis_suffix} domain",
                    yref=f"y{axis_suffix} domain",
                    text="Force unavailable",
                    showarrow=False,
                    font=dict(color="gray")
                )
            fig.update_yaxes(title_text="N", row=row_index, col=1)

        elif graph_type == 'is_moving':
            if moving_col:
                fig.add_trace(
                    go.Scatter(
                        x=d_time, y=exp_df[moving_col].astype(float),
                        name="IsMoving_Bool", line=dict(color='#9467bd', shape='hv')
                    ),
                    row=row_index, col=1
                )
            else:
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref=f"x{axis_suffix} domain",
                    yref=f"y{axis_suffix} domain",
                    text="IsMoving_Bool unavailable",
                    showarrow=False,
                    font=dict(color="gray")
                )
            fig.update_yaxes(title_text="bool", row=row_index, col=1)

        elif graph_type == 'up_down':
            if up_down_col:
                fig.add_trace(
                    go.Scatter(
                        x=d_time, y=exp_df[up_down_col].astype(float),
                        name="Motor_Up_Down_Bool", line=dict(color='#8c564b', shape='hv')
                    ),
                    row=row_index, col=1
                )
            else:
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref=f"x{axis_suffix} domain",
                    yref=f"y{axis_suffix} domain",
                    text="Motor_Up_Down_Bool unavailable",
                    showarrow=False,
                    font=dict(color="gray")
                )
            fig.update_yaxes(title_text="bool", row=row_index, col=1)

    fig.update_layout(
        title=title, height=max(380, 260 * len(row_specs)),
        template="plotly_white", showlegend=False, hovermode="x unified"
    )
    for marker_x in (cycle_markers or []):
        fig.add_vline(x=marker_x, line_dash='dot', line_color='gray', opacity=0.45, row='all', col=1)
    fig.update_xaxes(title_text="Time (s)", row=len(row_specs), col=1)

    return fig.to_html(include_plotlyjs='cdn', div_id='plot')


def create_signal_fft_plot(exp_df: pd.DataFrame, exp_path: str, gain: float = None,
                           signal_mode: str = 'voltage', notch_params: dict = None,
                           use_converted_signal: bool = False) -> str:
    df_plot = apply_gain_to_dataframe(exp_df, exp_path, gain) if use_converted_signal else exp_df.copy()
    signal_col = find_primary_signal_column(df_plot, signal_mode)
    signal_label = 'Current' if signal_mode == 'current' else 'Voltage'
    signal_unit = 'A' if signal_mode == 'current' else 'V'

    if not signal_col:
        return f'<p>{signal_label} FFT unavailable (signal column not found).</p>'
    if 'Time' not in df_plot.columns:
        return f'<p>{signal_label} FFT unavailable (missing time axis).</p>'

    try:
        fs = infer_sampling_rate(df_plot['Time'].values)
    except (ValueError, RuntimeError) as error:
        return f'<p>{signal_label} FFT unavailable ({error}).</p>'

    signal = df_plot[signal_col].astype(float).values
    signal = signal - np.mean(signal)
    n = len(signal)
    if n < 2:
        return f'<p>{signal_label} FFT unavailable (not enough samples).</p>'

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amplitude = (2.0 / n) * np.abs(np.fft.rfft(signal))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=amplitude,
            mode='lines',
            name=f'{signal_label} FFT (raw)',
            line=dict(color='#17a2b8')
        )
    )
    notch_enabled = bool(notch_params and notch_params.get('enabled'))
    if notch_enabled:
        try:
            filtered_signal, notch_freqs, notch_qs = apply_notch_filter_chain(
                df_plot[signal_col].astype(float).values,
                fs,
                notch_params
            )
            filtered_signal = filtered_signal - np.mean(filtered_signal)
            amplitude_filtered = (2.0 / n) * np.abs(np.fft.rfft(filtered_signal))
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=amplitude_filtered,
                    mode='lines',
                    name=f'{signal_label} FFT (notch, {len(notch_freqs)} filter(s))',
                    line=dict(color='#6f42c1', dash='dash')
                )
            )
        except (ValueError, RuntimeError):
            pass

    fig.update_layout(
        title=f'{signal_label} FFT',
        xaxis_title='Frequency (Hz)',
        yaxis_title=f'{signal_unit} amplitude',
        height=500,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    return fig.to_html(include_plotlyjs='cdn', div_id='fft_plot')


def create_cycles_overlay_plot(cycles_list: List[pd.DataFrame], exp_path: str, gain: float = None,
                               signal_mode: str = 'voltage', cycle_start: int = 1,
                               cycle_end: int = None, notch_params: dict = None,
                               show_raw_signal: bool = True, show_filtered_signal: bool = False,
                               use_converted_signal: bool = False) -> str:
    if not cycles_list:
        return '<p>No cycles available for overlay.</p>'
    notch_enabled = bool(notch_params and notch_params.get('enabled'))
    effective_show_filtered = bool(show_filtered_signal and notch_enabled)
    if not show_raw_signal and not effective_show_filtered:
        return '<p>Cycle overlay hidden (enable raw and/or filtered signal).</p>'

    total_cycles = len(cycles_list)
    start_cycle = max(1, int(cycle_start or 1))
    end_cycle = total_cycles if cycle_end is None else int(cycle_end)
    end_cycle = min(total_cycles, max(1, end_cycle))
    if end_cycle < start_cycle:
        end_cycle = start_cycle

    cycles_to_plot = cycles_list[start_cycle - 1:end_cycle]

    fig = go.Figure()
    signal_label = 'Current' if signal_mode == 'current' else 'Voltage'
    signal_unit = 'A' if signal_mode == 'current' else 'V'
    plotted_cycles = 0
    notch_unavailable_note = None
    if show_filtered_signal and not notch_enabled:
        notch_unavailable_note = 'Filtered overlay unavailable (enable notch filter first)'

    for index, cycle_df in enumerate(cycles_to_plot, start=start_cycle):
        if cycle_df is None or cycle_df.empty:
            continue
        cycle_plot = apply_gain_to_dataframe(cycle_df, exp_path, gain) if use_converted_signal else cycle_df.copy()
        signal_col = find_primary_signal_column(cycle_plot, signal_mode)
        if not signal_col:
            continue

        y_values = cycle_plot[signal_col].astype(float).values
        if 'Time' in cycle_plot.columns:
            x_values = cycle_plot['Time'].astype(float).values
            x_values = x_values - x_values[0] if len(x_values) else x_values
            x_title = 'Time relative to cycle start (s)'
        else:
            x_values = np.arange(len(cycle_plot))
            x_title = 'Sample index'

        cycle_has_trace = False
        if show_raw_signal:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines',
                    name=f'Cycle {index} (raw)',
                    line=dict(width=1.6, color='#1f77b4'),
                    opacity=0.3,
                    showlegend=False
                )
            )
            cycle_has_trace = True

        if effective_show_filtered:
            if 'Time' not in cycle_plot.columns:
                notch_unavailable_note = 'Filtered overlay unavailable (missing time axis)'
            else:
                try:
                    fs_notch = infer_sampling_rate(cycle_plot['Time'].astype(float).values)
                    filtered_signal, _, _ = apply_notch_filter_chain(y_values, fs_notch, notch_params)
                    filtered_line = dict(width=1.8, color='#6f42c1')
                    if show_raw_signal:
                        filtered_line['dash'] = 'dash'
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=filtered_signal,
                            mode='lines',
                            name=f'Cycle {index} (filtered)',
                            line=filtered_line,
                            opacity=0.45 if show_raw_signal else 0.55,
                            showlegend=False
                        )
                    )
                    cycle_has_trace = True
                except (ValueError, RuntimeError) as notch_error:
                    notch_unavailable_note = f'Filtered overlay unavailable ({notch_error})'

        if cycle_has_trace:
            plotted_cycles += 1

    if plotted_cycles == 0:
        if notch_unavailable_note:
            return f'<p>{notch_unavailable_note}.</p>'
        return f'<p>No {signal_label.lower()} signal found in cycles.</p>'

    if notch_unavailable_note:
        fig.add_annotation(
            x=0.5,
            y=0.98,
            xref='paper',
            yref='paper',
            text=notch_unavailable_note,
            showarrow=False,
            font=dict(color='gray')
        )

    if show_raw_signal and effective_show_filtered:
        signal_variant = 'raw + filtered (notch)'
    elif effective_show_filtered:
        signal_variant = 'filtered (notch)'
    else:
        signal_variant = 'raw'

    fig.update_layout(
        title=f'{signal_label} Cycle Overlay ({signal_variant}, cycles {start_cycle}-{end_cycle}, {plotted_cycles}/{len(cycles_to_plot)} cycles)',
        xaxis_title=x_title,
        yaxis_title=f'{signal_label} ({signal_unit})',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    return fig.to_html(include_plotlyjs='cdn', div_id='cycles_overlay_plot')


def create_mean_power_vs_req_plot(grouped_power: dict, grouped_peak_power: dict = None,
                                  title: str = 'Power Analysis vs Resistance',
                                  div_id: str = 'mean_power_plots') -> str:
    if not grouped_power:
        return '<p>No data available</p>'

    # Define explicit colors for clarity
    mean_color = '#1f77b4'  # Professional Blue
    peak_color = '#d62728'  # Professional Red

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for TribuId, points in grouped_power.items():
        if not points: continue
        points.sort(key=lambda x: x[0])
        reqs, powers = zip(*points)

        # Primary axis (Mean Power) - Solid Blue Line
        fig.add_trace(
            go.Scatter(
                x=reqs, y=powers,
                mode='markers+lines',
                name=f'{TribuId} (Mean)',
                line=dict(color=mean_color)
            ),
            secondary_y=False
        )

        # Secondary axis (Peak Power) - Dashed Red Line
        if grouped_peak_power and TribuId in grouped_peak_power:
            peak_points = grouped_peak_power[TribuId]
            peak_points.sort(key=lambda x: x[0])
            p_reqs, p_peaks = zip(*peak_points)
            fig.add_trace(
                go.Scatter(
                    x=p_reqs, y=p_peaks,
                    mode='markers+lines',
                    line=dict(dash='dash', color=peak_color),
                    name=f'{TribuId} (Peak)'
                ),
                secondary_y=True
            )

    fig.update_layout(title=title, xaxis_title='Resistance (Req) [ohms]', height=500)

    # Match Left Y-Axis to Mean Power color
    fig.update_yaxes(
        title_text="<b>Mean Power [W]</b>",
        title_font=dict(color=mean_color),
        tickfont=dict(color=mean_color),
        secondary_y=False
    )

    # Match Right Y-Axis to Peak Power color
    fig.update_yaxes(
        title_text="<b>Avg Peak Power (Last 10 cycles) [W]</b>",
        title_font=dict(color=peak_color),
        tickfont=dict(color=peak_color),
        secondary_y=True
    )

    return fig.to_html(include_plotlyjs='cdn', div_id=div_id)


def create_mean_vpp_vs_req_plot(grouped_data: dict, title: str = 'Mean Vpp vs Resistance') -> str:
    """This function remains untouched, takes a single dict of grouped_data"""
    if not grouped_data:
        return '<p>No data available</p>'

    fig = go.Figure()
    for TribuId, data_points in grouped_data.items():
        if not data_points: continue
        data_points.sort(key=lambda x: x[0])
        reqs, vpps = zip(*data_points)
        fig.add_trace(go.Scatter(x=reqs, y=vpps, mode='markers+lines', name=f'{TribuId}'))

    fig.update_layout(title=title, xaxis_title='Resistance (Req) [ohms]', yaxis_title='Mean Vpp [V]', height=400)
    return fig.to_html(include_plotlyjs='cdn', div_id='mean_vpp_plot')


def create_optimal_power_plot(optimal_points: list, title: str = 'Optimal Power Comparison across TribuIds') -> str:
    if not optimal_points:
        return ''

    from plotly.subplots import make_subplots

    # Sort alphabetically by TribuId
    optimal_points.sort(key=lambda x: str(x['TribuId']))

    tribu_ids = [str(p['TribuId']) for p in optimal_points]

    # Extract Mean Power Data
    max_means = [p.get('max_power', 0) for p in optimal_points]
    req_mean_labels = [f"Req: {p.get('req_mean', 'N/A')} Ω" for p in optimal_points]

    # Extract Peak Power Data
    max_peaks = [p.get('max_peak_power', 0) for p in optimal_points]
    req_peak_labels = [f"Req: {p.get('req_peak', 'N/A')} Ω" for p in optimal_points]

    # Create figure with secondary Y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Max Mean Power (Blue Bars) - Primary Axis
    fig.add_trace(
        go.Bar(
            x=tribu_ids,
            y=max_means,
            name='Max Mean Power',
            text=req_mean_labels,
            textposition='auto',
            marker=dict(color='#1f77b4'),  # Match the blue from the previous charts
            hovertemplate="<b>%{x}</b><br>Max Mean: %{y:.4g} W<br>%{text}<extra></extra>",
            offsetgroup=1
        ),
        secondary_y=False
    )

    # Trace 2: Max Peak Power (Red Markers/Line) - Secondary Axis
    fig.add_trace(
        go.Bar(
            x=tribu_ids,
            y=max_peaks,
            name='Max Peak Power',
            text=req_peak_labels,
            marker=dict(color='#d62728'),  # Match the red from previous chart
            hovertemplate="<b>%{x}</b><br>Max Peak: %{y:.4g} W<br>%{text}<extra></extra>",
            offsetgroup=2
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        xaxis_title='TribuId',
        height=500,
        template="plotly_white",
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)  # Move legend to top
    )

    # Apply colors to the Y-axes to match the data
    fig.update_yaxes(
        title_text="<b>Max Mean Power [W]</b>",
        title_font=dict(color='#1f77b4'),
        tickfont=dict(color='#1f77b4'),
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="<b>Max Peak Power [W]</b>",
        title_font=dict(color='#d62728'),
        tickfont=dict(color='#d62728'),
        secondary_y=True
    )

    return fig.to_html(include_plotlyjs='cdn', div_id='optimal_power_plot')


def create_comparison_summary_plot(voc_results: list, isc_results: list) -> str:
    """
    Creates a bar plot comparing Voc and Isc metrics across filenames.
    """

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Add Voc Mean Max (Blue Bars)
    if voc_results:
        voc_names = [item['name'] for item in voc_results]
        voc_values = [item['max_v'] for item in voc_results]
        fig.add_trace(
            go.Bar(
                x=voc_names,
                y=voc_values,
                name="Mean Max Voc (V)",
                marker_color='rgba(0, 123, 255, 0.7)', # Blue
                offsetgroup=1
            ),
            secondary_y=False,
        )

    # 2. Add Isc Pk-Pk (Red Bars)
    if isc_results:
        isc_names = [item['name'] for item in isc_results]
        isc_values = [item['vpp_i'] for item in isc_results]
        fig.add_trace(
            go.Bar(
                x=isc_names,
                y=isc_values,
                name="Avg Isc Pk-Pk (A)",
                marker_color='rgba(255, 0, 0, 0.7)', # Red
                offsetgroup=2
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title_text="Metric Comparison: Voc vs Isc",
        xaxis_title="Filename",
        template="plotly_white",
        height=500,
        # 'group' places bars side-by-side; 'overlay' would put them on top
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="<b>Mean Max Voc</b> (V)", secondary_y=False, title_font=dict(color="blue"))
    fig.update_yaxes(title_text="<b>Mean Pk-Pk Isc</b> (A)", secondary_y=True, title_font=dict(color="red"))

    return fig.to_html(include_plotlyjs='cdn', div_id='comparison_plot')
