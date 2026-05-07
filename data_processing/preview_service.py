import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.ndimage import label
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

def get_signal_peaks(y_raw: np.ndarray, custom_params: dict = None, cutoff: float = 0.1):
    """Shared logic for voltage peak detection with optional custom tuning."""
    if not HAS_SCIPY:
        return None, None, 0.0, 0.0, 0.0

    y_smooth = apply_lowpass_filter(y_raw, cutoff=cutoff)

    # These are your ORIGINAL default parameters
    params = {
        'height': np.percentile(y_smooth, 95),
        'prominence': np.std(y_smooth) * 2,
        'distance': 100
    }

    # If custom parameters are provided, override the defaults
    if custom_params:
        params.update(custom_params)

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
                                  title: str = 'Power Analysis vs Resistance',
                                  div_id: str = 'mean_power_plots') -> str:
    if not HAS_PLOTLY or not grouped_power:
        return '<p>No data available</p>'

    # Define explicit colors for clarity
    mean_color = '#1f77b4'  # Professional Blue
    peak_color = '#d62728'  # Professional Red

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for tribu_id, points in grouped_power.items():
        if not points: continue
        points.sort(key=lambda x: x[0])
        reqs, powers = zip(*points)

        # Primary axis (Mean Power) - Solid Blue Line
        fig.add_trace(
            go.Scatter(
                x=reqs, y=powers,
                mode='markers+lines',
                name=f'{tribu_id} (Mean)',
                line=dict(color=mean_color)
            ),
            secondary_y=False
        )

        # Secondary axis (Peak Power) - Dashed Red Line
        if grouped_peak_power and tribu_id in grouped_peak_power:
            peak_points = grouped_peak_power[tribu_id]
            peak_points.sort(key=lambda x: x[0])
            p_reqs, p_peaks = zip(*peak_points)
            fig.add_trace(
                go.Scatter(
                    x=p_reqs, y=p_peaks,
                    mode='markers+lines',
                    line=dict(dash='dash', color=peak_color),
                    name=f'{tribu_id} (Peak)'
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


def create_optimal_power_plot(optimal_points: list, title: str = 'Optimal Power Comparison across TribuIds') -> str:
    if not HAS_PLOTLY or not optimal_points:
        return ''

    from plotly.subplots import make_subplots

    # Sort alphabetically by TribuId
    optimal_points.sort(key=lambda x: str(x['tribu_id']))

    tribu_ids = [str(p['tribu_id']) for p in optimal_points]

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


def create_no_ra_plot(voc_data_list: list, isc_data_list: list, title: str) -> str:
    """
    voc_data_list: List of dicts [{'name': filename, 'df': dataframe}, ...]
    isc_data_list: List of dicts [{'name': filename, 'df': dataframe}, ...]
    """
    if not HAS_PLOTLY:
        return "<p>Plotly not installed</p>"

    # 1. Calculate total number of plots needed
    num_voc = len(voc_data_list)
    num_isc = len(isc_data_list)
    total_plots = num_voc + num_isc

    if total_plots == 0:
        return "<p>No data to plot</p>"

    # 2. Generate dynamic subplot titles based on filenames
    subplot_titles = []
    for item in voc_data_list:
        subplot_titles.append(f"Voc: {item['name']}")
    for item in isc_data_list:
        subplot_titles.append(f"Isc: {item['name']}")

    # 3. Create dynamic rows
    fig = make_subplots(
        rows=total_plots,
        cols=1,
        shared_xaxes=False,  # Set to True if you want them all to zoom together on the X axis
        subplot_titles=subplot_titles,
        vertical_spacing=0.05  # Add some space between graphs
    )

    def find_column(df, possible_names):
        for col in df.columns:
            if any(name.lower() in col.lower() for name in possible_names):
                return col
        return None

    all_voc_max = []
    all_isc_vpp = []

    # We use this counter to track which row we are plotting on
    current_row = 1

    # ==========================================
    # Process Multiple VOC Files
    # ==========================================
    for item in voc_data_list:
        df, name = item['df'], item['name']
        time_col = find_column(df, ['time'])
        val_col = find_column(df, ['voltage', 'input 0', 'voc'])

        if val_col:
            x_vals = df[time_col].values if time_col else np.arange(len(df))
            y_vals = df[val_col].values

            # Plot the main voltage line for this file on its OWN row
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=f"Voc: {name}"), row=current_row, col=1)

            p_idx, _, m_max, _, _ = get_plateau_peaks(y_vals, threshold_percentile=80, cutoff=0.05)

            if p_idx is not None:
                all_voc_max.append(m_max)
                # Plot the peak markers for this file
                fig.add_trace(go.Scatter(
                    x=x_vals[p_idx], y=y_vals[p_idx], mode='markers',
                    name=f'Peaks ({name})', showlegend=False,
                    marker=dict(size=8, symbol='circle-open')
                ), row=current_row, col=1)

            # Label the Y axis for this specific row
            fig.update_yaxes(title_text="Voc (V)", row=current_row, col=1)

        current_row += 1

    # ==========================================
    # Process Multiple ISC Files
    # ==========================================
    for item in isc_data_list:
        df, name = item['df'], item['name']
        time_col = find_column(df, ['time'])
        val_col = find_column(df, ['current', 'isc', 'ampere'])

        if val_col:
            x_vals = df[time_col].values if time_col else np.arange(len(df))
            y_vals = df[val_col].values

            # Plot the main current line for this file on its OWN row
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=f"Isc: {name}"), row=current_row, col=1)

            isc_params = {'distance': 20, 'prominence': np.std(y_vals) * 3}
            p_idx, t_idx, m_max, m_min, vpp = get_signal_peaks(y_vals, custom_params=isc_params, cutoff=0.3)

            if p_idx is not None and t_idx is not None:
                all_isc_vpp.append(vpp)

                # Plot Max Peaks
                fig.add_trace(go.Scatter(
                    x=x_vals[p_idx], y=y_vals[p_idx], mode='markers',
                    name=f'Isc Max ({name})', showlegend=False,
                    marker=dict(size=6, symbol='triangle-up')
                ), row=current_row, col=1)

                # Plot Min Peaks
                fig.add_trace(go.Scatter(
                    x=x_vals[t_idx], y=y_vals[t_idx], mode='markers',
                    name=f'Isc Min ({name})', showlegend=False,
                    marker=dict(size=6, symbol='triangle-down')
                ), row=current_row, col=1)

            # Label the axes for this specific row
            fig.update_yaxes(title_text="Isc (A)", row=current_row, col=1)
            fig.update_xaxes(title_text="Time (s)", row=current_row, col=1)

        current_row += 1

    # ==========================================
    # Dynamic Title & Layout Calculation
    # ==========================================
    # Change this line in create_no_ra_plot[cite: 10]:
    voc_summary = f"Avg Voc Max: {np.mean(np.abs(all_voc_max)):.3g} V" if all_voc_max else ""
    isc_summary = f"Avg Isc Pk-Pk: {np.mean(all_isc_vpp):.3g} A" if all_isc_vpp else ""

    summaries = [s for s in [voc_summary, isc_summary] if s]
    full_title = f"{title} | {' | '.join(summaries)}" if summaries else title

    # Calculate height dynamically: Give each plot 350 pixels of vertical space
    # (So 4 files = 1400px height, making it scrollable and easy to read)
    dynamic_height = max(800, 350 * total_plots)

    fig.update_layout(
        height=dynamic_height,
        title_text=full_title,
        showlegend=True,
        template="plotly_white"
    )

    return fig.to_html(include_plotlyjs='cdn', div_id='no_ra_plot')

def create_comparison_summary_plot(voc_results: list, isc_results: list) -> str:
    """
    Creates a bar plot comparing Voc and Isc metrics across filenames.
    """
    if not HAS_PLOTLY:
        return ""

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

def has_tdms_support() -> bool: return HAS_NPTDMS


def has_plotly_support() -> bool: return HAS_PLOTLY