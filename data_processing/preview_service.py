import pandas as pd
import json
import numpy as np

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


def csv_to_dataframe(path: str) -> pd.DataFrame:
    """Load CSV file as pandas DataFrame."""
    return pd.read_csv(path)


def tdms_to_dataframe(path: str) -> pd.DataFrame:
    """Load TDMS file as pandas DataFrame, looking for 'Input 0' channel."""
    if not HAS_NPTDMS:
        raise RuntimeError('nptdms library is not installed')
    tdms = TdmsFile.read(path)
    
    # Search for 'Input 0' channel across all groups
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


def calculate_mean_power(df: pd.DataFrame, gain: float, req: float) -> float:
    """Return the mean power value of the primary numeric voltage column after gain."""
    if gain is None or req is None or req == 0:
        raise ValueError('Gain and Req are required to calculate mean power')
    df_gain = apply_gain_to_dataframe(df, gain)
    power_df = calculate_power_dataframe(df_gain, req)
    return float(power_df['Power'].mean())


def calculate_mean_power_from_file(path: str, ext: str, gain: float, req: float) -> float:
    if ext == '.csv':
        df = csv_to_dataframe(path)
    elif ext == '.tdms':
        df = tdms_to_dataframe(path)
    else:
        raise ValueError('Unsupported file type for mean power')
    return calculate_mean_power(df, gain, req)


def create_plot_html(df: pd.DataFrame, title: str = 'Data Plot', downsample_percent: int = 80, gain: float = None, plot_mode: str = 'voltage', req: float = None) -> str:
    """Create interactive Plotly scatter plot from DataFrame with optional downsampling."""
    if not HAS_PLOTLY:
        raise RuntimeError('plotly library is not installed')

    if gain is not None:
        df = apply_gain_to_dataframe(df, gain)

    if plot_mode == 'power':
        if req is None:
            raise ValueError('Req value is required for power plot')
        df = calculate_power_dataframe(df, req)

    # Apply downsampling if requested
    original_length = len(df)
    if downsample_percent < 100:
        target_size = max(1, int(original_length * (downsample_percent / 100.0)))
        if target_size < original_length:
            indices = np.linspace(0, original_length - 1, target_size, dtype=int)
            df = df.iloc[indices].copy()

    # Filter out 'index' column if it exists
    plot_columns = [col for col in df.columns if col.lower() != 'index']
    if not plot_columns:
        raise ValueError('No data columns to plot (only index column found)')

    # Handle single or multiple columns
    if len(plot_columns) == 1:
        # Single column - plot against row index
        col = plot_columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df[col],
            mode='lines+markers',
            name=col,
            line=dict(color='blue', width=2),
        ))
        fig.update_layout(
            title=f'{title} (sampled {downsample_percent}%)',
            xaxis_title='Index',
            yaxis_title=col,
            hovermode='x unified',
            height=600,
        )
    else:
        # Multiple columns - plot all against first column or index
        fig = go.Figure()
        for col in plot_columns:
            fig.add_trace(go.Scatter(
                y=df[col],
                mode='lines+markers',
                name=col,
            ))
        fig.update_layout(
            title=f'{title} (sampled {downsample_percent}%)',
            xaxis_title='Index',
            yaxis_title='Value',
            hovermode='x unified',
            height=600,
        )
    
    return fig.to_html(include_plotlyjs='cdn', div_id='plot')


def create_mean_power_vs_req_plot(data_points: list, title: str = 'Mean Power vs Resistance') -> str:
    """Create a scatter plot of mean power vs Req from list of (req, mean_power) tuples."""
    if not HAS_PLOTLY:
        raise RuntimeError('plotly library is not installed')

    if not data_points:
        return '<p>No data available for Mean Power vs Resistance plot.</p>'

    reqs, powers = zip(*data_points)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=reqs,
        y=powers,
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Mean Power',
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Resistance (Req) [ohms]',
        yaxis_title='Mean Power [W]',
        height=400,
    )
    
    return fig.to_html(include_plotlyjs='cdn', div_id='mean_power_plot')


def has_tdms_support() -> bool:
    return HAS_NPTDMS


def has_plotly_support() -> bool:
    return HAS_PLOTLY

