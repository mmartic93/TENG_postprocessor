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

# Nueva importación para detectar picos
try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


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


def create_plot_html(df: pd.DataFrame, title: str = 'Data Plot', downsample_percent: int = 80, gain: float = None,
                     plot_mode: str = 'voltage', req: float = None) -> str:
    """Create interactive Plotly scatter plot from DataFrame with relative peak detection for Vpp."""
    if not HAS_PLOTLY:
        raise RuntimeError('plotly library is not installed')

    if gain is not None:
        df = apply_gain_to_dataframe(df, gain)

    if plot_mode == 'power':
        if req is None:
            raise ValueError('Req value is required for power plot')
        df = calculate_power_dataframe(df, req)

    # Identificar columnas ANTES del downsampling
    time_col = 'Time(s)' if 'Time(s)' in df.columns else None
    plot_columns = [col for col in df.columns if col.lower() != 'index' and col != time_col]

    if not plot_columns:
        raise ValueError('No data columns to plot (only index column found)')

    vpp_info = None
    if plot_mode == 'voltage':
        if not HAS_SCIPY:
            raise RuntimeError('scipy is required for peak detection. Run: pip install scipy')

        primary_col = plot_columns[0]
        y_data = df[primary_col].values

        # Calcular un valor de "prominencia" dinámico basado en la desviación estándar
        # Esto evita que el ruido de fondo se detecte como picos falsos.
        dinamic_prominence = np.std(y_data) * 0.5

        # Encontrar índices de máximos relativos (picos positivos)
        peaks_idx, _ = find_peaks(y_data, prominence=dinamic_prominence,height=0.1)

        # Encontrar índices de mínimos relativos (invirtiendo la señal)
        troughs_idx, _ = find_peaks(-y_data, prominence=dinamic_prominence,height=0.1)

        if len(peaks_idx) > 0 and len(troughs_idx) > 0:
            # Calcular las medias
            mean_max = np.mean(y_data[peaks_idx])
            mean_min = np.mean(y_data[troughs_idx])
            vpp_mean = mean_max - mean_min

            # Guardar coordenadas exactas X e Y de todos los picos para graficarlos
            x_peaks = df.loc[peaks_idx, time_col] if time_col else peaks_idx
            y_peaks = y_data[peaks_idx]

            x_troughs = df.loc[troughs_idx, time_col] if time_col else troughs_idx
            y_troughs = y_data[troughs_idx]

            vpp_info = {
                'x_peaks': x_peaks, 'y_peaks': y_peaks,
                'x_troughs': x_troughs, 'y_troughs': y_troughs,
                'mean_max': mean_max, 'mean_min': mean_min, 'vpp': vpp_mean
            }

    # Aplicar downsampling al DataFrame principal
    original_length = len(df)
    if downsample_percent < 100:
        target_size = max(1, int(original_length * (downsample_percent / 100.0)))
        if target_size < original_length:
            indices = np.linspace(0, original_length - 1, target_size, dtype=int)
            df = df.iloc[indices].copy()

    # Eje X
    x_values = df[time_col] if time_col else None
    x_title = 'Time (s)' if time_col else 'Index'

    fig = go.Figure()

    # Trazos de datos
    if len(plot_columns) == 1:
        col = plot_columns[0]
        fig.add_trace(go.Scatter(x=x_values, y=df[col], mode='lines', name=col, line=dict(color='blue', width=1.5)))
    else:
        for col in plot_columns:
            fig.add_trace(go.Scatter(x=x_values, y=df[col], mode='lines', name=col))

    plot_title = f'{title} (sampled {downsample_percent}%)'

    # Añadir marcadores y medias de Vpp si se encontraron
    if vpp_info:
        # Marcadores de todos los picos positivos
        fig.add_trace(go.Scatter(
            x=vpp_info['x_peaks'], y=vpp_info['y_peaks'],
            mode='markers', name='Relative Maximums',
            marker=dict(color='green', size=6, symbol='circle')
        ))

        # Marcadores de todos los picos negativos
        fig.add_trace(go.Scatter(
            x=vpp_info['x_troughs'], y=vpp_info['y_troughs'],
            mode='markers', name='Relative Minimums',
            marker=dict(color='red', size=6, symbol='circle')
        ))

        # Línea horizontal para la media de los máximos
        fig.add_hline(y=vpp_info['mean_max'], line_dash="dash", line_color="green",
                      annotation_text=f"Mean Max: {vpp_info['mean_max']:.2f}")

        # Línea horizontal para la media de los mínimos
        fig.add_hline(y=vpp_info['mean_min'], line_dash="dash", line_color="red",
                      annotation_text=f"Mean Min: {vpp_info['mean_min']:.2f}")

        plot_title += f' | Mean Vpp: {vpp_info["vpp"]:.3f}'

    fig.update_layout(
        title=plot_title,
        xaxis_title=x_title,
        yaxis_title='Voltage' if plot_mode == 'voltage' else 'Value',
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


def create_mean_vpp_vs_req_plot(data_points: list, title: str = 'Mean Vpp vs Resistance') -> str:
    """
    Crea una gráfica de dispersión de la media de Vpp vs Resistencia (Req)
    a partir de una lista de tuplas (req, mean_vpp).
    """
    if not HAS_PLOTLY:
        raise RuntimeError('plotly library is not installed')

    if not data_points:
        return '<p>No data available for Mean Vpp vs Resistance plot.</p>'

    # Desempaquetamos los valores de resistencia y Vpp
    reqs, vpps = zip(*data_points)

    fig = go.Figure()

    # Añadimos la traza de puntos
    fig.add_trace(go.Scatter(
        x=reqs,
        y=vpps,
        mode='markers+lines',  # Añadimos líneas para ver la tendencia
        marker=dict(size=10, color='blue'),
        name='Mean Vpp',
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Resistance (Req) [ohms]',
        yaxis_title='Mean Vpp [V]',
        height=400,
        template='plotly_white'
    )

    return fig.to_html(include_plotlyjs='cdn', div_id='mean_vpp_plot')


def has_tdms_support() -> bool:
    return HAS_NPTDMS


def has_plotly_support() -> bool:
    return HAS_PLOTLY

