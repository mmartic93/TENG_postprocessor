import os
from flask import render_template, request, redirect, url_for, session, flash
from data_processing.LoadData import ExtractCycles
from data_processing.preview_service import create_combined_motor_daq_plot
import pandas as pd

from data_processing.metadata_loader import (
    allowed_meta,
    parse_metadata_csv,
    format_metadata_rows,
    validate_metadata_columns,
    get_required_columns,
    get_rows_for_tribuid,
    get_experiment_folders,
    find_loads_description_file,
    load_loads_description,
    lookup_load_info,
)
from data_processing.preview_service import (
    create_plot_html,
    create_mean_power_vs_req_plot,
    calculate_mean_power,
    calculate_peak_power,
    calculate_mean_vpp,
    has_tdms_support
)
from data_processing.validators import validate_tribuid

LIST_FILES_CACHE = {}
EXPERIMENT_DATA_CACHE = {}


def register_routes(app):
    def is_open_short_rload(rload_id: str) -> bool:
        return str(rload_id or '').strip().upper() in {'OC', 'SC'}

    def is_short_circuit_rload(rload_id: str) -> bool:
        return str(rload_id or '').strip().upper() == 'SC'

    def get_cached_experiment_data(exp_path: str):
        cached = EXPERIMENT_DATA_CACHE.get(exp_path)
        if isinstance(cached, dict) and 'df' in cached:
            return cached['df'], cached.get('cycle_markers', [])
        if cached is not None:
            return cached, []

        Cycles_list = ExtractCycles(exp_path)
        if len(Cycles_list) == 0:
            return None, None

        df_data_all = pd.concat(Cycles_list, ignore_index=True)
        cycle_markers = []
        offset = 0
        for cycle_df in Cycles_list[:-1]:
            offset += len(cycle_df)
            if offset >= len(df_data_all):
                break
            if 'Time' in df_data_all.columns:
                cycle_markers.append(float(df_data_all.iloc[offset]['Time']))
            else:
                cycle_markers.append(float(offset))

        EXPERIMENT_DATA_CACHE[exp_path] = {
            'df': df_data_all,
            'cycle_markers': cycle_markers,
        }
        return df_data_all, cycle_markers

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            metadata_path = request.form.get('metadata_path', '').strip()

            if not metadata_path:
                flash('Please provide a metadata file path')
                return redirect(request.url)

            if not os.path.exists(metadata_path):
                flash('Metadata file not found at specified path')
                return redirect(request.url)

            if not allowed_meta(metadata_path):
                flash('Unsupported metadata file type (use .csv or .xlsx)')
                return redirect(request.url)

            try:
                df = parse_metadata_csv(metadata_path)
                if not validate_metadata_columns(df):
                    missing = [col for col in get_required_columns() if col not in df.columns]
                    raise ValueError('Missing required columns: ' + ', '.join(missing))
            except Exception as error:
                flash(f'Failed to parse metadata: {error}')
                return redirect(request.url)

            session['metadata_path'] = metadata_path
            session.pop('selected_tribuid', None)
            LIST_FILES_CACHE.clear()
            EXPERIMENT_DATA_CACHE.clear()
            return redirect(url_for('metadata_preview'))

        return render_template('index.html')

    @app.route('/metadata', methods=['GET', 'POST'])
    def metadata_preview():
        metadata_path = str(session.get('metadata_path'))
        if not metadata_path:
            flash('Upload a metadata CSV first')
            return redirect(url_for('index'))

        try:
            df = parse_metadata_csv(metadata_path)
        except Exception as error:
            flash(f'Unable to read metadata CSV: {error}')
            return redirect(url_for('index'))

        rows = format_metadata_rows(df)
        selected_tribuid = str(session.get('selected_tribuid'))

        if request.method == 'POST':
            selected = str(request.form.get('tribuid'))
            try:
                selected = validate_tribuid(selected, df)
            except ValueError as error:
                flash(str(error))
                return render_template(
                    'metadata_preview.html',
                    rows=rows,
                    selected_tribuid=request.form.get('tribuid'),
                    required_columns=get_required_columns(),
                )

            session['selected_tribuid'] = selected
            flash(f'Selected TribuId {selected}')
            return redirect(url_for('list_files'))

        return render_template(
            'metadata_preview.html',
            rows=rows,
            selected_tribuid=selected_tribuid,
            required_columns=get_required_columns(),
        )

    @app.route('/files')
    def list_files():
        metadata_path = str(session.get('metadata_path'))
        selected_tribuid = str(session.get('selected_tribuid'))
        downsample_percent = int(request.args.get('downsample', 100))
        cache_key = f'{metadata_path}::{selected_tribuid}'

        if not metadata_path:
            flash('Upload a metadata CSV first')
            return redirect(url_for('index'))
        if not selected_tribuid:
            flash('Please choose a TribuId for the sample')
            return redirect(url_for('metadata_preview'))

        cached = LIST_FILES_CACHE.get(cache_key)
        if cached is not None:
            return render_template(
                'view_files.html',
                files=cached['files'],
                has_tdms=has_tdms_support(),
                selected_tribuid=selected_tribuid,
                file_count=cached['file_count'],
                downsample_percent=downsample_percent,
                mean_power_plots=cached['mean_power_plots'],
                optimal_power_plot=cached['optimal_power_plot'],
                mean_vpp_plot=cached['mean_vpp_plot'],
                available_rload_ids=cached['available_rload_ids'],
            )

        # Load LoadsDescription file
        meta_dir = os.path.dirname(metadata_path)
        try:
            loads_file = find_loads_description_file(meta_dir)
            loads_info_df = load_loads_description(loads_file)
            available_rload_ids = sorted(
                r_id for r_id in loads_info_df['RloadId'].astype(str).str.strip().tolist() if r_id
            )
        except Exception as error:
            flash(f'Unable to read LoadsDescription file: {error}')
            return redirect(url_for('metadata_preview'))

        # Extract experiment folders and calculate power metrics
        try:
            df = parse_metadata_csv(metadata_path)
            sample = get_rows_for_tribuid(df, selected_tribuid)
            if sample.empty:
                raise ValueError(f'No rows found for TribuId {selected_tribuid}')
            experiment_folders = get_experiment_folders(sample, meta_dir)
        except Exception as error:
            flash(f'Unable to read selected TribuId rows: {error}')
            return redirect(url_for('metadata_preview'))

        file_entries = []
        param_store = session.get('peak_params_store', {})

        for experiment in experiment_folders:

            rload_id = experiment.get('RloadId', '')
            load_info = lookup_load_info(loads_info_df, rload_id)
            is_oc_sc = is_open_short_rload(rload_id)

            exp_path = experiment.get('exp_path', '')
            dfData_all, _ = get_cached_experiment_data(exp_path)
            if dfData_all is None:
                raise Exception("Cycles list is empty")

            # Create the base entry first
            entry = {
                'experiment_rel': exp_path,
                'RloadId': experiment.get('RloadId', ''),
                'TribuId': experiment.get('TribuId', ''),
                'SampleIdTriboNeg': experiment.get('SampleIdTriboNeg', ''),
                'SampleIdTriboPos': experiment.get('SampleIdTriboPos', ''),
                'Date': experiment.get('Date', ''),
                'mean_power': None,
                'peak_power': None,
                'mean_vpp': None,
                'rload_missing': load_info['missing'],
                'rload_options': available_rload_ids,
                'is_oc_sc': is_oc_sc,
            }

            # 2. Get Req and Gain (needed for the key)
            if not load_info['missing'] and not is_oc_sc:
                entry['req'] = load_info['Req']
                entry['gain'] = load_info['Gain']
            else:
                entry['req'] = ''
                entry['gain'] = load_info['Gain'] if not load_info['missing'] else ''

            # 3. NOW create the graph_key and get saved params
            TribuId = entry['TribuId']
            req_val = entry['req']
            graph_key = f"{TribuId}_R{req_val}"
            saved_params = param_store.get(graph_key, {})

            # 4. Use saved_params in calculations
            if not load_info['missing'] and not is_oc_sc:
                try:
                    # IMPORTANT: Pass saved_params to all calculation functions
                    entry['mean_power'] = calculate_mean_power(
                        dfData_all, exp_path, float(entry['gain']), float(entry['req']),
                        peak_params=saved_params  # <--- Pass here
                    )
                    entry['peak_power'] = calculate_peak_power(
                        dfData_all, exp_path, float(entry['gain']), float(entry['req']),
                        peak_params=saved_params  # <--- Pass here
                    )
                    entry['mean_vpp'] = calculate_mean_vpp(
                        dfData_all, exp_path, float(entry['gain']),
                        peak_params=saved_params  # <--- Pass here
                    )
                except Exception as error:
                    entry['read_error'] = str(error)

            file_entries.append(entry)

        from collections import defaultdict

        grouped_power_data = defaultdict(list)
        grouped_peak_power_data = defaultdict(list)
        grouped_vpp_data = defaultdict(list)

        for entry in file_entries:
            req = entry.get('req')
            t_id = entry.get('TribuId', 'Unknown')
            if req and not entry.get('is_oc_sc'):
                try:
                    req_val = float(req)
                except ValueError:
                    continue
                if entry.get('mean_power') is not None:
                    grouped_power_data[t_id].append((req_val, entry['mean_power']))
                if entry.get('peak_power') is not None:
                    grouped_peak_power_data[t_id].append((req_val, entry['peak_power']))
                if entry.get('mean_vpp') is not None:
                    grouped_vpp_data[t_id].append((req_val, entry['mean_vpp']))

        # --- REPLACING THE OPTIMAL POINTS CALCULATION IN routes.py ---
        # Calculate Optimal Points
        optimal_points = []
        # Get a unique list of all TribuIds present in either dictionary
        all_tribus = set(grouped_power_data.keys()).union(set(grouped_peak_power_data.keys()))

        for t_id in all_tribus:
            tribu_info = {'TribuId': t_id}

            # 1. Find the point with the maximum MEAN power
            points_mean = grouped_power_data.get(t_id, [])
            if points_mean:
                best_mean = max(points_mean, key=lambda x: x[1])
                tribu_info['req_mean'] = best_mean[0]
                tribu_info['max_power'] = best_mean[1]  # Mean Power
            else:
                tribu_info['max_power'] = 0
                tribu_info['req_mean'] = 'N/A'

            # 2. Find the point with the maximum PEAK power
            points_peak = grouped_peak_power_data.get(t_id, [])
            if points_peak:
                best_peak = max(points_peak, key=lambda x: x[1])
                tribu_info['req_peak'] = best_peak[0]
                tribu_info['max_peak_power'] = best_peak[1]  # Peak Power
            else:
                tribu_info['max_peak_power'] = 0
                tribu_info['req_peak'] = 'N/A'

            if points_mean or points_peak:
                optimal_points.append(tribu_info)

        mean_power_plots = []
        optimal_power_plot = None
        mean_vpp_plot = None

        try:
            if grouped_power_data:
                # Pass both mean power and peak power to the plot
                for t_id in grouped_power_data.keys():
                    # Extract only the data for THIS specific TribuId
                    single_tribu_mean = {t_id: grouped_power_data[t_id]}
                    single_tribu_peak = {t_id: grouped_peak_power_data.get(t_id, [])}
                    # Create the individual plot
                    p_plot = create_mean_power_vs_req_plot(
                        single_tribu_mean,
                        single_tribu_peak,
                        f'Power Analysis vs Resistance: {t_id}',
                        div_id=f'power_plot_{t_id}'
                    )
                    mean_power_plots.append(p_plot)

                    # The "Benchmarking" plot remains a single comparison graph
                from data_processing.preview_service import create_optimal_power_plot
                optimal_power_plot = create_optimal_power_plot(
                    optimal_points,
                    "Optimal Power Comparison across TribuIds"
                )
            if grouped_vpp_data:
                # Keep this as a SINGLE graph (passing all data)
                from data_processing.preview_service import create_mean_vpp_vs_req_plot
                mean_vpp_plot = create_mean_vpp_vs_req_plot(
                    grouped_vpp_data,
                    f'Mean Vpp vs Resistance ({selected_tribuid})'
                )


        except Exception as e:
            print(f"Error generando gráficas: {e}")

        LIST_FILES_CACHE[cache_key] = {
            'files': file_entries,
            'file_count': len(sample),
            'mean_power_plots': mean_power_plots,
            'optimal_power_plot': optimal_power_plot,
            'mean_vpp_plot': mean_vpp_plot,
            'available_rload_ids': available_rload_ids,
        }

        return render_template(
            'view_files.html',
            files=file_entries,
            has_tdms=has_tdms_support(),
            selected_tribuid=selected_tribuid,
            file_count=len(sample),
            downsample_percent=downsample_percent,
            mean_power_plots=mean_power_plots,
            optimal_power_plot=optimal_power_plot,
            mean_vpp_plot=mean_vpp_plot,
            available_rload_ids=available_rload_ids,
        )

    @app.route('/view')
    def view_experiment():
        metadata_path = str(session.get('metadata_path'))
        exp_path = request.args.get('rel')  # This is the primary file clicked (Motor)
        if exp_path:
            exp_path = exp_path.replace('//', '/').replace('\\\\', '\\')
        downsample_percent = int(request.args.get('downsample', 80))

        if not metadata_path or not exp_path:
            flash('Missing parameters')
            return redirect(url_for('index'))

        # 1. Get the identifiers from the URL
        TribuId = request.args.get('TribuId', 'Unknown')
        req_val = request.args.get('req', '0')
        graph_key = f"{TribuId}_R{req_val}"  # Example: "Tribu123_R1000"
        default_graphs = ['voltage', 'power', 'position', 'force', 'is_moving', 'up_down']
        requested_graphs = request.args.getlist('graphs')
        selected_graphs = [g for g in requested_graphs if g in default_graphs] if requested_graphs else default_graphs
        if not selected_graphs:
            selected_graphs = default_graphs
        auto_gain_values = request.args.getlist('auto_gain')
        auto_gain = auto_gain_values[-1] == '1' if auto_gain_values else True
        notch_enable_values = request.args.getlist('notch_enable')
        notch_enabled = notch_enable_values[-1] == '1' if notch_enable_values else False
        rload_id_value = request.args.get('rload_id', '').strip()
        rload_options = []

        loads_info_df = None
        if metadata_path:
            try:
                meta_dir = os.path.dirname(metadata_path)
                loads_file = find_loads_description_file(meta_dir)
                loads_info_df = load_loads_description(loads_file)
                rload_options = sorted(
                    r_id for r_id in loads_info_df['RloadId'].astype(str).str.strip().tolist() if r_id
                )
            except Exception:
                loads_info_df = None

        if 'peak_params_store' not in session:
            session['peak_params_store'] = {}

        # 2. Capture parameters from URL and SAVE them
        if request.args.get('reset_peaks') == '1':
            store = session.get('peak_params_store', {})
            if graph_key in store:
                del store[graph_key]
                session['peak_params_store'] = store
        else:
            store = session.get('peak_params_store', {})
            if graph_key not in store:
                store[graph_key] = {}

            # For each parameter, if it's explicitly present in the URL:
            # - If it has a value, save it.
            # - If it's an empty string, delete it from the store (revert to auto).
            for param, p_type, key in [
                ('pk_height', float, 'height'),
                ('pk_prom', float, 'prominence'),
                ('pk_dist', int, 'distance'),
                ('pk_cutoff', float, 'cutoff')
            ]:
                if param in request.args:
                    val_str = request.args.get(param, '').strip()
                    if val_str == '':
                        # User cleared the field -> remove from store to revert to Auto
                        store[graph_key].pop(key, None)
                    else:
                        try:
                            store[graph_key][key] = p_type(val_str)
                        except ValueError:
                            pass

            session['peak_params_store'] = store  # Trigger session save
        # 3. LOAD the final parameters (Saved + Defaults)
        # This ensures that even if url_params is empty, we get the history
        final_peak_params = {'cutoff': 0.1}  # Default fallback
        final_peak_params.update(session['peak_params_store'].get(graph_key, {}))
        try:
            dfData_all, cycle_markers = get_cached_experiment_data(exp_path)
            if dfData_all is None:
                flash(f'Error Cycles list is empty')
                return redirect(url_for('list_files'))

            gain_value = request.args.get('gain', '').strip()
            req_value = request.args.get('req', '').strip()
            is_oc_sc = is_open_short_rload(rload_id_value)
            is_sc = is_short_circuit_rload(rload_id_value)
            signal_mode = 'current' if is_sc else 'voltage'

            if rload_id_value and loads_info_df is not None and not is_oc_sc:
                load_info = lookup_load_info(loads_info_df, rload_id_value)
                if load_info.get('missing'):
                    flash(f"RloadId '{rload_id_value}' not found in LoadsDescription. Keeping current Req/Gain values.")
                else:
                    req_from_load = str(load_info.get('Req', '') or '').strip()
                    gain_from_load = str(load_info.get('Gain', '') or '').strip()
                    if req_from_load:
                        req_value = req_from_load
                    if auto_gain and gain_from_load:
                        gain_value = gain_from_load

            try:
                gain = float(gain_value) if gain_value else None
            except ValueError:
                flash(f"Invalid gain value '{gain_value}'.")
                gain = None

            try:
                req = float(req_value) if req_value else None
            except ValueError:
                flash(f"Invalid Req value '{req_value}'.")
                req = None

            if is_oc_sc:
                selected_graphs = [g for g in selected_graphs if g != 'power']
                if not selected_graphs:
                    selected_graphs = ['voltage', 'position', 'force', 'is_moving', 'up_down']
                req = None
                req_value = ''
            if is_sc:
                notch_enabled = False

            notch_freq_value = request.args.get('notch_freq', '').strip()
            notch_q_value = request.args.get('notch_q', '').strip()
            try:
                notch_freq = float(notch_freq_value) if notch_freq_value else 50.0
                if notch_freq <= 0:
                    raise ValueError
            except ValueError:
                flash(f"Invalid notch frequency '{notch_freq_value}'. Using default 50 Hz.")
                notch_freq = 50.0
                notch_freq_value = '50'

            try:
                notch_q = float(notch_q_value) if notch_q_value else 30.0
                if notch_q <= 0:
                    raise ValueError
            except ValueError:
                flash(f"Invalid notch Q factor '{notch_q_value}'. Using default 30.")
                notch_q = 30.0
                notch_q_value = '30'

            final_notch_params = {
                'enabled': notch_enabled,
                'frequency': notch_freq,
                'q': notch_q,
                'frequency_display': notch_freq_value if notch_freq_value else '50',
                'q_display': notch_q_value if notch_q_value else '30'
            }

            try:
                plot_html = create_combined_motor_daq_plot(
                    exp_path=exp_path,
                    exp_df=dfData_all,
                    title=f"Combined Analysis: {exp_path}",
                    downsample_percent=downsample_percent,
                    gain=gain,
                    req=req,
                    peak_params=final_peak_params,
                    selected_graphs=selected_graphs,
                    notch_params=final_notch_params,
                    signal_mode=signal_mode,
                    cycle_markers=cycle_markers
                )
            except Exception as e:
                flash(f"Could not load associated voltage file: {e}")
                plot_html = create_plot_html(
                    dfData_all,
                    exp_path,
                    f"Motor Data: {exp_path}",
                    downsample_percent,
                    gain=gain,
                    plot_mode='both',
                    req=req,
                    peak_params=final_peak_params,
                    include_graphs=selected_graphs,
                    notch_params=final_notch_params,
                    signal_mode=signal_mode,
                    cycle_markers=cycle_markers
                )

            mean_power = None
            if gain is not None and req is not None and not is_oc_sc:
                mean_power = calculate_mean_power(dfData_all, exp_path, gain, req, peak_params=final_peak_params)

            df_info = f'{len(dfData_all)} rows × {len(dfData_all.columns)} columns'
            return render_template(
                'plot_view.html',
                plot=plot_html,
                filename=exp_path,
                df_info=df_info,
                downsample_percent=downsample_percent,
                gain_display=gain_value,
                req_display=req_value,
                mean_power=mean_power,
                peak_params=final_peak_params,
                selected_graphs=selected_graphs,
                tribu_id=TribuId,
                rload_id_display=rload_id_value,
                rload_options=rload_options,
                auto_gain=auto_gain,
                notch_params=final_notch_params,
                is_oc_sc=is_oc_sc,
                is_sc=is_sc
            )
        except Exception as error:
            import traceback
            print("ERROR DETECTADO EN VIEW_FILE:")
            print(traceback.format_exc())
            flash(f'Failed to process file: {error}')
            return redirect(url_for('list_files'))
