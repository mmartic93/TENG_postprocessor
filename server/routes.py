import os
from flask import render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from data_processing.LoadData import ExtractCycles
from data_processing.preview_service import create_combined_motor_daq_plot
import pandas as pd
import numpy as np

from server.config import UPLOAD_FOLDER, MAX_PREVIEW_ROWS
from data_processing.file_resolver import resolve_relative_path, file_exists, normalize_display_path
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
    has_tdms_support,
    create_no_ra_plot,
    get_plateau_peaks,
    get_signal_peaks,
    create_comparison_summary_plot
)
from data_processing.validators import validate_tribuid


def register_routes(app):
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
                flash('Unsupported metadata file type (use .csv or .ods)')
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
            return redirect(url_for('metadata_preview'))

        return render_template('index.html')

    @app.route('/metadata', methods=['GET', 'POST'])
    def metadata_preview():
        metadata_path = session.get('metadata_path')
        if not metadata_path:
            flash('Upload a metadata CSV first')
            return redirect(url_for('index'))

        try:
            df = parse_metadata_csv(metadata_path)
        except Exception as error:
            flash(f'Unable to read metadata CSV: {error}')
            return redirect(url_for('index'))

        rows = format_metadata_rows(df)
        selected_tribuid = session.get('selected_tribuid')

        if request.method == 'POST':
            selected = request.form.get('tribuid')
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
        metadata_path = session.get('metadata_path')
        selected_tribuid = session.get('selected_tribuid')
        downsample_percent = int(request.args.get('downsample', 100))

        if not metadata_path:
            flash('Upload a metadata CSV first')
            return redirect(url_for('index'))
        if not selected_tribuid:
            flash('Please choose a TribuId for the sample')
            return redirect(url_for('metadata_preview'))

        meta_dir = os.path.dirname(metadata_path)
        loads_description_error = None
        loads_info_df = None
        try:
            loads_file = find_loads_description_file(meta_dir)
            loads_info_df = load_loads_description(loads_file)
        except Exception as error:
            loads_description_error = str(error)

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
            exp_path = experiment.get('exp_path', '')
            Cycles_list = ExtractCycles(exp_path)
            if len(Cycles_list) == 0:
                raise Exception("Cycles list is empty")
            dfData_all = pd.concat(Cycles_list, ignore_index=True)

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
            }

            # 2. Get Req and Gain (needed for the key)
            if loads_info_df is not None:
                load_info = lookup_load_info(loads_info_df, experiment.get('RloadId', ''))
                entry['req'] = load_info['Req']
                entry['gain'] = load_info['Gain']
                entry['rload_missing'] = load_info['missing']
            else:
                entry['req'] = '0'
                entry['gain'] = '1'

            # 3. NOW create the graph_key and get saved params
            TribuId = entry['TribuId']
            req_val = entry['req']
            graph_key = f"{TribuId}_R{req_val}"
            saved_params = param_store.get(graph_key, {})

            # 4. Use saved_params in calculations
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
            if req:
                req_val = float(req)
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

        return render_template(
            'view_files.html',
            files=file_entries,
            has_tdms=has_tdms_support(),
            selected_tribuid=selected_tribuid,
            file_count=len(sample),
            downsample_percent=downsample_percent,
            loads_description_error=loads_description_error,
            mean_power_plots=mean_power_plots,
            optimal_power_plot=optimal_power_plot,
            mean_vpp_plot=mean_vpp_plot,
        )

    @app.route('/view')
    def view_experiment():
        metadata_path = session.get('metadata_path')
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
            Cycles_list = ExtractCycles(exp_path)
            if len(Cycles_list) == 0:
                flash(f'Error Cycles list is empty')
                return redirect(url_for('list_files'))

            dfData_all = pd.concat(Cycles_list, ignore_index=True)

            gain_value = request.args.get('gain', '').strip()
            gain = float(gain_value) if gain_value else None
            req_value = request.args.get('req', '').strip()
            req = float(req_value) if req_value else None

            plot_mode = request.args.get('plot_mode', 'voltage')
            try:
                plot_html = create_combined_motor_daq_plot(
                    exp_path=exp_path,
                    exp_df=dfData_all,
                    title=f"Combined Analysis: {exp_path}",
                    downsample_percent=downsample_percent,
                    gain=gain
                )
            except Exception as e:
                flash(f"Could not load associated voltage file: {e}")
                plot_html = create_plot_html(dfData_all, exp_path, f"Motor Data: {exp_path}", downsample_percent,
                                             peak_params=final_peak_params)

            mean_power = None
            if plot_mode == 'power' and gain is not None and req is not None:
                # Aquí también podrías pasar peak_params si calculate_mean_power lo requiere
                mean_power = calculate_mean_power(dfData_all, exp_path, gain, req, peak_params=final_peak_params)

            df_info = f'{len(dfData_all)} rows × {len(dfData_all.columns)} columns'
            return render_template(
                'plot_view.html',
                plot=plot_html,
                filename=exp_path,
                df_info=df_info,
                downsample_percent=downsample_percent,
                gain_display=gain_value,
                plot_mode=plot_mode,
                req_display=req_value,
                mean_power=mean_power,
                peak_params=final_peak_params  # PASAR A LA PLANTILLA
            )
        except Exception as error:
            import traceback
            print("ERROR DETECTADO EN VIEW_FILE:")
            print(traceback.format_exc())  # Esto imprimirá el error real en tu terminal negra
            flash(f'Failed to process file: {error}')
            return redirect(url_for('list_files'))

    @app.route('/no_ra', methods=['POST'])
    def no_ra_analysis():
        # DEBUG PRINTS
        voc_files = request.files.getlist('voc_files')
        isc_files = request.files.getlist('isc_files')
        print(f"DEBUG: Received {len(voc_files)} VOC files and {len(isc_files)} ISC files")

        if not voc_files or not voc_files[0].filename:
            print("DEBUG: VOC files list was empty or first file has no name")
            flash("Please select at least one VOC file.")
            return redirect(url_for('index'))

        try:
            voc_data, isc_data = [], []
            voc_summary_stats, isc_summary_stats = [], []

            for f in voc_files:
                if f.filename:
                    df = pd.read_csv(f)
                    voc_data.append({'name': f.filename, 'df': df})
                    # Attempt to find the voltage column by name first, then by index
                    y_vals = df.iloc[:, 1].values
                    _, _, m_max, _, _ = get_plateau_peaks(y_vals)
                    voc_summary_stats.append({'name': f.filename, 'max_v': abs(m_max)})

            for f in isc_files:
                if f.filename:
                    df = pd.read_csv(f)
                    isc_data.append({'name': f.filename, 'df': df})
                    y_vals = df.iloc[:, 1].values
                    # Ensure we use a safe prominence calculation
                    prom = np.std(y_vals) * 3 if len(y_vals) > 0 else 1
                    isc_params = {'distance': 20, 'prominence': prom}
                    _, _, _, _, vpp = get_signal_peaks(y_vals, custom_params=isc_params)
                    isc_summary_stats.append({'name': f.filename, 'vpp_i': vpp})

            individual_plots_html = create_no_ra_plot(voc_data, isc_data, "Multi-File Analysis")
            comparison_plot_html = create_comparison_summary_plot(voc_summary_stats, isc_summary_stats)
            combined_html = comparison_plot_html + "<hr>" + individual_plots_html

            return render_template(
                'plot_view.html',
                plot=combined_html,
                filename="No Ra Analysis Results",
                df_info=f"{len(voc_files)} Voc, {len(isc_files)} Isc",
                downsample_percent=100,
                plot_mode='No Ra Summary',
                gain_display=None, req_display=None, mean_power=None
            )
        except Exception as e:
            print(f"DEBUG ERROR: {str(e)}")  # This will print the exact error to your terminal
            flash(f"Error processing files: {e}")
            return redirect(url_for('index'))