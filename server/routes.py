import os
import json
from flask import render_template, request, redirect, url_for, session, flash
from data_processing.LoadData import ExtractCycles
from data_processing.preview_service import create_combined_motor_daq_plot
import pandas as pd
import numpy as np
from collections import defaultdict

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
    create_cycles_overlay_plot,
    create_signal_fft_plot,
    create_comparison_summary_plot,
    apply_gain_to_dataframe,
    get_plateau_peaks,
    get_signal_peaks,
)
from data_processing.validators import validate_tribuid

LIST_FILES_CACHE = {}
EXPERIMENT_DATA_CACHE = {}


def register_routes(app):
    def is_open_short_rload(rload_id: str) -> bool:
        return str(rload_id or '').strip().upper() in {'OC', 'SC'}

    def is_short_circuit_rload(rload_id: str) -> bool:
        return str(rload_id or '').strip().upper() == 'SC'

    def load_experiment_cycles(exp_path: str):
        cached = EXPERIMENT_DATA_CACHE.get(exp_path)
        if isinstance(cached, dict) and 'cycles' in cached:
            return cached['cycles']
        cycles_list = ExtractCycles(exp_path)
        if isinstance(cached, dict):
            cached['cycles'] = cycles_list
            EXPERIMENT_DATA_CACHE[exp_path] = cached
        else:
            EXPERIMENT_DATA_CACHE[exp_path] = {'cycles': cycles_list}
        return cycles_list

    def normalize_cycle_range(total_cycles: int, start_raw, end_raw):
        if total_cycles <= 0:
            return 0, 0
        try:
            start_cycle = int(str(start_raw).strip()) if str(start_raw).strip() else 1
        except ValueError:
            start_cycle = 1
        try:
            end_cycle = int(str(end_raw).strip()) if str(end_raw).strip() else total_cycles
        except ValueError:
            end_cycle = total_cycles
        start_cycle = max(1, min(start_cycle, total_cycles))
        end_cycle = max(1, min(end_cycle, total_cycles))
        if end_cycle < start_cycle:
            end_cycle = start_cycle
        return start_cycle, end_cycle

    def build_cycle_window_dataframe(cycles_list, start_cycle: int, end_cycle: int):
        if not cycles_list:
            return None, []
        if start_cycle < 1 or end_cycle < start_cycle:
            return None, []

        selected_cycles = cycles_list[start_cycle - 1:end_cycle]
        if not selected_cycles:
            return None, []

        df_data = pd.concat(selected_cycles, ignore_index=True)
        cycle_markers = []
        offset = 0
        for cycle_df in selected_cycles[:-1]:
            offset += len(cycle_df)
            if offset >= len(df_data):
                break
            if 'Time' in df_data.columns:
                cycle_markers.append(float(df_data.iloc[offset]['Time']))
            else:
                cycle_markers.append(float(offset))
        return df_data, cycle_markers

    def parse_notch_csv_values(raw_value: str, field_label: str) -> list[float]:
        parts = [part.strip() for part in str(raw_value).split(',') if part.strip()]
        if not parts:
            return []
        values = []
        for part in parts:
            try:
                value = float(part)
            except ValueError:
                raise ValueError(f'Invalid {field_label} value "{part}"')
            if value <= 0:
                raise ValueError(f'{field_label} values must be > 0')
            values.append(value)
        return values

    def get_experiment_overrides():
        return session.get('experiment_overrides', {})

    def save_experiment_override(exp_path: str, override: dict):
        overrides = session.get('experiment_overrides', {})
        previous = overrides.get(exp_path, {})
        merged = dict(previous)
        merged.update(override)
        if previous != merged:
            overrides[exp_path] = merged
            session['experiment_overrides'] = overrides
            LIST_FILES_CACHE.clear()

    def as_bool(value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in {'1', 'true', 'yes', 'on'}:
            return True
        if text in {'0', 'false', 'no', 'off'}:
            return False
        return default

    def get_keithley_conversion_factors(exp_path: str):
        json_path = os.path.join(exp_path, 'experiment_metadata.json')
        if not os.path.exists(json_path):
            return [], 'experiment_metadata.json not found'
        try:
            with open(json_path, 'r', encoding='utf-8') as file_handle:
                metadata = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return [], 'unable to read experiment_metadata.json'

        conversion_info = []
        for task in metadata.get('DAQTasks', []):
            channels = task.get('DAQ_CHANNELS', {})
            if not isinstance(channels, dict):
                continue
            for channel_name, channel_config in channels.items():
                if not isinstance(channel_config, dict):
                    continue
                conversion_info.append({
                    'channel': str(channel_name),
                    'conversion_factor': channel_config.get('conversion_factor'),
                })
        return conversion_info, None

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
            session.pop('experiment_overrides', None)
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
        unique_tribuids = list(dict.fromkeys(df['TribuId'].astype(str).str.strip().tolist()))
        selected_tribuids = [value.strip() for value in selected_tribuid.split(',') if value.strip()]

        if request.method == 'POST':
            selected_values = [value.strip() for value in request.form.getlist('tribuid') if value.strip()]
            selected = ', '.join(selected_values)
            try:
                selected = validate_tribuid(selected, df)
            except ValueError as error:
                flash(str(error))
                return render_template(
                    'metadata_preview.html',
                    rows=rows,
                    selected_tribuid=selected,
                    selected_tribuids=selected_values,
                    tribuid_options=unique_tribuids,
                    required_columns=get_required_columns(),
                )

            session['selected_tribuid'] = selected
            flash(f'Selected TribuId(s): {selected}', 'success')
            return redirect(url_for('list_files'))

        return render_template(
            'metadata_preview.html',
            rows=rows,
            selected_tribuid=selected_tribuid,
            selected_tribuids=selected_tribuids,
            tribuid_options=unique_tribuids,
            required_columns=get_required_columns(),
        )

    @app.route('/files')
    def list_files():
        metadata_path = str(session.get('metadata_path'))
        selected_tribuid = str(session.get('selected_tribuid'))
        downsample_percent = int(request.args.get('downsample', 100))
        selected_oc_exp = request.args.get('selected_oc_exp', '').strip()
        selected_sc_exp = request.args.get('selected_sc_exp', '').strip()
        cache_key = f'{metadata_path}::{selected_tribuid}::{selected_oc_exp}::{selected_sc_exp}'

        if not metadata_path:
            flash('Upload a metadata CSV first')
            return redirect(url_for('index'))
        if not selected_tribuid:
            flash('Please choose a TribuId for the sample')
            return redirect(url_for('metadata_preview'))

        if request.args.get('action') == 'update_metrics':
            rel_values = request.args.getlist('rel')
            rload_values = request.args.getlist('rload_id')
            gain_values = request.args.getlist('gain')
            req_values = request.args.getlist('req')
            auto_gain_values = request.args.getlist('auto_gain')
            overlay_start_values = request.args.getlist('overlay_cycle_start')
            overlay_end_values = request.args.getlist('overlay_cycle_end')
            reset_gain_rel = request.args.get('reset_gain_rel', '').strip()
            selected_oc_exp = request.args.get('selected_oc_exp', '').strip()
            selected_sc_exp = request.args.get('selected_sc_exp', '').strip()

            for index, override_exp_path in enumerate(rel_values):
                override_exp_path = str(override_exp_path or '').strip()
                if not override_exp_path:
                    continue
                rload_id = str(rload_values[index]).strip() if index < len(rload_values) else ''
                gain = str(gain_values[index]).strip() if index < len(gain_values) else ''
                req = str(req_values[index]).strip() if index < len(req_values) else ''
                auto_gain = auto_gain_values[index] == '1' if index < len(auto_gain_values) else False
                overlay_cycle_start = str(overlay_start_values[index]).strip() if index < len(overlay_start_values) else ''
                overlay_cycle_end = str(overlay_end_values[index]).strip() if index < len(overlay_end_values) else ''
                if reset_gain_rel and override_exp_path == reset_gain_rel:
                    gain = ''
                    req = ''
                    auto_gain = True
                save_experiment_override(override_exp_path, {
                    'rload_id': rload_id,
                    'gain': gain,
                    'req': req,
                    'auto_gain': auto_gain,
                    'overlay_cycle_start': overlay_cycle_start,
                    'overlay_cycle_end': overlay_cycle_end,
                })
            return redirect(url_for(
                'list_files',
                downsample=downsample_percent,
                selected_oc_exp=selected_oc_exp,
                selected_sc_exp=selected_sc_exp
            ))

        cached = LIST_FILES_CACHE.get(cache_key)
        if cached is not None:
            return render_template(
                'view_files.html',
                files=cached['files'],
                selected_tribuid=selected_tribuid,
                file_count=cached['file_count'],
                downsample_percent=downsample_percent,
                mean_power_plots=cached['mean_power_plots'],
                optimal_power_plot=cached['optimal_power_plot'],
                mean_vpp_plot=cached['mean_vpp_plot'],
                oc_sc_comparison_plot=cached.get('oc_sc_comparison_plot'),
                oc_candidates=cached.get('oc_candidates', []),
                sc_candidates=cached.get('sc_candidates', []),
                selected_oc_exp=cached.get('selected_oc_exp', ''),
                selected_sc_exp=cached.get('selected_sc_exp', ''),
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

        experiment_list = []
        param_store = session.get('peak_params_store', {})
        experiment_overrides = get_experiment_overrides()
        oc_candidates = []
        sc_candidates = []

        def find_metric_column(df: pd.DataFrame, keywords):
            for col_name in df.columns:
                if any(keyword in str(col_name).lower() for keyword in keywords):
                    return col_name
            return None

        for experiment in experiment_folders:

            # 1. Extract RloadId and check if it's open/short circuit
            exp_path = experiment.get('exp_path', '')
            saved_override = experiment_overrides.get(exp_path, {})
            rload_id = str(saved_override.get('rload_id') or experiment.get('RloadId', '')).strip()
            load_info = lookup_load_info(loads_info_df, rload_id)  # Extract Req and Gain from LoadsDescription
            is_oc_sc = is_open_short_rload(rload_id)
            auto_gain = bool(saved_override.get('auto_gain', True))

            # 2. Load the selected cycle range from experiment data
            cycles_list = load_experiment_cycles(exp_path)
            overlay_cycles_max = len(cycles_list) if cycles_list else 0
            overlay_cycle_start, overlay_cycle_end = normalize_cycle_range(
                overlay_cycles_max,
                saved_override.get('overlay_cycle_start', ''),
                saved_override.get('overlay_cycle_end', '')
            )
            dfData_all, _ = build_cycle_window_dataframe(cycles_list, overlay_cycle_start, overlay_cycle_end)
            if dfData_all is None:
                raise Exception("Cycles list is empty")

            # 3. Create the experiment structure with all necessary information
            experiment_data = {
                'experiment_rel': exp_path,
                'RloadId': rload_id,
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
                'auto_gain': auto_gain,
                'overlay_cycle_start': overlay_cycle_start,
                'overlay_cycle_end': overlay_cycle_end,
                'overlay_cycles_max': overlay_cycles_max,
            }

            # 4. Add Req and Gain values in the experiment_data, considering the RloadId and whether it's open/short circuit
            if not load_info['missing'] and not is_oc_sc:
                experiment_data['req'] = float(load_info['Req'])
                experiment_data['gain'] = float(load_info['Gain']) if auto_gain else ''
            else:
                experiment_data['req'] = ''
                experiment_data['gain'] = ''

            override_req = str(saved_override.get('req', '')).strip()
            if override_req:
                try:
                    experiment_data['req'] = float(override_req)
                except ValueError:
                    pass
            override_gain = str(saved_override.get('gain', '')).strip()
            if override_gain:
                try:
                    experiment_data['gain'] = float(override_gain)
                except ValueError:
                    pass

            # 5. Create the graph_key
            TribuId = experiment_data['TribuId']
            req_val = experiment_data['req']
            graph_key = f"{TribuId}_R{req_val}"

            # 6. Calculate mean power, peak power, and mean Vpp only if the RloadId is valid and not open/short circuit
            saved_params = param_store.get(graph_key, {})
            if not load_info['missing'] and not is_oc_sc:
                try:
                    calc_gain = float(experiment_data['gain'])
                    calc_req = float(experiment_data['req'])
                    # IMPORTANT: Pass saved_params to all calculation functions
                    experiment_data['mean_power'] = calculate_mean_power(
                        dfData_all, exp_path, calc_gain, calc_req,
                        peak_params=saved_params
                    )
                    experiment_data['peak_power'] = calculate_peak_power(
                        dfData_all, exp_path, calc_gain, calc_req,
                        peak_params=saved_params
                    )
                    experiment_data['mean_vpp'] = calculate_mean_vpp(
                        dfData_all, exp_path, calc_gain,
                        peak_params=saved_params
                    )
                except Exception as error:
                    experiment_data['read_error'] = str(error)
            elif is_oc_sc:
                try:
                    calc_gain = float(experiment_data['gain']) if str(experiment_data.get('gain', '')).strip() else None
                except (TypeError, ValueError):
                    calc_gain = None
                try:
                    analysis_df = apply_gain_to_dataframe(dfData_all, exp_path, calc_gain)
                    exp_name = os.path.basename(str(exp_path).rstrip('\\/'))
                    rload_upper = str(rload_id).strip().upper()
                    if rload_upper == 'SC':
                        current_col = find_metric_column(analysis_df, ['current', 'isc', 'ampere', 'input 1'])
                        if current_col:
                            y_vals = analysis_df[current_col].astype(float).values
                            isc_params = {'distance': 20, 'prominence': np.std(y_vals) * 3}
                            _, _, _, _, vpp_i = get_signal_peaks(y_vals, custom_params=isc_params, cutoff=0.3)
                            sc_candidates.append({
                                'exp_path': exp_path,
                                'name': exp_name,
                                'vpp_i': float(vpp_i),
                            })
                    elif rload_upper == 'OC':
                        voltage_col = find_metric_column(analysis_df, ['voltage', 'input 0', 'voc'])
                        if voltage_col:
                            y_vals = analysis_df[voltage_col].astype(float).values
                            _, _, mean_voc, _, _ = get_plateau_peaks(y_vals, threshold_percentile=80, cutoff=0.05)
                            oc_candidates.append({
                                'exp_path': exp_path,
                                'name': exp_name,
                                'max_v': abs(float(mean_voc)),
                            })
                except Exception:
                    pass

            experiment_list.append(experiment_data)

        grouped_power_data = defaultdict(list)
        grouped_peak_power_data = defaultdict(list)
        grouped_vpp_data = defaultdict(list)

        for experiment_data in experiment_list:
            req = experiment_data.get('req')
            t_id = experiment_data.get('TribuId', 'Unknown')
            if req and not experiment_data.get('is_oc_sc'):
                try:
                    req_val = float(req)
                except ValueError:
                    continue
                if experiment_data.get('mean_power') is not None:
                    grouped_power_data[t_id].append((req_val, experiment_data['mean_power']))
                if experiment_data.get('peak_power') is not None:
                    grouped_peak_power_data[t_id].append((req_val, experiment_data['peak_power']))
                if experiment_data.get('mean_vpp') is not None:
                    grouped_vpp_data[t_id].append((req_val, experiment_data['mean_vpp']))

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
        oc_sc_comparison_plot = None

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
            if oc_candidates and sc_candidates:
                oc_by_path = {item['exp_path']: item for item in oc_candidates}
                sc_by_path = {item['exp_path']: item for item in sc_candidates}
                selected_oc_candidate = oc_by_path.get(selected_oc_exp) if selected_oc_exp else None
                selected_sc_candidate = sc_by_path.get(selected_sc_exp) if selected_sc_exp else None
                if selected_oc_candidate is None:
                    selected_oc_candidate = oc_candidates[0]
                    selected_oc_exp = selected_oc_candidate['exp_path']
                if selected_sc_candidate is None:
                    selected_sc_candidate = sc_candidates[0]
                    selected_sc_exp = selected_sc_candidate['exp_path']

                oc_sc_comparison_plot = create_comparison_summary_plot(
                    [{'name': selected_oc_candidate['name'], 'max_v': selected_oc_candidate['max_v']}],
                    [{'name': selected_sc_candidate['name'], 'vpp_i': selected_sc_candidate['vpp_i']}]
                )

        except Exception as e:
            print(f"Error generating plots: {e}")

        LIST_FILES_CACHE[cache_key] = {
            'files': experiment_list,
            'file_count': len(sample),
            'mean_power_plots': mean_power_plots,
            'optimal_power_plot': optimal_power_plot,
            'mean_vpp_plot': mean_vpp_plot,
            'oc_sc_comparison_plot': oc_sc_comparison_plot,
            'oc_candidates': [{'exp_path': item['exp_path'], 'name': item['name']} for item in oc_candidates],
            'sc_candidates': [{'exp_path': item['exp_path'], 'name': item['name']} for item in sc_candidates],
            'selected_oc_exp': selected_oc_exp,
            'selected_sc_exp': selected_sc_exp,
            'available_rload_ids': available_rload_ids,
        }

        return render_template(
            'view_files.html',
            files=experiment_list,
            selected_tribuid=selected_tribuid,
            file_count=len(sample),
            downsample_percent=downsample_percent,
            mean_power_plots=mean_power_plots,
            optimal_power_plot=optimal_power_plot,
            mean_vpp_plot=mean_vpp_plot,
            oc_sc_comparison_plot=oc_sc_comparison_plot,
            oc_candidates=[{'exp_path': item['exp_path'], 'name': item['name']} for item in oc_candidates],
            sc_candidates=[{'exp_path': item['exp_path'], 'name': item['name']} for item in sc_candidates],
            selected_oc_exp=selected_oc_exp,
            selected_sc_exp=selected_sc_exp,
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

        experiment_overrides = get_experiment_overrides()
        saved_override = experiment_overrides.get(exp_path, {}) if exp_path else {}

        # 1. Get the identifiers from the URL
        TribuId = request.args.get('TribuId', 'Unknown')
        req_val = request.args.get('req', str(saved_override.get('req', '0')))
        graph_key = f"{TribuId}_R{req_val}"  # Example: "Tribu123_R1000"
        default_graphs = ['voltage', 'power', 'fft', 'position', 'force', 'is_moving', 'up_down']
        requested_graphs = request.args.getlist('graphs')
        selected_graphs = [g for g in requested_graphs if g in default_graphs] if requested_graphs else default_graphs
        if not selected_graphs:
            selected_graphs = default_graphs
        auto_gain_values = request.args.getlist('auto_gain')
        auto_gain = auto_gain_values[-1] == '1' if auto_gain_values else bool(saved_override.get('auto_gain', True))
        notch_enable_values = request.args.getlist('notch_enable')
        notch_enabled = notch_enable_values[-1] == '1' if notch_enable_values else False
        raw_signal_values = request.args.getlist('show_raw_signal')
        show_raw_signal = raw_signal_values[-1] == '1' if raw_signal_values else True
        filtered_signal_values = request.args.getlist('show_filtered_signal')
        show_filtered_signal = filtered_signal_values[-1] == '1' if filtered_signal_values else True
        overlay_raw_signal_values = request.args.getlist('overlay_show_raw_signal')
        overlay_show_raw_signal = (
            overlay_raw_signal_values[-1] == '1'
            if overlay_raw_signal_values
            else as_bool(saved_override.get('overlay_show_raw_signal', True), True)
        )
        overlay_filtered_signal_values = request.args.getlist('overlay_show_filtered_signal')
        overlay_show_filtered_signal = (
            overlay_filtered_signal_values[-1] == '1'
            if overlay_filtered_signal_values
            else as_bool(saved_override.get('overlay_show_filtered_signal', False), False)
        )
        rload_id_value = request.args.get('rload_id', '').strip()
        if not rload_id_value:
            rload_id_value = str(saved_override.get('rload_id', '')).strip()
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
            cycles_list = load_experiment_cycles(exp_path)
            overlay_cycles_max = len(cycles_list) if cycles_list else 0
            overlay_cycle_start, overlay_cycle_end = normalize_cycle_range(
                overlay_cycles_max,
                request.args.get('overlay_cycle_start', saved_override.get('overlay_cycle_start', '')),
                request.args.get('overlay_cycle_end', saved_override.get('overlay_cycle_end', ''))
            )
            dfData_all, cycle_markers = build_cycle_window_dataframe(
                cycles_list,
                overlay_cycle_start,
                overlay_cycle_end
            )
            if dfData_all is None:
                flash(f'Error Cycles list is empty')
                return redirect(url_for('list_files'))

            gain_value = request.args.get('gain', str(saved_override.get('gain', ''))).strip()
            req_value = request.args.get('req', str(saved_override.get('req', ''))).strip()
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
                    selected_graphs = ['voltage', 'fft', 'position', 'force', 'is_moving', 'up_down']
                req = None
                req_value = ''

            save_experiment_override(exp_path, {
                'rload_id': rload_id_value,
                'gain': gain_value,
                'req': req_value,
                'auto_gain': auto_gain,
                'overlay_cycle_start': str(overlay_cycle_start),
                'overlay_cycle_end': str(overlay_cycle_end),
                'overlay_show_raw_signal': overlay_show_raw_signal,
                'overlay_show_filtered_signal': overlay_show_filtered_signal,
            })

            notch_freq_value = request.args.get('notch_freq', '').strip()
            notch_q_value = request.args.get('notch_q', '').strip()
            try:
                notch_frequencies = parse_notch_csv_values(notch_freq_value, 'notch frequency')
                if not notch_frequencies:
                    notch_frequencies = [50.0]
                    notch_freq_value = '50'
                notch_q_values = parse_notch_csv_values(notch_q_value, 'notch Q factor')
                if not notch_q_values:
                    notch_q_values = [30.0]
                    notch_q_value = '30'
                if len(notch_q_values) == 1 and len(notch_frequencies) > 1:
                    notch_q_values = notch_q_values * len(notch_frequencies)
                elif len(notch_q_values) != len(notch_frequencies):
                    raise ValueError('Notch Q factor count must be 1 or equal to notch frequency count')
            except ValueError as error:
                flash(f'{error}. Using default notch parameters (50 Hz, Q=30).')
                notch_frequencies = [50.0]
                notch_q_values = [30.0]
                notch_freq_value = '50'
                notch_q_value = '30'

            final_notch_params = {
                'enabled': notch_enabled,
                'frequency': notch_frequencies,
                'q': notch_q_values,
                'show_raw_signal': show_raw_signal,
                'show_filtered_signal': show_filtered_signal,
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

            cycles_overlay_html = create_cycles_overlay_plot(
                cycles_list,
                exp_path=exp_path,
                gain=gain,
                signal_mode=signal_mode,
                cycle_start=overlay_cycle_start,
                cycle_end=overlay_cycle_end,
                notch_params=final_notch_params,
                show_raw_signal=overlay_show_raw_signal,
                show_filtered_signal=overlay_show_filtered_signal
            )
            fft_plot_html = None
            if 'fft' in set(selected_graphs):
                fft_plot_html = create_signal_fft_plot(
                    dfData_all,
                    exp_path=exp_path,
                    gain=gain,
                    signal_mode=signal_mode,
                    notch_params=final_notch_params
                )
            mean_power = None
            if gain is not None and req is not None and not is_oc_sc:
                mean_power = calculate_mean_power(dfData_all, exp_path, gain, req, peak_params=final_peak_params)
            conversion_factors, conversion_factors_error = get_keithley_conversion_factors(exp_path)

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
                is_sc=is_sc,
                fft_plot=fft_plot_html,
                cycles_overlay=cycles_overlay_html,
                overlay_cycle_start=overlay_cycle_start,
                overlay_cycle_end=overlay_cycle_end,
                overlay_cycles_max=overlay_cycles_max,
                overlay_show_raw_signal=overlay_show_raw_signal,
                overlay_show_filtered_signal=overlay_show_filtered_signal,
                keithley_conversion_factors=conversion_factors,
                keithley_conversion_factors_error=conversion_factors_error
            )
        except Exception as error:
            import traceback
            print("ERROR DETECTED IN VIEW_FILE:")
            print(traceback.format_exc())
            flash(f'Failed to process file: {error}')
            return redirect(url_for('list_files'))
