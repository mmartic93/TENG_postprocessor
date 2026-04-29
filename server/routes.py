import os
from flask import render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename

from server.config import UPLOAD_FOLDER, MAX_PREVIEW_ROWS
from data_processing.file_resolver import resolve_relative_path, file_exists, normalize_display_path
from data_processing.metadata_loader import (
    allowed_meta,
    parse_metadata_csv,
    format_metadata_rows,
    validate_metadata_columns,
    get_required_columns,
    get_rows_for_tribuid,
    get_paired_files,
    find_loads_description_file,
    load_loads_description,
    lookup_load_info,
)
from data_processing.preview_service import (
    csv_to_dataframe,
    tdms_to_dataframe,
    create_plot_html,
    calculate_mean_power_from_file,
    calculate_peak_power_from_file,
    create_mean_power_vs_req_plot,
    has_tdms_support,
    has_plotly_support,
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
            file_pairs = get_paired_files(sample)
        except Exception as error:
            flash(f'Unable to read selected TribuId rows: {error}')
            return redirect(url_for('metadata_preview'))

        file_entries = []
        for pair in file_pairs:
            entry = {'exp_id': pair['exp_id'], 'rload_id': pair.get('rload_id', ''),
                     'tribu_id': pair.get('tribu_id', '')}
            if loads_info_df is not None:
                load_info = lookup_load_info(loads_info_df, pair.get('rload_id', ''))
                entry['req'] = load_info['Req']
                entry['gain'] = load_info['Gain']
                entry['rload_missing'] = load_info['missing']
            else:
                entry['req'] = ''
                entry['gain'] = ''
                entry['rload_missing'] = False
            if loads_description_error:
                entry['loads_description_error'] = loads_description_error

            # Process DAQ file
            if pair['daq']:
                try:
                    daq_path = resolve_relative_path(meta_dir, pair['daq'])
                    if not file_exists(daq_path):
                        entry['daq_error'] = 'File not found'
                    else:
                        _, ext = os.path.splitext(daq_path)
                        entry['daq_rel'] = normalize_display_path(pair['daq'])
                        entry['daq_ext'] = ext.lower()
                        entry['daq_abs'] = daq_path

                        if entry.get('req') and entry.get('gain'):
                            try:
                                # Calculate Mean Power
                                entry['mean_power'] = calculate_mean_power_from_file(
                                    daq_path,
                                    entry['daq_ext'],
                                    float(entry['gain']),
                                    float(entry['req']),
                                )
                                # Calculate Peak Power
                                entry['peak_power'] = calculate_peak_power_from_file(
                                    daq_path,
                                    entry['daq_ext'],
                                    float(entry['gain']),
                                    float(entry['req']),
                                )
                                # Calculate VPP
                                from data_processing.preview_service import calculate_mean_vpp_from_file
                                entry['mean_vpp'] = calculate_mean_vpp_from_file(
                                    daq_path,
                                    entry['daq_ext'],
                                    float(entry['gain'])
                                )
                            except Exception:
                                entry['mean_power'] = None
                                entry['peak_power'] = None
                                entry['mean_vpp'] = None
                except Exception as error:
                    entry['daq_error'] = str(error)

            # Process Motor file
            if pair['motor']:
                try:
                    motor_path = resolve_relative_path(meta_dir, pair['motor'])
                    if not file_exists(motor_path):
                        entry['motor_error'] = 'File not found'
                    else:
                        _, ext = os.path.splitext(motor_path)
                        entry['motor_rel'] = normalize_display_path(pair['motor'])
                        entry['motor_ext'] = ext.lower()
                        entry['motor_abs'] = motor_path
                except Exception as error:
                    entry['motor_error'] = str(error)

            file_entries.append(entry)

        from collections import defaultdict

        grouped_power_data = defaultdict(list)
        grouped_peak_power_data = defaultdict(list)
        grouped_vpp_data = defaultdict(list)

        for entry in file_entries:
            req = entry.get('req')
            t_id = entry.get('tribu_id', 'Unknown')
            if req:
                req_val = float(req)
                if entry.get('mean_power') is not None:
                    grouped_power_data[t_id].append((req_val, entry['mean_power']))
                if entry.get('peak_power') is not None:
                    grouped_peak_power_data[t_id].append((req_val, entry['peak_power']))
                if entry.get('mean_vpp') is not None:
                    grouped_vpp_data[t_id].append((req_val, entry['mean_vpp']))

        # Calculate Optimal Points
        optimal_points = []
        for t_id, points in grouped_power_data.items():
            if points:
                # Find the tuple (req, power) with the highest power
                best_point = max(points, key=lambda x: x[1])
                optimal_points.append({
                    'tribu_id': t_id,
                    'req': best_point[0],
                    'max_power': best_point[1]
                })

        mean_power_plot = None
        optimal_power_plot = None
        mean_vpp_plot = None

        try:
            if grouped_power_data:
                # Pass both mean power and peak power to the plot
                mean_power_plot = create_mean_power_vs_req_plot(
                    grouped_power_data,
                    grouped_peak_power_data,
                    f'Power Analysis vs Resistance ({selected_tribuid})'
                )
                from data_processing.preview_service import create_optimal_power_plot
                optimal_power_plot = create_optimal_power_plot(
                    optimal_points,
                    "Optimal Power Comparison across TribuIds"
                )
            if grouped_vpp_data:
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
            mean_power_plot=mean_power_plot,
            optimal_power_plot=optimal_power_plot,
            mean_vpp_plot=mean_vpp_plot,
        )

    @app.route('/view')
    def view_file():
        metadata_path = session.get('metadata_path')
        rel = request.args.get('rel')  # This is the primary file clicked (Motor)
        daq_rel = request.args.get('daq_rel')  # Passed from the template for combined view
        downsample_percent = int(request.args.get('downsample', 80))

        if not metadata_path or not rel:
            flash('Missing parameters')
            return redirect(url_for('index'))

        meta_dir = os.path.dirname(metadata_path)

        try:
            target = resolve_relative_path(meta_dir, rel)
            if not file_exists(target):
                flash(f'File not found: {target}')
                return redirect(url_for('list_files'))

            _, ext = os.path.splitext(target)
            ext = ext.lower()

            gain_value = request.args.get('gain', '').strip()
            gain = float(gain_value) if gain_value else None
            req_value = request.args.get('req', '').strip()
            req = float(req_value) if req_value else None

            plot_mode = request.args.get('plot_mode', 'voltage')

            if ext == '.csv':
                df = csv_to_dataframe(target)
            elif ext == '.tdms':
                if not has_tdms_support():
                    flash('nptdms library not installed')
                    return redirect(url_for('list_files'))
                df = tdms_to_dataframe(target)
            else:
                flash(f'Unsupported file type: {ext}')
                return redirect(url_for('list_files'))

            if daq_rel:
                try:
                    daq_abs = resolve_relative_path(meta_dir, daq_rel)
                    _, daq_ext = os.path.splitext(daq_abs)
                    daq_ext = daq_ext.lower()

                    if daq_ext == '.csv':
                        daq_df = csv_to_dataframe(daq_abs)
                    elif daq_ext == '.tdms':
                        daq_df = tdms_to_dataframe(daq_abs)
                    else:
                        raise ValueError(f"Unsupported DAQ extension: {daq_ext}")

                    from data_processing.preview_service import create_combined_motor_daq_plot
                    plot_html = create_combined_motor_daq_plot(
                        daq_df=daq_df,
                        motor_df=df,
                        title=f"Combined Analysis: {rel}",
                        downsample_percent=downsample_percent,
                        gain=gain
                    )
                except Exception as e:
                    flash(f"Could not load associated voltage file: {e}")
                    plot_html = create_plot_html(df, f"Motor Data: {rel}", downsample_percent)
            else:
                plot_html = create_plot_html(df, f"{ext.upper()} : {rel}", downsample_percent, gain=gain,
                                             plot_mode=plot_mode, req=req)

            mean_power = None
            if plot_mode == 'power' and gain is not None and req is not None:
                mean_power = calculate_mean_power_from_file(target, ext, gain, req)

            df_info = f'{len(df)} rows × {len(df.columns)} columns'
            return render_template(
                'plot_view.html',
                plot=plot_html,
                filename=rel,
                df_info=df_info,
                downsample_percent=downsample_percent,
                gain_display=gain_value,
                plot_mode=plot_mode,
                req_display=req_value,
                mean_power=mean_power,
                daq_rel=daq_rel
            )

        except Exception as error:
            flash(f'Failed to process file: {error}')
            return redirect(url_for('list_files'))