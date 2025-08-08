import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import base64
import io
import json
import re
import logging
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Dash app with Tailwind CSS
app = dash.Dash(__name__, external_stylesheets=[
    "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"
])

app.title = "PBS DataClean Dashboard"

# Custom CSS for additional styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            body {
                font-family: 'Inter', sans-serif;
                background-color: #f9fafb;
                color: #1f2937;
                margin: 0;
                padding: 0;
            }
            
            .dashboard-header {
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                color: white;
                padding: 2rem;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            
            .logo-container {
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            
            .app-title {
                font-size: 2rem;
                font-weight: 600;
                margin: 0;
            }
            
            .app-tagline {
                font-size: 1rem;
                font-weight: 300;
                opacity: 0.9;
                margin: 0;
            }
            
            .dashboard-card {
                background: white;
                border-radius: 0.5rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 1.5rem;
                padding: 1.5rem;
                border-left: 4px solid #3b82f6;
                transition: transform 0.2s ease;
            }
            
            .dashboard-card:hover {
                transform: translateY(-2px);
            }
            
            .card-header {
                font-size: 1.25rem;
                font-weight: 500;
                color: #1f2937;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .kpi-card {
                background: #f3f4f6;
                border-radius: 0.5rem;
                padding: 1rem;
                text-align: center;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
            }
            
            .kpi-value {
                font-size: 1.75rem;
                font-weight: 600;
                color: #3b82f6;
                margin: 0;
            }
            
            .kpi-label {
                font-size: 0.875rem;
                color: #6b7280;
                margin: 0;
            }
            
            .btn-primary {
                background-color: #3b82f6;
                border: none;
                border-radius: 0.375rem;
                padding: 0.5rem 1.5rem;
                font-weight: 500;
                color: white;
                transition: background-color 0.2s ease;
            }
            
            .btn-primary:hover {
                background-color: #2563eb;
            }
            
            .btn-secondary {
                background-color: #6b7280;
                border: none;
                border-radius: 0.375rem;
                padding: 0.5rem 1.5rem;
                font-weight: 500;
                color: white;
                transition: background-color 0.2s ease;
            }
            
            .btn-secondary:hover {
                background-color: #4b5563;
            }
            
            .legend-item {
                display: inline-flex;
                align-items: center;
                margin-right: 1rem;
                margin-bottom: 0.5rem;
            }
            
            .legend-color {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                margin-right: 0.5rem;
            }
            
            .null-highlight {
                background-color: #fee2e2 !important;
            }
            
            .outlier-highlight {
                background-color: #ffedd5 !important;
            }
            
            .form-control, .form-select {
                border-radius: 0.375rem;
                border: 1px solid #d1d5db;
                background-color: white;
                color: #1f2937;
                padding: 0.5rem;
            }
            
            .form-control:focus, .form-select:focus {
                border-color: #3b82f6;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
                outline: none;
            }
            
            .rule-error {
                color: #dc2626;
                font-size: 0.875rem;
                margin-top: 0.5rem;
                display: none;
            }
            
            @media (max-width: 768px) {
                .app-title {
                    font-size: 1.5rem;
                }
                
                .dashboard-card {
                    padding: 1rem;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Helper functions (unchanged)
def parse_contents(contents, filename):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename.lower():
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
                    break
                except UnicodeDecodeError:
                    continue
                if df is None:
                    return None, "Error: Could not decode CSV file with any supported encoding", "Encoding error"
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, "Error: Unsupported file type. Please upload CSV or Excel files.", "Invalid file type"
        if df.empty:
            return None, "Error: The uploaded file is empty", "Empty file"
        logger.info(f"Successfully parsed file: {filename}, Shape: {df.shape}")
        return df, f"Success: Loaded {len(df)} rows and {len(df.columns)} columns", "File loaded successfully"
    except Exception as e:
        logger.error(f"Error processing file upload: {str(e)}")
        return None, f"Error: Could not process file upload - {str(e)}", "Upload error"

def parse_if_then_rule(rule_text, df_columns):
    if not rule_text or not rule_text.strip():
        return None, "", False, ""
    pattern = r"If\s+(.+?)\s+then\s+(.+)"
    match = re.match(pattern, rule_text.strip(), re.IGNORECASE)
    if not match:
        return None, "Invalid rule syntax. Use: If [column] [operator] [value] then [column] as [value] (e.g., If Age > 60 then Status as Senior)", False, ""
    condition_str, target_str = match.groups()
    if re.search(r'\s+or\s+', condition_str, re.IGNORECASE):
        logic = 'or'
        condition_parts = re.split(r'\s+or\s+', condition_str, flags=re.IGNORECASE)
    else:
        logic = 'and'
        condition_parts = re.split(r'\s+and\s+', condition_str, flags=re.IGNORECASE)
    conditions = []
    error_msg = ""
    missing_condition_cols = []
    for cond in condition_parts:
        cond_match = re.match(r'(\w+)\s+(>|>=|<|<=|==|!=|is)\s+([^\s].*?)(?=\s+then|\s+and|\s+or|$)', cond.strip(), re.IGNORECASE)
        if not cond_match:
            return None, f"Invalid condition syntax: {cond}. Use operators like >, <, ==, etc.", False, ""
        col, operator, value = cond_match.groups()
        if col not in df_columns:
            error_msg += f"Condition column '{col}' not found in dataset. <br>"
            missing_condition_cols.append(col)
        conditions.append({
            'column': col,
            'operator': operator.lower(),
            'value': value.strip()
        })
    targets = []
    target_parts = re.split(r'\s+and\s+', target_str, flags=re.IGNORECASE)
    for target in target_parts:
        target_match = re.match(r'(\w+)\s+as\s+(.+)', target.strip(), re.IGNORECASE)
        if not target_match:
            return None, f"Invalid target syntax: {target}. Use 'as' keyword (e.g., Status as Senior)", False, ""
        col, value = target_match.groups()
        targets.append({
            'column': col,
            'value': value.strip()
        })
    if not conditions or not targets:
        return None, "At least one condition and one target required", False, ""
    rule_dict = {
        'conditions': conditions,
        'targets': targets,
        'logic': logic
    }
    if missing_condition_cols:
        return None, error_msg, False, error_msg
    return rule_dict, "Rule parsed successfully", True, error_msg

def apply_if_then_rule(df, rule_dict):
    try:
        df_copy = df.copy()
        logic = rule_dict.get('logic', 'and')
        masks = []
        for cond in rule_dict['conditions']:
            col = cond['column']
            operator = cond['operator']
            value = cond['value']
            try:
                value_num = float(value)
                is_numeric = True
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            except ValueError:
                is_numeric = False
            if operator == 'is' or (operator == '==' and not is_numeric):
                condition = df_copy[col].astype(str) == value
            elif is_numeric:
                if operator == '>':
                    condition = df_copy[col] > value_num
                elif operator == '>=':
                    condition = df_copy[col] >= value_num
                elif operator == '<':
                    condition = df_copy[col] < value_num
                elif operator == '<=':
                    condition = df_copy[col] <= value_num
                elif operator == '==':
                    condition = df_copy[col] == value_num
                elif operator == '!=':
                    condition = df_copy[col] != value_num
                else:
                    return df, 0, f"Unsupported operator: {operator}"
            else:
                return df, 0, f"Cannot apply numeric operator {operator} to non-numeric value"
            masks.append(condition)
        if not masks:
            return df, 0, "No valid conditions"
        if logic == 'and':
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m
        else:
            mask = masks[0]
            for m in masks[1:]:
                mask = mask | m
        rows_modified = mask.sum()
        for target in rule_dict['targets']:
            col = target['column']
            value = target['value']
            if col not in df_copy.columns:
                df_copy[col] = np.nan
            df_copy.loc[mask, col] = value
        logger.info(f"Applied If-Then rule: {rows_modified} rows modified")
        return df_copy, rows_modified, f"Rule applied successfully: {rows_modified} rows modified"
    except Exception as e:
        logger.error(f"Error applying If-Then rule: {str(e)}")
        return df, 0, f"Error applying rule: {str(e)}"

def clean_data(df, numerical_cols, categorical_cols, null_method, outlier_method, outlier_threshold, if_then_rule=None, outlier_handling='remove'):
    try:
        from sklearn.impute import KNNImputer
    except ImportError:
        KNNImputer = None
    df_cleaned = df.copy()
    nulls_handled = 0
    outliers_handled = 0
    rule_status = "No rule applied"
    if_then_rule = if_then_rule or None
    outlier_columns = numerical_cols
    if if_then_rule:
        df_cleaned, rows_modified, rule_status = apply_if_then_rule(df_cleaned, if_then_rule)
    if null_method != 'none':
        initial_nulls = df_cleaned[numerical_cols + categorical_cols].isnull().sum().sum() if (numerical_cols or categorical_cols) else df_cleaned.isnull().sum().sum()
        if null_method == 'remove_rows':
            selected_cols = numerical_cols + categorical_cols
            if selected_cols:
                df_cleaned = df_cleaned.dropna(subset=selected_cols)
                nulls_handled = initial_nulls - df_cleaned[numerical_cols + categorical_cols].isnull().sum().sum()
        elif null_method in ['fill_mean', 'fill_median', 'fill_mode', 'interpolate', 'knn']:
            for col in numerical_cols:
                if col in df_cleaned.columns:
                    nulls_in_col = df_cleaned[col].isnull().sum()
                    if null_method == 'fill_mean':
                        fill_value = df_cleaned[col].mean()
                        df_cleaned[col].fillna(fill_value, inplace=True)
                    elif null_method == 'fill_median':
                        fill_value = df_cleaned[col].median()
                        df_cleaned[col].fillna(fill_value, inplace=True)
                    elif null_method == 'fill_mode':
                        fill_value = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 0
                        df_cleaned[col].fillna(fill_value, inplace=True)
                    elif null_method == 'interpolate':
                        interpolated = df_cleaned[col].interpolate(method='linear', limit_direction='both')
                        df_cleaned[col] = interpolated
                    elif null_method == 'knn' and KNNImputer is not None:
                        temp = df_cleaned[numerical_cols].copy()
                        imputer = KNNImputer(n_neighbors=3)
                        imputed = imputer.fit_transform(temp)
                        df_cleaned[numerical_cols] = imputed
                        break
                    nulls_handled += nulls_in_col
            if null_method == 'fill_mode':
                for col in categorical_cols:
                    if col in df_cleaned.columns:
                        fill_value = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 'Unknown'
                        nulls_in_col = df_cleaned[col].isnull().sum()
                        df_cleaned[col].fillna(fill_value, inplace=True)
                        nulls_handled += nulls_in_col
    if outlier_method != 'none' and outlier_columns:
        X = df_cleaned[outlier_columns].select_dtypes(include=[np.number]).dropna()
        if not X.empty:
            outlier_idx = set()
            if outlier_method == 'zscore':
                z = np.abs(stats.zscore(X))
                outlier_mask = (z > (outlier_threshold or 3)).any(axis=1)
                outlier_idx = set(X.index[outlier_mask])
            elif outlier_method == 'iqr':
                outlier_mask = np.zeros(len(X), dtype=bool)
                for col in X.columns:
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - (outlier_threshold or 1.5) * IQR
                    upper = Q3 + (outlier_threshold or 1.5) * IQR
                    outlier_mask = outlier_mask | ((X[col] < lower) | (X[col] > upper))
                outlier_idx = set(X.index[outlier_mask])
            elif outlier_method == 'isoforest':
                clf = IsolationForest(contamination='auto', random_state=42)
                preds = clf.fit_predict(X)
                outlier_idx = set(X.index[preds == -1])
            elif outlier_method == 'lof':
                n_samples = X.shape[0]
                n_neighbors = min(10, max(2, n_samples - 1))
                if n_samples < 3:
                    outlier_idx = set()
                else:
                    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
                    preds = clf.fit_predict(X)
                    outlier_idx = set(X.index[preds == -1])
            elif outlier_method == 'dbscan':
                n_samples = X.shape[0]
                min_samples = min(5, n_samples) if n_samples >= 2 else 2
                if n_samples < min_samples:
                    outlier_idx = set()
                else:
                    clf = DBSCAN(eps=outlier_threshold or 0.5, min_samples=min_samples)
                    preds = clf.fit_predict(X)
                    outlier_idx = set(X.index[preds == -1])
            if outlier_handling == 'remove':
                initial_count = len(df_cleaned)
                df_cleaned = df_cleaned.drop(index=outlier_idx)
                outliers_handled = initial_count - len(df_cleaned)
            elif outlier_handling in ['fill_mean', 'fill_median', 'fill_mode', 'interpolate', 'knn']:
                outliers_handled = len(outlier_idx)
                for col in outlier_columns:
                    if col in df_cleaned.columns:
                        if outlier_handling == 'fill_mean':
                            fill_value = df_cleaned[col].mean()
                            df_cleaned.loc[outlier_idx, col] = fill_value
                        elif outlier_handling == 'fill_median':
                            fill_value = df_cleaned[col].median()
                            df_cleaned.loc[outlier_idx, col] = fill_value
                        elif outlier_handling == 'fill_mode':
                            fill_value = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 0
                            df_cleaned.loc[outlier_idx, col] = fill_value
                        elif outlier_handling == 'interpolate':
                            interpolated = df_cleaned[col].interpolate(method='linear', limit_direction='both')
                            df_cleaned.loc[outlier_idx, col] = interpolated.loc[outlier_idx]
                        elif outlier_handling == 'knn' and KNNImputer is not None:
                            temp = df_cleaned[[col]].copy()
                            temp.loc[outlier_idx, col] = np.nan
                            imputer = KNNImputer(n_neighbors=3)
                            imputed = imputer.fit_transform(temp)
                            for idx in outlier_idx:
                                df_cleaned.at[idx, col] = imputed[list(temp.index).index(idx), 0]
            elif outlier_handling == 'keep':
                outliers_handled = len(outlier_idx)
            else:
                outliers_handled = len(outlier_idx)
    logger.info(f"Data cleaning completed: {nulls_handled} NULLs handled, {outliers_handled} outliers handled")
    return df_cleaned, nulls_handled, outliers_handled, rule_status

def create_data_table(df, table_id, max_rows=1000):
    if df is None or df.empty:
        return html.Div("No data to display", className="text-gray-500 text-center py-4")
    display_df = df.copy()
    columns = []
    for col in display_df.columns:
        max_length = max(
            len(str(col)),
            display_df[col].astype(str).str.len().max() if not display_df[col].empty else 0
        )
        width = min(max(max_length * 8, 100), 200)
        columns.append({
            "name": col,
            "id": col,
            "type": "numeric" if pd.api.types.is_numeric_dtype(display_df[col]) else "text",
            "format": {"specifier": ".2f"} if pd.api.types.is_numeric_dtype(display_df[col]) else None
        })
    style_data_conditional = [
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9fafb'}
    ]
    for col in display_df.columns:
        null_rows = display_df[display_df[col].isnull()].index.tolist()
        for row in null_rows:
            style_data_conditional.append({
                'if': {'row_index': row, 'column_id': col},
                'backgroundColor': '#fee2e2',
                'color': '#1f2937'
            })
    return dash_table.DataTable(
        id=table_id,
        data=display_df.to_dict('records'),
        columns=columns,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '0.75rem',
            'fontFamily': 'Inter, sans-serif',
            'fontSize': '0.875rem',
            'border': '1px solid #e5e7eb'
        },
        style_header={
            'backgroundColor': '#3b82f6',
            'color': 'white',
            'fontWeight': '500',
            'textAlign': 'center'
        },
        style_data_conditional=style_data_conditional,
        sort_action="native",
        filter_action="native",
        page_action="native",
        page_current=0,
        page_size=20,
        export_format="csv"
    )

def create_data_table_with_outlier_highlight(df, table_id, outlier_indices, outlier_cols, highlight_cols, max_rows=1000):
    if df is None or df.empty:
        return html.Div("No data to display", className="text-gray-500 text-center py-4")
    display_df = df.copy()
    columns = []
    for col in display_df.columns:
        max_length = max(
            len(str(col)),
            display_df[col].astype(str).str.len().max() if not display_df[col].empty else 0
        )
        width = min(max(max_length * 8, 100), 200)
        columns.append({
            "name": col,
            "id": col,
            "type": "numeric" if pd.api.types.is_numeric_dtype(display_df[col]) else "text",
            "format": {"specifier": ".2f"} if pd.api.types.is_numeric_dtype(display_df[col]) else None
        })
    style_data_conditional = [
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9fafb'}
    ]
    for col in display_df.columns:
        null_rows = display_df[display_df[col].isnull()].index.tolist()
        for row in null_rows:
            style_data_conditional.append({
                'if': {'row_index': row, 'column_id': col},
                'backgroundColor': '#fee2e2',
                'color': '#1f2937'
            })
    if outlier_indices and outlier_cols:
        for col in outlier_cols:
            if col in display_df.columns:
                for row in display_df.index:
                    if row in outlier_indices:
                        style_data_conditional.append({
                            'if': {'row_index': row, 'column_id': col},
                            'backgroundColor': '#ffedd5',
                            'color': '#1f2937'
                        })
    return dash_table.DataTable(
        id=table_id,
        data=display_df.to_dict('records'),
        columns=columns,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '0.75rem',
            'fontFamily': 'Inter, sans-serif',
            'fontSize': '0.875rem',
            'border': '1px solid #e5e7eb'
        },
        style_header={
            'backgroundColor': '#3b82f6',
            'color': 'white',
            'fontWeight': '500',
            'textAlign': 'center'
        },
        style_data_conditional=style_data_conditional,
        sort_action="native",
        filter_action="native",
        page_action="native",
        page_current=0,
        page_size=20,
        export_format="csv"
    )

def create_upload_section():
    return html.Div([
        html.Div([
            html.I(className="fas fa-cloud-upload-alt mr-2"),
            "Upload Data"
        ], className="card-header"),
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.I(className="fas fa-cloud-upload-alt text-4xl mb-4 text-blue-500"),
                    html.H4("Drag and Drop or Click to Upload", className="text-lg font-medium text-gray-900"),
                    html.P("Supported formats: CSV, Excel (.xlsx, .xls)", className="text-sm text-gray-500")
                ], className="text-center"),
                className="dashboard-card border-dashed border-2 border-gray-300 p-6 hover:border-blue-500 transition-colors"),
            html.Div(id='upload-success-message', className="mt-4"),
            html.Div(id='upload-status', className="mt-4"),
            dbc.Modal([
                dbc.ModalBody([
                    html.Div([
                        html.I(className="fas fa-spinner fa-spin text-3xl text-blue-500"),
                        html.H5("Uploading...", className="mt-4 text-blue-500")
                    ], className="text-center")
                ])
            ], id="uploading-modal", is_open=False, backdrop=True, centered=True),
            dcc.Loading(id="loading-upload", type="default", children=html.Div(id="upload-progress"))
        ])
    ], className="dashboard-card")

def create_config_section():
    return html.Div([
        html.Div([
            html.I(className="fas fa-cog mr-2"),
            "Configuration"
        ], className="card-header"),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Numerical Columns", className="block text-sm font-medium text-gray-700 mb-2"),
                    dcc.Dropdown(
                        id='numerical-columns',
                        placeholder="Select numerical columns...",
                        multi=True,
                        className="form-control"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Categorical Columns", className="block text-sm font-medium text-gray-700 mb-2"),
                    dcc.Dropdown(
                        id='categorical-columns',
                        placeholder="Select categorical columns...",
                        multi=True,
                        className="form-control"
                    )
                ], width=6)
            ], className="mb-4"),
            html.Div(id='upload-status', className="mt-4 mb-4"),
            dbc.Row([
                dbc.Col([
                    html.Label("If-Then Rule", className="block text-sm font-medium text-gray-700 mb-2"),
                    dbc.Input(
                        id='if-then-rule',
                        placeholder="e.g., If Age > 60 then Status as Senior",
                        type="text",
                        className="form-control"
                    ),
                    html.Small(
                        "Syntax: Use 'If [column] [operator] [value] then [column] as [value]' (e.g., If Age > 60 then Status as Senior). Operators: >, <, >=, <=, ==, !=, is.",
                        className="text-sm text-gray-500"
                    ),
                    html.Div(id='rule-error', className="rule-error")
                ], width=12)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    html.Label("NULL Handling", className="block text-sm font-medium text-gray-700 mb-2"),
                    dcc.Dropdown(
                        id='null-method',
                        options=[
                            {'label': 'None', 'value': 'none'},
                            {'label': 'Remove rows with NULLs', 'value': 'remove_rows'},
                            {'label': 'Fill with mean (numerical)', 'value': 'fill_mean'},
                            {'label': 'Fill with median (numerical)', 'value': 'fill_median'},
                            {'label': 'Fill with mode (all columns)', 'value': 'fill_mode'},
                            {'label': 'Fill with linear interpolation (numerical)', 'value': 'interpolate'},
                            {'label': 'Fill with KNN imputer (numerical)', 'value': 'knn'}
                        ],
                        value='none',
                        className="form-control"
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Outlier Detection", className="block text-sm font-medium text-gray-700 mb-2"),
                    dcc.Dropdown(
                        id='outlier-method',
                        options=[
                            {'label': 'None', 'value': 'none'},
                            {'label': 'Z-Score', 'value': 'zscore'},
                            {'label': 'IQR (Interquartile Range)', 'value': 'iqr'},
                            {'label': 'Isolation Forest', 'value': 'isoforest'},
                            {'label': 'Local Outlier Factor', 'value': 'lof'},
                            {'label': 'DBSCAN', 'value': 'dbscan'}
                        ],
                        value='none',
                        className="form-control"
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Outlier Handling", className="block text-sm font-medium text-gray-700 mb-2"),
                    dcc.Dropdown(
                        id='outlier-handling',
                        options=[
                            {'label': 'Keep outliers', 'value': 'keep'},
                            {'label': 'Remove outliers', 'value': 'remove'},
                            {'label': 'Fill with mean', 'value': 'fill_mean'},
                            {'label': 'Fill with median', 'value': 'fill_median'},
                            {'label': 'Fill with mode', 'value': 'fill_mode'},
                            {'label': 'Fill with linear interpolation', 'value': 'interpolate'},
                            {'label': 'Fill with KNN imputer', 'value': 'knn'}
                        ],
                        value='remove',
                        clearable=False,
                        className="form-control"
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Outlier Threshold", className="block text-sm font-medium text-gray-700 mb-2"),
                    dbc.Input(
                        id='outlier-threshold',
                        type="number",
                        value=3,
                        step=0.1,
                        min=0.1,
                        className="form-control"
                    )
                ], width=3)
            ])
        ], className="card-content")
    ], className="dashboard-card")

def create_preview_section():
    return html.Div([
        html.Div([
            html.I(className="fas fa-eye mr-2"),
            "Data Preview"
        ], className="card-header"),
        html.Div([
            dbc.Tabs([
                dbc.Tab(label="Data Table", tab_id="table-tab", tabClassName="px-4 py-2 text-sm font-medium text-gray-700"),
                dbc.Tab(label="NULL Distribution", tab_id="chart-tab", tabClassName="px-4 py-2 text-sm font-medium text-gray-700")
            ], id="preview-tabs", active_tab="table-tab", className="border-b border-gray-200"),
            html.Div(id='data-preview-content', className="mt-4")
        ], className="card-content")
    ], className="dashboard-card")

def create_stats_section():
    return html.Div([
        html.Div([
            html.I(className="fas fa-chart-pie mr-2"),
            "Statistics"
        ], className="card-header"),
        html.Div([
            html.H6("Dataset Overview", className="text-base font-medium text-gray-900 mb-2"),
            html.Div(id='data-overview-table'),
            html.H6("Selected Columns Statistics", className="mt-4 mb-2 text-base font-medium text-gray-900"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("0", className="kpi-value", id="null-count-selected"),
                        html.P("NULL Count", className="kpi-label")
                    ], className="kpi-card")
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H3("0", className="kpi-value", id="outlier-count-selected"),
                        html.P("Outlier Count", className="kpi-label")
                    ], className="kpi-card")
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H3("0", className="kpi-value", id="nulls-handled"),
                        html.P("NULLs Handled", className="kpi-label")
                    ], className="kpi-card")
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H3("0", className="kpi-value", id="outliers-removed"),
                        html.P("Outliers Removed", className="kpi-label")
                    ], className="kpi-card")
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H3("0", className="kpi-value", id="final-rows"),
                        html.P("Final Rows", className="kpi-label")
                    ], className="kpi-card")
                ], width=2)
            ])
        ], className="card-content")
    ], className="dashboard-card")

def create_action_section():
    return html.Div([
        html.Div([
            html.I(className="fas fa-tools mr-2"),
            "Actions"
        ], className="card-header"),
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-magic mr-2"), "Clean Data"],
                        id="clean-btn",
                        color="primary",
                        size="lg",
                        disabled=True,
                        className="btn-primary w-full"
                    )
                ], width=4),
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-redo mr-2"), "Reset Data"],
                        id="reset-btn",
                        color="warning",
                        size="lg",
                        disabled=True,
                        className="btn-secondary w-full"
                    )
                ], width=4),
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-download mr-2"), "Download Cleaned Data"],
                        id="download-btn",
                        color="success",
                        size="lg",
                        disabled=True,
                        className="btn-primary w-full bg-green-500 hover:bg-green-600"
                    )
                ], width=4)
            ])
        ], className="card-content")
    ], className="dashboard-card")

def create_results_section():
    return html.Div([
        html.Div([
            html.I(className="fas fa-check-circle mr-2"),
            "Cleaned Data"
        ], className="card-header"),
        html.Div([
            html.Div(id='cleaning-summary', className="mb-4"),
            html.Div(id='cleaned-data-table')
        ], className="card-content")
    ], className="dashboard-card", id="results-section", style={'display': 'none'})

def create_header():
    return html.Div([
        html.Div([
            html.Img(src="/assets/PBS Transparent Logo.png", className="mr-4", style={"height": "48px", "width": "auto", "maxWidth": "200px", "objectFit": "contain"}),
            html.H1("PBS DataClean Dashboard", className="app-title"),
        ], className="logo-container"),
        html.P("Advanced Data Cleaning & Analysis Tool", className="app-tagline")
    ], className="dashboard-header")

def create_summary_plot(df, highlight_cols):
    figs = []
    null_counts = df[highlight_cols].isnull().sum() if highlight_cols else df.isnull().sum()
    fig_null = go.Figure([go.Bar(
        x=null_counts.index, y=null_counts.values,
        marker=dict(
            color=null_counts.values,
            colorscale='Viridis',
            line=dict(color='#1f2937', width=1)
        ),
        hoverinfo='x+y',
        text=null_counts.values,
        textposition='outside'
    )])
    fig_null.update_layout(
        title="NULL Distribution",
        xaxis_title="Columns",
        yaxis_title="NULL Count",
        template="plotly_white",
        plot_bgcolor='#f9fafb',
        paper_bgcolor='#f9fafb',
        font=dict(color='#1f2937', family='Inter'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    figs.append(fig_null)
    num_cols = [col for col in highlight_cols if pd.api.types.is_numeric_dtype(df[col])] if highlight_cols else df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        fig_hist = go.Figure([go.Histogram(
            x=df[col].dropna(),
            marker=dict(
                color='#3b82f6',
                line=dict(color='#fff', width=1)
            ),
            opacity=0.85
        )])
        fig_hist.update_layout(
            title=f"Histogram: {col}",
            xaxis_title=col,
            yaxis_title="Frequency",
            template="plotly_white",
            plot_bgcolor='#f9fafb',
            paper_bgcolor='#f9fafb',
            font=dict(color='#1f2937', family='Inter'),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        figs.append(fig_hist)
    cat_cols = [col for col in highlight_cols if not pd.api.types.is_numeric_dtype(df[col])] if highlight_cols else df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        value_counts = df[col].value_counts().nlargest(10)
        fig_bar = go.Figure([go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            marker=dict(
                color='#10b981',
                line=dict(color='#fff', width=1)
            ),
            opacity=0.85,
            text=value_counts.values,
            textposition='outside'
        )])
        fig_bar.update_layout(
            title=f"Bar Chart: {col}",
            xaxis_title=col,
            yaxis_title="Count",
            template="plotly_white",
            plot_bgcolor='#f9fafb',
            paper_bgcolor='#f9fafb',
            font=dict(color='#1f2937', family='Inter'),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        figs.append(fig_bar)
    if len(figs) == 1:
        return figs[0]
    else:
        from plotly.subplots import make_subplots
        subplot_titles = [f.layout.title.text for f in figs]
        fig = make_subplots(rows=len(figs), cols=1, subplot_titles=subplot_titles, vertical_spacing=0.15)
        for i, f in enumerate(figs):
            for trace in f.data:
                fig.add_trace(trace, row=i+1, col=1)
        fig.update_layout(
            height=350*len(figs),
            showlegend=False,
            template="plotly_white",
            plot_bgcolor='#f9fafb',
            paper_bgcolor='#f9fafb',
            font=dict(color='#1f2937', family='Inter'),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

def create_outlier_boxplot(df, outlier_indices, outlier_cols):
    if not outlier_cols or df.empty:
        return go.Figure()
    fig = go.Figure()
    colors = px.colors.sequential.Viridis
    for i, col in enumerate(outlier_cols):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            box_color = colors[i % len(colors)]
            outlier_points = df.loc[list(outlier_indices), col] if outlier_indices else pd.Series([], dtype=float)
            fig.add_trace(go.Box(
                y=df[col],
                name=col,
                boxpoints='outliers',
                marker=dict(color=box_color, outliercolor='#ef4444', line=dict(outliercolor='#ef4444', outlierwidth=2)),
                line=dict(color=box_color, width=2),
                fillcolor='rgba(59,130,246,0.15)',
                boxmean='sd',
                hoverinfo='y+name'
            ))
            if not outlier_points.empty:
                jitter = np.random.uniform(-0.2, 0.2, size=len(outlier_points))
                fig.add_trace(go.Scatter(
                    x=[i + 1 + j for j in jitter],
                    y=outlier_points,
                    mode='markers',
                    marker=dict(color='#ef4444', size=10, symbol='diamond'),
                    name=f"{col} Outliers",
                    showlegend=False,
                    hoverinfo='y+name'
                ))
    fig.update_layout(
        title="Outlier Boxplot Visualization",
        xaxis_title="Columns",
        yaxis_title="Value",
        template="plotly_white",
        plot_bgcolor='#f9fafb',
        paper_bgcolor='#f9fafb',
        font=dict(color='#1f2937', family='Inter'),
        boxmode='group',
        legend=dict(font=dict(color='#1f2937')),
        margin=dict(l=80, r=40, t=60, b=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridcolor='#e5e7eb')
    return fig

# Main layout
app.layout = html.Div([
    html.Div(id='theme-container', className="min-h-screen bg-gray-50", children=[
        create_header(),
        dbc.Container([
            create_upload_section(),
            create_config_section(),
            create_stats_section(),
            create_preview_section(),
            create_action_section(),
            create_results_section(),
            html.Div(id='raw-data-store', style={'display': 'none'}),
            html.Div(id='cleaned-data-store', style={'display': 'none'}),
            html.Div(id='alert-container', className="mt-4"),
            html.Div(id='upload-spinner', className='upload-spinner'),
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Confirm Reset", className="text-lg font-semibold")),
                dbc.ModalBody("Are you sure you want to reset all data and settings? This action cannot be undone.", className="text-gray-700"),
                dbc.ModalFooter([
                    dbc.Button("Cancel", id="reset-cancel", className="btn-secondary mr-2"),
                    dbc.Button("Reset", id="reset-confirm", className="btn-primary bg-red-500 hover:bg-red-600")
                ])
            ], id="reset-modal", is_open=False),
            dcc.Download(id="download-cleaned-data")
        ], fluid=True, className="py-8")
    ])
])

# Callbacks (unchanged)
@app.callback(
    Output('raw-data-store', 'children'),
    Output('upload-status', 'children'),
    Output('numerical-columns', 'options'),
    Output('categorical-columns', 'options'),
    Output('alert-container', 'children'),
    Output('upload-spinner', 'className'),
    Output('loading-upload', 'children'),
    Output('upload-success-message', 'children'),
    Output('uploading-modal', 'is_open'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    try:
        if contents is None:
            return '', '', [], [], '', 'upload-spinner', html.Div(id='upload-progress'), '', False
        uploading_modal_open = True
        df, status_msg, debug_msg = parse_contents(contents, filename)
        if df is None:
            alert = dbc.Alert(status_msg, color="danger", dismissable=True, className="mt-4")
            return '', '', [], [], alert, 'upload-spinner', html.Div(id='upload-progress'), '', False
        data_json = df.to_json(date_format='iso', orient='split')
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        num_options = [{'label': col, 'value': col} for col in numerical_cols]
        cat_options = [{'label': col, 'value': col} for col in categorical_cols]
        alert = dbc.Alert(status_msg, color="success", dismissable=True, className="mt-4")
        success_message = html.Div([
            html.I(className="fas fa-check-circle mr-2 text-green-500 text-xl"),
            html.Span("Successfully uploaded!", className="text-green-500 font-semibold")
        ], className="bg-green-50 p-4 rounded-lg text-center")
        uploading_modal_open = False
        return data_json, status_msg, num_options, cat_options, alert, 'upload-spinner show', html.Div(id='upload-progress'), success_message, uploading_modal_open
    except Exception as e:
        logger.error(f"Error in update_output: {str(e)}")
        alert = dbc.Alert(f"Unexpected error: {str(e)}", color="danger", dismissable=True, className="mt-4")
        return '', '', [], [], alert, 'upload-spinner', html.Div(id='upload-progress'), '', False

@app.callback(
    Output('data-preview-content', 'children'),
    Output('data-overview-table', 'children'),
    Output('null-count-selected', 'children'),
    Output('outlier-count-selected', 'children'),
    Output('clean-btn', 'disabled'),
    Output('rule-error', 'children'),
    Output('rule-error', 'style'),
    [Input('raw-data-store', 'children'),
     Input('numerical-columns', 'value'),
     Input('categorical-columns', 'value'),
     Input('if-then-rule', 'value'),
     Input('null-method', 'value'),
     Input('outlier-method', 'value'),
     Input('outlier-handling', 'value'),
     Input('outlier-threshold', 'value'),
     Input('preview-tabs', 'active_tab')],
    prevent_initial_call=True
)
def update_data_preview_and_stats(data_json, num_cols, cat_cols, if_then_rule, null_method, 
                                outlier_method, outlier_handling, outlier_threshold, active_tab):
    if not data_json:
        return "No data uploaded", "", "0", "0", True, "", {'display': 'none'}
    try:
        df = pd.read_json(data_json, orient='split')
        selected_cols = (num_cols or []) + (cat_cols or [])
        selected_cols = [col for col in selected_cols if col in df.columns]
        outlier_cols = num_cols or []
        outlier_cols = [col for col in outlier_cols if col in df.columns]
        rule_dict = None
        rule_error = ""
        rule_style = {'display': 'none'}
        popup_alert = None
        if if_then_rule and if_then_rule.strip():
            rule_dict, rule_msg, is_valid, error_msg = parse_if_then_rule(if_then_rule, df.columns.tolist())
            if not is_valid:
                rule_error = error_msg or rule_msg
                rule_style = {'display': 'block'}
                popup_alert = dbc.Alert(rule_error, color="danger", dismissable=True, is_open=True, className="mt-4")
            else:
                rule_error = ""
                rule_style = {'display': 'none'}
        clean_btn_disabled = not (if_then_rule and if_then_rule.strip()) and not (num_cols or cat_cols)
        overview_data = [
            {"Metric": "Number of Rows", "Value": len(df)},
            {"Metric": "Number of Columns", "Value": len(df.columns)},
            {"Metric": "Column Names", "Value": ", ".join(df.columns.tolist())},
            {"Metric": "Data Types", "Value": ", ".join([str(dtype) for dtype in df.dtypes])}
        ]
        overview_table = dash_table.DataTable(
            data=overview_data,
            columns=[{"name": i, "id": i} for i in ["Metric", "Value"]],
            style_table={'overflowX': 'auto', 'marginTop': '1rem'},
            style_cell={
                'textAlign': 'left',
                'padding': '0.75rem',
                'fontFamily': 'Inter, sans-serif',
                'fontSize': '0.875rem',
                'backgroundColor': 'white',
                'color': '#1f2937',
                'border': '1px solid #e5e7eb'
            },
            style_header={
                'backgroundColor': '#3b82f6',
                'color': 'white',
                'fontWeight': '500',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9fafb'}
            ]
        )
        null_count_selected = "0"
        outlier_count_selected = "0"
        outlier_indices = set()
        highlight_cols = selected_cols if selected_cols else []
        if highlight_cols:
            null_count_selected = str(df[highlight_cols].isnull().sum().sum())
        if outlier_cols and outlier_method and outlier_method != 'none' and highlight_cols:
            try:
                X = df[[col for col in outlier_cols if col in highlight_cols]].select_dtypes(include=[np.number]).dropna()
                if not X.empty:
                    outlier_idx = set()
                    if outlier_method == 'zscore':
                        z = np.abs(stats.zscore(X))
                        outlier_mask = (z > (outlier_threshold or 3)).any(axis=1)
                        outlier_idx = set(X.index[outlier_mask])
                    elif outlier_method == 'iqr':
                        outlier_mask = np.zeros(len(X), dtype=bool)
                        for col in X.columns:
                            Q1 = X[col].quantile(0.25)
                            Q3 = X[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - (outlier_threshold or 1.5) * IQR
                            upper = Q3 + (outlier_threshold or 1.5) * IQR
                            outlier_mask = outlier_mask | ((X[col] < lower) | (X[col] > upper))
                        outlier_idx = set(X.index[outlier_mask])
                    elif outlier_method == 'isoforest':
                        clf = IsolationForest(contamination='auto', random_state=42)
                        preds = clf.fit_predict(X)
                        outlier_idx = set(X.index[preds == -1])
                    elif outlier_method == 'lof':
                        n_samples = X.shape[0]
                        n_neighbors = min(10, max(2, n_samples - 1))
                        if n_samples < 3:
                            outlier_idx = set()
                        else:
                            clf = LocalOutlierFactor(n_neighbors=n_neighbors)
                            preds = clf.fit_predict(X)
                            outlier_idx = set(X.index[preds == -1])
                    elif outlier_method == 'dbscan':
                        n_samples = X.shape[0]
                        min_samples = min(5, n_samples) if n_samples >= 2 else 2
                        if n_samples < min_samples:
                            outlier_idx = set()
                        else:
                            clf = DBSCAN(eps=outlier_threshold or 0.5, min_samples=min_samples)
                            preds = clf.fit_predict(X)
                            outlier_idx = set(X.index[preds == -1])
                    outlier_indices = outlier_idx
                    outlier_count_selected = str(len(outlier_indices))
            except Exception as e:
                logger.error(f"Error during outlier detection in preview: {str(e)}")
        if active_tab == "table-tab":
            preview_content = create_data_table_with_outlier_highlight(df, 'raw-data-table', outlier_indices, outlier_cols, highlight_cols)
        elif active_tab == "chart-tab":
            preview_content = dcc.Graph(figure=create_summary_plot(df, highlight_cols))
        else:
            preview_content = html.Div("No visualization available")
        if outlier_indices and outlier_cols:
            outlier_chart = dcc.Graph(figure=create_outlier_boxplot(df, outlier_indices, outlier_cols), className="mt-4")
            preview_content = html.Div([preview_content, outlier_chart])
        if popup_alert:
            return (html.Div([popup_alert, preview_content]), overview_table, null_count_selected, outlier_count_selected,
                    clean_btn_disabled, rule_error, rule_style)
        return (preview_content, overview_table, null_count_selected, outlier_count_selected,
                clean_btn_disabled, rule_error, rule_style)
    except Exception as e:
        logger.error(f"Error in update_data_preview_and_stats: {str(e)}")
        return ("Error loading data", "", "0", "0", True, f"Error: {str(e)}", {'display': 'block'})

@app.callback(
    [Output('cleaned-data-store', 'children'),
     Output('cleaning-summary', 'children'),
     Output('cleaned-data-table', 'children'),
     Output('results-section', 'style'),
     Output('nulls-handled', 'children'),
     Output('outliers-removed', 'children'),
     Output('final-rows', 'children'),
     Output('download-btn', 'disabled'),
     Output('reset-btn', 'disabled'),
     Output('alert-container', 'children', allow_duplicate=True)],
    [Input('clean-btn', 'n_clicks'),
     Input('reset-confirm', 'n_clicks')],
    [State('raw-data-store', 'children'),
     State('numerical-columns', 'value'),
     State('categorical-columns', 'value'),
     State('null-method', 'value'),
     State('outlier-method', 'value'),
     State('outlier-handling', 'value'),
     State('outlier-threshold', 'value'),
     State('if-then-rule', 'value')],
    prevent_initial_call=True
)
def process_data(clean_clicks, reset_clicks, data_json, num_cols, cat_cols, null_method, 
                outlier_method, outlier_handling, outlier_threshold, if_then_rule):
    ctx = callback_context
    if not ctx.triggered:
        return '', '', '', {'display': 'none'}, "0", "0", "0", True, True, ''
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'reset-confirm':
        return '', '', '', {'display': 'none'}, "0", "0", "0", True, True, dbc.Alert("Data and settings have been reset", color="info", dismissable=True, className="mt-4")
    if trigger_id == 'clean-btn' and data_json:
        try:
            df = pd.read_json(data_json, orient='split')
            if_then_rule_dict = None
            if if_then_rule and if_then_rule.strip():
                rule_dict, rule_msg, is_valid, error_msg = parse_if_then_rule(if_then_rule, df.columns.tolist())
                if is_valid:
                    if_then_rule_dict = rule_dict
                else:
                    alert = dbc.Alert(f"If-Then Rule Error: {error_msg or rule_msg}", color="danger", dismissable=True, className="mt-4")
                    return '', '', '', {'display': 'none'}, "0", "0", "0", True, True, alert
            outlier_cols = num_cols or []
            df_cleaned, nulls_handled, outliers_removed, rule_status = clean_data(
                df, num_cols or [], cat_cols or [], null_method, outlier_method, outlier_threshold, if_then_rule_dict if if_then_rule_dict else None
            )
            cleaned_data_json = df_cleaned.to_json(date_format='iso', orient='split')
            summary_items = []
            if rule_status and "successfully" in rule_status:
                summary_items.append(html.Li(f"✅ {rule_status}", className="text-gray-700"))
            if nulls_handled > 0:
                summary_items.append(html.Li(f"🔧 Handled {nulls_handled} NULL values using {null_method}", className="text-gray-700"))
            if outliers_removed > 0:
                if outlier_handling == 'remove':
                    summary_items.append(html.Li(f"🎯 Removed {outliers_removed} outliers using {outlier_method}", className="text-gray-700"))
                elif outlier_handling in ['fill_mean', 'fill_median', 'fill_mode', 'interpolate', 'knn']:
                    summary_items.append(html.Li(f"💡 Filled {outliers_removed} outliers using {outlier_method}", className="text-gray-700"))
                else:
                    summary_items.append(html.Li(f"💡 Handled {outliers_removed} outliers using {outlier_method}", className="text-gray-700"))
            if not summary_items:
                summary_items.append(html.Li("ℹ️ No cleaning operations were performed", className="text-gray-700"))
            summary = html.Div([
                html.H5("Cleaning Summary", className="text-base font-medium text-gray-900 mb-2"),
                html.Ul(summary_items, className="list-disc pl-5")
            ])
            cleaned_table = create_data_table(df_cleaned, 'cleaned-data-table-display')
            alert = dbc.Alert("Data cleaning completed successfully!", color="success", dismissable=True, className="mt-4")
            return (cleaned_data_json, summary, cleaned_table, {'display': 'block'}, 
                    str(nulls_handled), str(outliers_removed), str(len(df_cleaned)), False, False, alert)
        except Exception as e:
            logger.error(f"Error in process_data: {str(e)}")
            return '', '', '', {'display': 'none'}, "0", "0", "0", True, True, \
                   dbc.Alert(f"Error during cleaning: {str(e)}", color="danger", dismissable=True, className="mt-4")

@app.callback(
    Output('download-cleaned-data', 'data'),
    [Input('download-btn', 'n_clicks')],
    [State('cleaned-data-store', 'children')],
    prevent_initial_call=True
)
def download_cleaned_data(n_clicks, cleaned_data_json):
    if n_clicks and cleaned_data_json:
        df_cleaned = pd.read_json(cleaned_data_json, orient='split')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_data_{timestamp}.csv"
        return dcc.send_data_frame(df_cleaned.to_csv, filename, index=False)

@app.callback(
    Output('outlier-threshold', 'value'),
    Output('outlier-threshold', 'disabled'),
    Input('outlier-method', 'value')
)
def update_threshold_value(method):
    if method == 'zscore':
        return 3, False
    elif method == 'iqr':
        return 1.5, False
    else:
        return 3, True

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    print("\n" + "="*70)
    print("🚀 PBS DataClean Dashboard - Advanced Data Cleaning & Analysis Tool")
    print("="*70)
    print("\n📋 Features:")
    print("   • File Upload (CSV, Excel)")
    print("   • Data Preview with NULL/Outlier Highlighting")
    print("   • Advanced If-Then Rule Engine")
    print("   • NULL Value Handling")
    print("   • Outlier Detection & Removal")
    print("   • Interactive Data Tables")
    print("   • Statistics Dashboard")
    print("   • Data Export")
    print("   • Responsive Design")
    print("\n🔧 Technical Stack:")
    print("   • Dash + Tailwind CSS")
    print("   • Pandas + NumPy")
    print("   • Plotly + SciPy")
    print("   • Font Awesome Icons")
    print("\n🌐 Access the application at: http://localhost:{}".format(port))
    print("📊 Upload your data files and start cleaning!")
    print("\n⚠️ Note: Large files may impact performance. For optimal use, consider files under 100MB.")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=port)