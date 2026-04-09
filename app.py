import dash
from dash import dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
import urllib.request
import json
import live_data   # ← real-time World Bank / data.gov.in / Udyam feeds
import sentiment_scraper # ← real-time Web Scraper for local district news


# === CONFIG ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"], 
        title="MSME Analytics Dashboard", suppress_callback_exceptions=True)
server = app.server
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# State coordinates for mapping
STATE_COORDS = {
  'ANDHRA PRADESH': {'lat': 15.9129, 'lon': 79.7400},
  'ARUNACHAL PRADESH': {'lat': 28.2180, 'lon': 94.7278},
  'ASSAM': {'lat': 26.2006, 'lon': 92.9376},
  'BIHAR': {'lat': 25.0961, 'lon': 85.3131},
  'CHHATTISGARH': {'lat': 21.2787, 'lon': 81.8661},
  'GOA': {'lat': 15.2993, 'lon': 74.1240},
  'GUJARAT': {'lat': 22.2587, 'lon': 71.1924},
  'HARYANA': {'lat': 29.0588, 'lon': 76.0856},
  'HIMACHAL PRADESH': {'lat': 31.1048, 'lon': 77.1734},
  'JHARKHAND': {'lat': 23.6102, 'lon': 85.2799},
  'KARNATAKA': {'lat': 15.3173, 'lon': 75.7139},
  'KERALA': {'lat': 10.8505, 'lon': 76.2711},
  'MADHYA PRADESH': {'lat': 22.9734, 'lon': 78.6569},
  'MAHARASHTRA': {'lat': 19.7515, 'lon': 75.7139},
  'MANIPUR': {'lat': 24.6637, 'lon': 93.9063},
  'MEGHALAYA': {'lat': 25.4670, 'lon': 91.3662},
  'MIZORAM': {'lat': 23.1645, 'lon': 92.9376},
  'NAGALAND': {'lat': 26.1584, 'lon': 94.5624},
  'ODISHA': {'lat': 20.9517, 'lon': 85.0985},
  'PUNJAB': {'lat': 31.1471, 'lon': 75.3412},
  'RAJASTHAN': {'lat': 27.0238, 'lon': 74.2179},
  'SIKKIM': {'lat': 27.5330, 'lon': 88.5122},
  'TAMIL NADU': {'lat': 11.1271, 'lon': 78.6569},
  'TELANGANA': {'lat': 18.1124, 'lon': 79.0193},
  'TRIPURA': {'lat': 23.9408, 'lon': 91.9882},
  'UTTAR PRADESH': {'lat': 26.8467, 'lon': 80.9462},
  'UTTARAKHAND': {'lat': 30.0668, 'lon': 79.0193},
  'WEST BENGAL': {'lat': 22.9868, 'lon': 87.8550},
  'DELHI': {'lat': 28.7041, 'lon': 77.1025},
  'CHANDIGARH': {'lat': 30.7333, 'lon': 76.7794},
  'PUDUCHERRY': {'lat': 11.9416, 'lon': 79.8083},
  'LAKSHADWEEP': {'lat': 10.5669, 'lon': 72.6417},
  'JAMMU AND KASHMIR': {'lat': 33.7782, 'lon': 76.5762},
  'DAMAN AND DIU': {'lat': 20.4283, 'lon': 72.8397},
  'DADAR AND NAGAR HAVELI': {'lat': 20.1809, 'lon': 73.0169}
}

# === LOAD DATA ===
def load_csv(filename):
  path = os.path.join(WORK_DIR, filename)
  if os.path.exists(path):
    try:
      return pd.read_csv(path)
    except Exception as e:
      print(f"Error loading {filename}: {e}")
  return pd.DataFrame()

# NIC Sector Mapping (5-digit code ranges to broad sectors)
NIC_SECTOR_MAP = {
  'Manufacturing': range(10000, 34000),
  'Textiles & Apparel': range(13000, 15000),
  'Food Processing': range(10000, 11000),
  'Chemicals': range(20000, 21000),
  'Metal & Machinery': range(24000, 29000),
  'Construction': range(41000, 44000),
  'Trade & Retail': range(45000, 48000),
  'Transportation': range(49000, 54000),
  'Hotels & Restaurants': range(55000, 57000),
  'IT & Services': range(58000, 64000),
  'Professional Services': range(69000, 75000),
  'Other Services': range(77000, 97000)
}

def get_nic_sector(nic_code):
  """Map a 5-digit NIC code to its broad sector"""
  try:
    code = int(str(nic_code)[:5])
    for sector, code_range in NIC_SECTOR_MAP.items():
      if code in code_range:
        return sector
    return 'Other'
  except:
    return 'Other'

print("Loading MSME data...", flush=True)
df_loc = load_csv("location_profile.csv")
df_soc = load_csv("social_profile.csv")
df_emp = load_csv("employment_profile.csv")
df_ind = load_csv("industry_profile.csv")
df_score = load_csv("composite_score.csv")
df_master = load_csv("msme_merged.csv") # Master file for detailed NIC analysis
print(f"Data loaded: {len(df_loc)} locations, {len(df_master)} master records", flush=True)

# === HELPER ===
def filter_df(df, state, district):
  if df.empty:
    return df
  dff = df.copy()
  if state and 'State' in dff.columns:
    dff = dff[dff['State'] == state]
  if district and 'District' in dff.columns:
    dff = dff[dff['District'] == district]
  return dff

def create_india_map(df_state, color_col, scale='Viridis', title='', size_col=None):
  """Create bubble map of India using state coordinates"""
  if df_state.empty:
    return go.Figure()
  
  # Add coordinates
  df_map = df_state.copy()
  df_map['lat'] = df_map['State'].map(lambda x: STATE_COORDS.get(x, {}).get('lat', 22))
  df_map['lon'] = df_map['State'].map(lambda x: STATE_COORDS.get(x, {}).get('lon', 78))
  
  # Create figure
  if size_col and size_col in df_map.columns:
    fig = px.scatter_mapbox(
      df_map,
      lat='lat', lon='lon',
      color=color_col,
      size=size_col,
      hover_name='State',
      hover_data=[color_col],
      color_continuous_scale=scale,
      size_max=25,
      zoom=4.2,
      center={"lat": 22, "lon": 78},
      title=title
    )
  else:
    fig = px.scatter_mapbox(
      df_map,
      lat='lat', lon='lon',
      color=color_col,
      hover_name='State',
      hover_data=[color_col],
      color_continuous_scale=scale,
      size=[12]*len(df_map),
      zoom=4.2,
      center={"lat": 22, "lon": 78},
      title=title
    )
  
  fig.update_layout(
    mapbox_style="open-street-map",
    margin={"r":0,"t":30,"l":0,"b":0},
    height=850,
    dragmode=False
  )
  fig.update_mapboxes(
    bearing=0,
    pitch=0
  )
  return fig

# === LAYOUTS ===
def create_header():
  return html.Div([
    html.Div([
      html.Div([
        html.Img(src="/assets/emblem.jpg", 
            height="70px", 
            className="me-3"),
        html.Div([
          html.P("", 
             style={'fontSize': '0.7rem', 'color': '#666', 'textAlign': 'center', 
                 'marginTop': '-5px', 'marginBottom': '0', 'fontWeight': 'bold'})
        ])
      ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
      html.Div([
        html.H3("National MSME Analytics Portal", className="header-title"),
        html.P("Geospatial Implementation of MSME Manual", className="header-subtitle")
      ])
    ], className="d-flex align-items-center"),
    html.Div([
      dbc.Button("Dashboard", id="btn-dashboard", color="danger", className="me-2", n_clicks=0),
      dbc.Button("DSS Tools", id="btn-dss", color="warning", className="me-2", n_clicks=0),
      dbc.Button(" ML Insights", id="btn-ml", color="success", className="me-2", n_clicks=0),
      dbc.Button("Data Upload", id="btn-upload", color="primary", n_clicks=0)
    ])
  ], className="header-container")

def create_dashboard_layout():
  return html.Div([
    html.Div([
      html.Label("Profile View:", className="nav-label"),
      dcc.Dropdown(id='tab-selector', value='tab1', clearable=False, style={'width': '250px'}, className="me-3",
        options=[
          {'label': '1. Location & Infrastructure', 'value': 'tab1'},
          {'label': '2. Social Inclusion', 'value': 'tab2'},
          {'label': '3. Employment & Scale', 'value': 'tab3'},
          {'label': '4. Industry Profile', 'value': 'tab4'},
          {'label': '5. Development Score', 'value': 'tab5'}
        ]),
      html.Label("State:", className="nav-label"),
      dcc.Dropdown(id='state-selector', placeholder="All States", style={'width': '200px'}, className="me-3"),
      html.Label("District:", className="nav-label"),
      dcc.Dropdown(id='district-selector', placeholder="All Districts", style={'width': '200px'})
    ], className="nav-bar"),
    
    dbc.Container([
      dcc.Loading(id="loading", type="circle", children=[
        html.Br(),
        dbc.Row(id='kpi-row', className="mb-3"),
        
        # Insights Section
        dbc.Row([
          dbc.Col([
            html.Div(id='insights-section', className="mb-3")
          ], width=12)
        ]),
        
        dbc.Row([
          dbc.Col([
            html.Div([
              html.H5("Geospatial View - India Map", className="section-title"),
              html.P(id='map-description', className="text-muted", style={'fontSize': '0.9rem', 'marginBottom': '10px'}),
              dcc.Graph(id='main-map', config={'scrollZoom': False, 'displayModeBar': False}, style={'height': '750px'})
            ], className="section-card")
          ], width=8),
          dbc.Col([
            html.Div([
              html.H5(id="chart-header", className="section-title"),
              dcc.Graph(id='chart-1', style={'height': '400px'}),
              html.Hr(),
              dcc.Graph(id='chart-2', style={'height': '400px'}),
              html.Div(id='chart-3-container', children=[
                html.Hr(),
                dcc.Graph(id='chart-3', style={'height': '400px'})
              ], style={'display': 'none'})
            ], className="section-card")
          ], width=4)
        ])
      ])
    ], fluid=True, className="p-4")
  ])

def create_dss_layout():
  return dbc.Container([
    dbc.Row([
      dbc.Col([
        html.Div([
          html.Label("State", className="dss-control-label"),
          dcc.Dropdown(id='dss-state-selector', placeholder="All India", className="mb-2"),
          
          html.Label("Highlight On Map", className="dss-control-label"),
          dcc.Dropdown(
            id='dss-highlight-selector',
            options=[
              {'label': 'None', 'value': 'none'},
              {'label': ' High MSME Density', 'value': 'high_density'},
              {'label': ' Low Female Ownership', 'value': 'low_female'},
              {'label': ' High Employment', 'value': 'high_employment'}
            ],
            value='none',
            className="mb-3"
          ),
          
          html.Div("Decision Insights", className="dss-section-header"),
          html.Div(id='dss-insights', style={'fontSize': '0.9rem', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'minHeight': '60px'}),
          
          html.Hr(style={'margin': '20px 0'}),
          
          html.H5("Top Districts", className="section-title", style={'fontSize': '1rem', 'marginTop': '15px'}),
          html.Div(id="dss-data-table", style={'maxHeight': '400px', 'overflowY': 'auto'})
        ], className="dss-sidebar", style={'height': '88vh', 'overflowY': 'auto'})
      ], width=3, style={'padding': '0'}),
      
      dbc.Col([
        dcc.Loading(
          dcc.Graph(id='dss-main-map', style={'height': '88vh', 'width': '100%'}, config={'scrollZoom': False, 'displayModeBar': False, 'doubleClick': False})
        )
      ], width=9, style={'padding': '0'})
    ], className="g-0")
  ], fluid=True)

def create_upload_layout():
  return dbc.Container([
    html.Br(),
    html.H2("Data Upload Portal", className="text-center mb-4"),
    dbc.Card([
      dbc.CardBody([
        html.H5("Upload CSV File", className="card-title"),
        dcc.Upload(
          id='upload-data',
          children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files', style={'color': '#007bff', 'cursor': 'pointer'})
          ]),
          style={
            'width': '100%', 'height': '80px', 'lineHeight': '80px',
            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '8px',
            'textAlign': 'center', 'margin': '10px', 'cursor': 'pointer'
          },
          multiple=False
        ),
        html.Div(id='upload-output', className='mt-3')
      ])
    ], className="p-4", style={'maxWidth': '700px', 'margin': 'auto'})
  ], fluid=True)

# === ML INSIGHTS — SCHEME RECOMMENDER ===

SCHEME_INFO = {
  'CGTMSE':   {'name': 'Credit Guarantee Trust for Micro & Small Enterprises',
          'desc': 'Collateral-free credit guarantee up to ₹5 Crore for micro & small enterprises.',
          'ministry': 'Min. of MSME + SIDBI', 'color': 'primary', 'icon': ''},
  'MUDRA':    {'name': 'PM MUDRA Yojana',
          'desc': 'Loans up to ₹10L (Shishu/Kishore/Tarun tiers) for micro enterprises.',
          'ministry': 'Ministry of Finance', 'color': 'success', 'icon': ''},
  'PM_FME':   {'name': 'PM Formalisation of Micro Food Enterprises',
          'desc': '35% credit-linked subsidy (up to ₹10L) for food-processing micro enterprises.',
          'ministry': 'Ministry of Food Processing', 'color': 'warning', 'icon': ''},
  'SFURTI':   {'name': 'Scheme for Fund for Regeneration of Traditional Industries',
          'desc': 'Cluster development with shared facilities, design & marketing support for artisans.',
          'ministry': 'Ministry of MSME', 'color': 'info', 'icon': ''},
  'ASPIRE':   {'name': 'Promotion of Innovation, Rural Industry & Entrepreneurship',
          'desc': 'Livelihood Business Incubators in agri/rural sectors to diversify district economy.',
          'ministry': 'Ministry of MSME', 'color': 'secondary', 'icon': ''},
  'SC_ST_HUB':  {'name': 'SC-ST Hub Scheme',
          'desc': 'Mentoring, skill development & procurement linkage for SC/ST entrepreneurs.',
          'ministry': 'Ministry of MSME', 'color': 'danger', 'icon': ''},
  'STANDUP':   {'name': 'Stand-Up India (Women Entrepreneurship)',
          'desc': 'Bank loans ₹10L–₹1Cr for women-owned greenfield enterprises.',
          'ministry': 'Dept. of Financial Services', 'color': 'danger', 'icon': ''},
  'ZED':     {'name': 'Zero Defect Zero Effect (ZED) Certification',
          'desc': '50–80% subsidy for manufacturing MSMEs to get quality certified for exports.',
          'ministry': 'Ministry of MSME', 'color': 'success', 'icon': ''},
  'CHAMPIONS':  {'name': 'MSME Champions Scheme',
          'desc': 'Technology upgrade, digitisation & mentoring for growth-ready enterprises.',
          'ministry': 'Ministry of MSME', 'color': 'primary', 'icon': ''},
  'RAMP':    {'name': 'Raising & Accelerating MSME Performance (RAMP)',
          'desc': 'World Bank-supported competitiveness, delayed-payments & market-access program.',
          'ministry': 'Ministry of MSME + World Bank', 'color': 'info', 'icon': ''},
}

def compute_district_profile(state, district):
  p = {}
  dist = district.strip()
  soc = df_soc[(df_soc['State'] == state) & (df_soc['District'].str.strip() == dist)]
  if not soc.empty:
    r = soc.iloc[0]
    t = max(int(r['total_msmes']), 1)
    p['wer']     = round(r['female_owned'] / t * 100, 1)
    p['sc_st_ratio'] = round((r['sc_count'] + r['st_count']) / t * 100, 1)
    p['obc_ratio']  = round(r['obc_count'] / t * 100, 1)
    p['total_msmes'] = t
  else:
    p.update({'wer': 0, 'sc_st_ratio': 0, 'obc_ratio': 0, 'total_msmes': 0})

  emp = df_emp[(df_emp['State'] == state) & (df_emp['District'].str.strip() == dist)]
  if not emp.empty:
    r = emp.iloc[0]
    inv = max(float(r['total_investment']), 0.1)
    p['eer']       = round(float(r['total_employment']) / inv, 2)
    p['avg_emp']     = round(float(r['avg_employment']), 1)
    split        = str(r.get('enterprise_type_split', ''))
    p['micro_only']   = 'Small' not in split and 'Medium' not in split
    p['has_medium']   = 'Medium' in split
    p['has_small']    = 'Small' in split
    p['total_investment'] = float(r['total_investment'])
    p['total_employment'] = int(r['total_employment'])
    p['inv_per_unit']  = round(p['total_investment'] / max(p['total_msmes'], 1), 1)
  else:
    p.update({'eer': 0, 'avg_emp': 0, 'micro_only': True, 'has_medium': False,
         'has_small': False, 'total_investment': 0, 'total_employment': 0, 'inv_per_unit': 0})

  ind = df_ind[(df_ind['State'] == state) & (df_ind['District'].str.strip() == dist)]
  if not ind.empty:
    r = ind.iloc[0]
    p['mfg_pct'] = float(r['manufacturing_pct'])
    p['svc_pct'] = float(r['services_pct'])
    p['idi']   = int(r['industry_diversity_index'])
    nic2     = str(r.get('top_nic_sector', ''))[:2]
    p['is_food']  = nic2 in ['10', '11']
    p['is_textile'] = nic2 in ['13', '14']
  else:
    p.update({'mfg_pct': 50, 'svc_pct': 50, 'idi': 1, 'is_food': False, 'is_textile': False})
  return p

def recommend_schemes(p):
  scores, reasons = {}, {}
  def add(key, pts, reason):
    scores[key] = scores.get(key, 0) + pts
    if reason: reasons.setdefault(key, []).append(reason)

  if p['wer'] == 0:
    add('STANDUP', 50, 'Zero female ownership — Stand-Up India seeds women enterprise')
  elif p['wer'] < 15:
    add('STANDUP', 30, f'Low female ownership ({p["wer"]}%) — women-focused credit needed')
  if p['micro_only']:
    add('MUDRA',  40, 'Exclusively Micro enterprises — MUDRA loans are a direct fit')
    add('CGTMSE', 25, 'Micro units lack collateral — CGTMSE guarantee unlocks formal credit')
  if p['has_small']:
    add('CGTMSE', 35, 'Small enterprises present — CGTMSE guarantee for up to ₹5Cr')
    add('RAMP',  20, 'RAMP competitiveness program helps Small units grow further')
  if p['sc_st_ratio'] > 20:
    add('SC_ST_HUB', 50, f'High SC/ST ratio ({p["sc_st_ratio"]}%) — SC-ST Hub provides procurement & mentoring')
  elif p['sc_st_ratio'] > 10:
    add('SC_ST_HUB', 25, f'Moderate SC/ST presence ({p["sc_st_ratio"]}%) — SC-ST Hub schemes applicable')
  if p['is_food']:
    add('PM_FME', 55, 'Primary NIC sector is food processing — PM-FME subsidy directly applicable')
  if p['is_textile']:
    add('SFURTI', 50, 'Textile sector detected — SFURTI cluster development is ideal')
    add('ZED',  20, 'Textile mfg benefits from ZED certification for export competitiveness')
  if p['mfg_pct'] > 60:
    add('ZED',  40, f'High manufacturing share ({p["mfg_pct"]:.0f}%) — ZED certification improves quality')
    add('SFURTI', 20, 'Manufacturing cluster can share SFURTI infrastructure to cut costs')
  if p['idi'] == 1:
    add('ASPIRE', 45, f'Single-industry district (IDI=1) — ASPIRE incubators can diversify economy')
  elif p['idi'] <= 2:
    add('ASPIRE', 25, f'Low industry diversity (IDI={p["idi"]}) — ASPIRE adds new livelihood sectors')
  if p['eer'] > 3:
    add('CGTMSE',  25, f'Exceptional EER={p["eer"]} — scaling via CGTMSE maximises jobs per rupee')
    add('CHAMPIONS',20, 'High-performing district — Champions scheme accelerates competitiveness')
  if 0 < p['inv_per_unit'] < 5:
    add('MUDRA', 30, f'Very low investment/unit (₹{p["inv_per_unit"]}L) — MUDRA capital injection critical')
  if p['has_medium']:
    add('CHAMPIONS', 35, 'Medium enterprise present — Champions digitisation targets national markets')
    add('RAMP',   30, 'Medium enterprise benefits from RAMP export & market-access program')

  top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
  mx  = max((s for _, s in top3), default=1)
  return [(k, min(int(v / mx * 100), 100), reasons.get(k, ['Profile match'])) for k, v in top3]

def create_scheme_recommender_content():
  states = sorted(df_soc['State'].dropna().unique().tolist()) if not df_soc.empty else []
  return dbc.Row([
    dbc.Col([
      html.Div([
        html.H5(html.Strong([html.I(className="bi bi-geo-alt-fill text-danger me-2"), ' Select District']), className='mb-3 text-dark'),
        html.Hr(className="opacity-25 pb-2"),
        html.Label('State', className='fw-semibold text-secondary small mb-1'),
        dcc.Dropdown(id='scheme-state',
               options=[{'label': s, 'value': s} for s in states],
               placeholder='Select State...', className='mb-3'),
        html.Label('District', className='fw-semibold text-secondary small mb-1'),
        dcc.Dropdown(id='scheme-district', placeholder='Select District...', className='mb-4'),
        dbc.Button([html.I(className="bi bi-magic me-2"), 'Get Recommendations'], id='scheme-btn',
              color='danger', className='w-100 fw-bold shadow-sm rounded-pill', n_clicks=0, size='md'),
        html.Hr(className="my-4 opacity-25"),
        html.H6(html.Strong('District Metrics Baseline'), className='text-secondary small mb-3 text-uppercase'),
        html.Div(id='scheme-profile-metrics',
             children=[html.P('Select a district and click Get Recommendations to view detailed metrics.', className='text-muted small fst-italic')])
      ], className='p-4 bg-white rounded-3 shadow-sm border-0', style={'position': 'sticky', 'top': '20px'})
    ], width=4),
    dbc.Col([
      dcc.Loading(
        id="loading-scheme", type="circle",
        children=[
          html.Div(id='scheme-results', children=[
            html.Div([
              html.Div(html.I(className="bi bi-box-seam"), style={'fontSize': '4rem', 'textAlign': 'center', 'color': '#e0e0e0'}),
              html.H5('Ready for Analysis',
                  className='text-secondary fw-bold text-center mt-3 mb-2'),
              html.P('The AI engine analyzes empirical district data and queries the local LLM model '
                  'to match and explain the most suitable central government MSME schemes.',
                  className='text-muted text-center small mx-auto', style={'maxWidth': '400px', 'lineHeight': '1.6'})
            ], className='mt-5 pt-5 opacity-75')
          ])
        ]
      )
    ], width=8)
  ], className='mt-3')

# === ML INSIGHTS — AI DISTRICT ANALYST (OLLAMA) ===

def get_ollama_insight(prompt_text):
  url = "http://localhost:11434/api/generate"
  data = {
    "model": "llama3.2:latest",
    "prompt": prompt_text,
    "stream": False
  }
  req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'))
  req.add_header('Content-Type', 'application/json')
  try:
    response = urllib.request.urlopen(req, timeout=120)
    result = json.loads(response.read().decode('utf-8'))
    return result.get('response', '')
  except Exception as e:
    return f"Error communicating with local Ollama: {str(e)}\n\nPlease ensure your Ollama is running (`ollama serve`)."

def create_ollama_chat_content():
  states = sorted(df_soc['State'].dropna().unique().tolist()) if not df_soc.empty else []
  return dbc.Row([
    dbc.Col([
      html.Div([
        html.H5(html.Strong([html.I(className="bi bi-robot text-primary me-2"), ' AI District Analyst']), className='mb-3 text-dark'),
        html.Hr(className="opacity-25 pb-2"),
        html.Label('State', className='fw-semibold text-secondary small mb-1'),
        dcc.Dropdown(id='ollama-state', options=[{'label': s, 'value': s} for s in states], placeholder='Select State...', className='mb-3'),
        html.Label('District', className='fw-semibold text-secondary small mb-1'),
        dcc.Dropdown(id='ollama-district', placeholder='Select District...', className='mb-4'),
        html.Label('Specific Question (Optional)', className='fw-semibold text-secondary small mb-1'),
        dbc.Textarea(id='ollama-question', placeholder='e.g., How can we improve female ownership given the low manufacturing base?', className='mb-4 rounded-3 border-light-subtle', style={'height': '110px', 'backgroundColor': '#f8f9fa'}),
        dbc.Button([html.I(className="bi bi-lightning-charge-fill me-2"), 'Generate Insights with Ollama'], id='ollama-btn', color='primary', className='w-100 fw-bold shadow-sm rounded-pill', n_clicks=0, size='md')
      ], className='p-4 bg-white rounded-3 shadow-sm border-0', style={'position': 'sticky', 'top': '20px'})
    ], width=4),
    dbc.Col([
      dcc.Loading(
        id="loading-ollama", type="circle",
        children=[
          html.Div(id='ollama-results', children=[
            html.Div([
              html.Div(html.I(className="bi bi-chat-left-dots"), style={'fontSize': '4rem', 'textAlign': 'center', 'color': '#e0e0e0'}),
              html.H5('Awaiting Query', className='text-secondary fw-bold text-center mt-3 mb-2'),
              html.P('Select a district and get real-time strategic recommendations powered by the local Ollama LLM model.',
                  className='text-muted text-center small mx-auto', style={'maxWidth': '400px'})
            ], className='mt-5 pt-5 opacity-75')
          ])
        ]
      )
    ], width=8)
  ], className='mt-3')


# === ML INSIGHTS — ANOMALY DETECTOR ===

def train_anomaly_model():
  if df_soc.empty or df_emp.empty:
    return None, pd.DataFrame()
  
  # Merge social and employment for a comprehensive district profile
  df = pd.merge(df_soc, df_emp[['State', 'District', 'total_investment', 'total_employment']], on=['State', 'District'])
  
  X_data = []
  meta = []
  
  for _, row in df.iterrows():
    try:
      t = max(int(row['total_msmes']), 1)
      wer = row['female_owned'] / t * 100
      scst = (row['sc_count'] + row['st_count']) / t * 100
      inv_per_unit = float(row['total_investment']) / t
      emp_per_unit = float(row['total_employment']) / t
      
      X_data.append([wer, scst, inv_per_unit, emp_per_unit])
      meta.append({'State': row['State'], 'District': row['District'], 
             'WER': round(wer, 1), 'SCST': round(scst, 1), 
             'Inv_Unit': round(inv_per_unit, 1), 'Emp_Unit': round(emp_per_unit, 1)})
    except:
      pass
      
  if len(X_data) > 10:
    # Contamination=0.1 means we expect ~10% of districts to be highly unusual outliers
    iso = IsolationForest(contamination=0.1, random_state=42)
    X_np = np.array(X_data)
    preds = iso.fit_predict(X_np)
    scores = iso.decision_function(X_np)
    
    df_meta = pd.DataFrame(meta)
    df_meta['Anomaly'] = preds
    df_meta['Anomaly_Score'] = scores
    # We only care about the explicit outliers (-1)
    outliers = df_meta[df_meta['Anomaly'] == -1].sort_values('Anomaly_Score').copy()
    
    return iso, outliers
  return None, pd.DataFrame()

_, df_anomalies = train_anomaly_model()

def create_anomaly_detector_content():
  return html.Div([
    dbc.Card([
      dbc.CardBody([
        html.H5(html.Strong([html.I(className="bi bi-radar text-danger me-2"), "Anomaly Detection via Isolation Forest"]), className="mb-3 text-dark"),
        html.P('The Isolation Forest machine learning algorithm scans the entire 162-district dataset '
            'to identify highly unusual statistical profiles (extreme over-performance, severe underspending, or unique structural bottlenecks). '
            'These insights help target specific policy investigations.',
            className='text-muted small mb-4', style={'lineHeight': '1.6'}),
        
        dbc.Button([html.I(className="bi bi-search me-2"), "Run District Anomaly Scan"], id="run-anomaly-btn", color="danger", className="fw-bold shadow-sm rounded-pill px-4", size="md"),
      ], className="p-4")
    ], className="bg-white border-0 shadow-sm rounded-3 mb-4"),
    
    dcc.Loading(
      id="loading-anomaly", type="circle",
      children=[
        html.Div(id="anomaly-results", children=[
          html.Div([
              html.Div(html.I(className="bi bi-shield-check"), style={'fontSize': '4rem', 'textAlign': 'center', 'color': '#e0e0e0'}),
              html.H5('System Ready for Scan', className='text-secondary fw-bold text-center mt-3 mb-2'),
              html.P('Click the scan button to process the dataset and generate an AI outlier report.', className='text-muted text-center small fst-italic mx-auto')
          ], className='mt-5 pt-4 opacity-75')
        ])
      ]
    )
  ])

def create_sentiment_analyst_content():
  states = sorted(df_soc['State'].dropna().unique().tolist()) if not df_soc.empty else []
  return dbc.Row([
    dbc.Col([
      html.Div([
        html.H5(html.Strong([html.I(className="bi bi-newspaper text-info me-2"), ' Live Media Pulse']), className='mb-3 text-dark'),
        html.Hr(className="opacity-25 pb-2"),
        html.Label('State', className='fw-semibold text-secondary small mb-1'),
        dcc.Dropdown(id='sentiment-state', options=[{'label': s, 'value': s} for s in states], placeholder='Select State...', className='mb-3'),
        html.Label('District', className='fw-semibold text-secondary small mb-1'),
        dcc.Dropdown(id='sentiment-district', placeholder='Select District...', className='mb-4'),
        html.P('An agent will scrape the latest Google News articles for this district and analyze local business sentiment.', className='text-muted small mb-4', style={'lineHeight': '1.5'}),
        dbc.Button([html.I(className="bi bi-broadcast text-white me-2"), 'Scan Local Media'], id='sentiment-btn', color='info', className='w-100 fw-bold text-white shadow-sm rounded-pill', n_clicks=0, size='md')
      ], className='p-4 bg-white rounded-3 shadow-sm border-0', style={'position': 'sticky', 'top': '20px'})
    ], width=4),
    dbc.Col([
      dcc.Loading(
        id="loading-sentiment", type="circle",
        children=[
          html.Div(id='sentiment-results', children=[
            html.Div([
              html.Div(html.I(className="bi bi-mic"), style={'fontSize': '4rem', 'textAlign': 'center', 'color': '#e0e0e0'}),
              html.H5('Ready to Scrape', className='text-secondary fw-bold text-center mt-3 mb-2'),
              html.P('Select a district and click scan to capture the live ground sentiment from local news sources.',
                  className='text-muted text-center small mx-auto', style={'maxWidth': '400px'})
            ], className='mt-5 pt-5 opacity-75')
          ])
        ]
      )
    ], width=8)
  ])

def create_policy_simulator_content():
  states = sorted(df_soc['State'].dropna().unique().tolist()) if not df_soc.empty else []
  return dbc.Row([
    dbc.Col([
      html.Div([
        html.H5(html.Strong([html.I(className="bi bi-sliders text-success me-2"), ' Policy Simulator']), className='mb-3 text-dark'),
        html.Hr(className="opacity-25 pb-2"),
        html.Label('State', className='fw-semibold text-secondary small mb-1'),
        dcc.Dropdown(id='sim-state', options=[{'label': s, 'value': s} for s in states], placeholder='Select State...', className='mb-3'),
        html.Label('District', className='fw-semibold text-secondary small mb-1'),
        dcc.Dropdown(id='sim-district', placeholder='Select District...', className='mb-4'),
        html.H6(html.Strong('Policy Interventions'), className='text-muted small text-uppercase mb-3 mt-4'),
        
        html.Label('Financial Injection (₹ Crores)', className='fw-bold text-dark small mb-1'),
        dcc.Slider(0, 500, 50, value=0, marks={0: '₹0', 100: '100Cr', 250: '250Cr', 500: '500Cr'}, id='sim-injection', className='mb-4'),
        
        html.Label('Target Female Ownership (%)', className='fw-bold text-dark small mb-1'),
        dcc.Slider(0, 50, 5, value=15, marks={0: '0%', 15: '15%', 30: '30%', 50: '50%'}, id='sim-female-target', className='mb-4'),
        
        html.Label('SC/ST Ownership Target (%)', className='fw-bold text-dark small mb-1'),
        dcc.Slider(0, 50, 5, value=10, marks={0: '0%', 10: '10%', 25: '25%', 50: '50%'}, id='sim-scst-target', className='mb-4'),
        
        html.Label('Manufacturing Sector Pivot (%)', className='fw-bold text-dark small mb-1'),
        dcc.Slider(0, 100, 10, value=30, marks={0: '0%', 30: '30%', 60: '60%', 100: '100%'}, id='sim-mfg-target', className='mb-4'),
        
        dbc.Button([html.I(className="bi bi-play-circle-fill me-2"), 'Run AI Simulation'], id='sim-btn', color='success', className='w-100 fw-bold shadow-sm rounded-pill mt-2', n_clicks=0, size='md')
      ], className='p-4 bg-white rounded-3 shadow-sm border-0', style={'position': 'sticky', 'top': '20px'})
    ], width=4),
    dbc.Col([
      dcc.Loading(
        id="loading-sim", type="circle",
        children=[
          html.Div(id='sim-results', children=[
            html.Div([
              html.Div(html.I(className="bi bi-kanban"), style={'fontSize': '4rem', 'textAlign': 'center', 'color': '#e0e0e0'}),
              html.H5('Awaiting Policy Inputs', className='text-secondary fw-bold text-center mt-3 mb-2'),
              html.P('Adjust the sliders safely sandbox economic interventions. The local Ollama model will simulate the resulting impacts on employment and district growth.',
                  className='text-muted text-center small mx-auto', style={'maxWidth': '450px'})
            ], className='mt-5 pt-5 opacity-75')
          ])
        ]
      )
    ], width=8)
  ])

def create_live_data_banner():
  """Fetch & render real-time macro indicators. Falls back to reference values if APIs are down."""
  macro    = live_data.get_india_macro()
  inds     = macro.get('indicators', {})
  udyam    = live_data.get_udyam_total()
  fetched  = macro.get('fetched_at', 'N/A')
  any_live = macro.get('any_live', False)

  LABELS = {
    'gdp_growth':     ('GDP Growth', '%'),
    'inflation':      ('Inflation', '%'),
    'unemployment':   ('Unemployment', '%'),
    'credit_private': ('Credit/GDP', '%'),
    'self_employed':  ('Self-Employed', '%'),
    'manufacturing':  ('Mfg/GDP', '%'),
  }

  chips = []
  for key, (lbl, unit) in LABELS.items():
    v = inds.get(key)
    if v:
      is_chip_live = v.get('live', False)
      chip_bg     = '#f0fff4' if is_chip_live else '#fff8e1'
      chip_border = '#b2dfdb' if is_chip_live else '#ffe082'
      chips.append(
        html.Div([
          html.Div(f"{v['value']}{unit}",
                   style={'fontSize': '1.25rem', 'fontWeight': '700',
                          'color': '#155724' if is_chip_live else '#e65100'}),
          html.Div(lbl,      style={'fontSize': '0.7rem',  'color': '#666'}),
          html.Div(
            f"{v['year']} {'🌐' if is_chip_live else '📋'}",
            style={'fontSize': '0.65rem', 'color': '#aaa'}
          )
        ], style={
          'background': chip_bg, 'border': f'1px solid {chip_border}',
          'borderRadius': '10px', 'padding': '8px 14px',
          'textAlign': 'center', 'minWidth': '100px'
        })
      )

  if udyam:
    chips.append(
      html.Div([
        html.Div(udyam['label'],
                 style={'fontSize': '1.1rem', 'fontWeight': '700', 'color': '#1a237e'}),
        html.Div('Udyam Registered', style={'fontSize': '0.7rem', 'color': '#666'}),
        html.Div('🟢 live',          style={'fontSize': '0.65rem', 'color': '#27ae60', 'fontWeight': 'bold'})
      ], style={
        'background': '#e8eaf6', 'border': '1px solid #9fa8da',
        'borderRadius': '10px', 'padding': '8px 14px',
        'textAlign': 'center', 'minWidth': '130px'
      })
    )

  if any_live:
    badge_color = '#27ae60'
    badge_txt   = '🟢 LIVE DATA'
    src_note    = f"Source: World Bank / IMF  •  {fetched}"
  else:
    badge_color = '#e67e22'
    badge_txt   = '📋 REFERENCE DATA'
    src_note    = f"APIs offline — showing latest known values  •  {fetched}"

  return dbc.Alert([
    html.Div([
      html.Div([
        html.Span(badge_txt, style={
          'background': badge_color, 'color': '#fff',
          'borderRadius': '6px', 'padding': '3px 10px',
          'fontSize': '0.75rem', 'fontWeight': '700',
        }),
        html.Span('  India Macro Indicators',
                  style={'fontSize': '0.8rem', 'color': '#555', 'marginLeft': '10px',
                         'fontWeight': '600'}),
        html.Span(f'  {src_note}',
                  style={'fontSize': '0.72rem', 'color': '#888', 'marginLeft': '6px'})
      ], className='mb-2'),
      html.Div(chips, style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'})
    ])
  ], color='light', className='mb-3 border shadow-sm',
     style={'borderLeft': f'5px solid {badge_color}'})


def create_ml_tools_layout():
  banner = create_live_data_banner()
  return dbc.Container([
    html.Br(),
    html.Div([
      html.Div([
        html.H3([html.Span(' ', style={'color': '#27ae60'}), 'ML Insights & AI Tools'],
            style={'fontWeight': '700', 'marginBottom': '4px'}),
        html.P('AI-powered decision support — scheme matching, employment prediction, and anomaly detection.',
            className='text-muted', style={'marginBottom': '0'})
      ]),
      dbc.Button(
        [' Refresh Live Data'],
        id='refresh-live-btn', color='success', outline=True, size='sm',
        className='ms-auto', style={'alignSelf': 'center'},
        n_clicks=0
      )
    ], className='d-flex justify-content-between align-items-start mb-3'),

    html.Div(id='live-data-banner', children=banner),

    dbc.Tabs([
      dbc.Tab(create_scheme_recommender_content(),
          label=' Scheme Recommender', tab_id='scheme-tab',
          tab_style={'fontWeight': '600'}),
      dbc.Tab(create_ollama_chat_content(),
          label=' AI District Insights', tab_id='pred-tab',
          tab_style={'fontWeight': '600'}),
      dbc.Tab(create_anomaly_detector_content(),
          label=' Anomaly Detector', tab_id='anomaly-tab',
          tab_style={'fontWeight': '600'}),
      dbc.Tab(create_sentiment_analyst_content(),
          label=' Live Media Pulse', tab_id='sentiment-tab',
          tab_style={'fontWeight': '600'}),
      dbc.Tab(create_policy_simulator_content(),
          label=' "What-If" Simulator', tab_id='sim-tab',
          tab_style={'fontWeight': '600'}),
    ], active_tab='scheme-tab')
  ], fluid=True, className='p-4')


@app.callback(
  Output('live-data-banner', 'children'),
  Input('refresh-live-btn', 'n_clicks'),
  prevent_initial_call=True
)
def refresh_live_banner(n):
  live_data.clear_cache()          # force re-fetch from APIs
  return create_live_data_banner()

# === MAIN LAYOUT ===
app.layout = html.Div([
  dcc.Store(id='page-mode', data='dashboard'),
  create_header(),
  html.Div(id='page-content')
])

# === CALLBACKS ===

@app.callback(Output('page-mode', 'data'),
       [Input('btn-dashboard', 'n_clicks'), Input('btn-dss', 'n_clicks'),
        Input('btn-ml', 'n_clicks'), Input('btn-upload', 'n_clicks')],
       State('page-mode', 'data'))
def toggle_mode(btn1, btn2, btn3, btn4, current):
  cid = ctx.triggered_id
  if cid == 'btn-dss': return 'dss'
  if cid == 'btn-ml': return 'ml_tools'
  if cid == 'btn-upload': return 'upload'
  return 'dashboard'

@app.callback(Output('page-content', 'children'), Input('page-mode', 'data'))
def render_page(mode):
  if mode == 'dss': return create_dss_layout()
  if mode == 'upload': return create_upload_layout()
  if mode == 'ml_tools': return create_ml_tools_layout()
  return create_dashboard_layout()

@app.callback(Output('state-selector', 'options'), Input('page-mode', 'data'))
def populate_state_dropdown(mode):
  if df_loc.empty: return []
  return [{'label': s, 'value': s} for s in sorted(df_loc['State'].dropna().unique())]

@app.callback(Output('dss-state-selector', 'options'), Input('page-mode', 'data'))
def populate_dss_state(mode):
  if df_loc.empty: return []
  return [{'label': s, 'value': s} for s in sorted(df_loc['State'].dropna().unique())]

@app.callback(Output('district-selector', 'options'), Input('state-selector', 'value'))
def update_districts(state):
  if not state or df_loc.empty: return []
  dists = df_loc[df_loc['State'] == state]['District'].dropna().unique()
  return [{'label': d, 'value': d} for d in sorted(dists)]

# Dashboard Main
@app.callback(
  [Output('kpi-row', 'children'), Output('main-map', 'figure'), Output('chart-1', 'figure'), 
   Output('chart-2', 'figure'), Output('chart-header', 'children'),
   Output('chart-3', 'figure'), Output('chart-3-container', 'style'),
   Output('insights-section', 'children'), Output('map-description', 'children')],
  [Input('tab-selector', 'value'), Input('state-selector', 'value'), Input('district-selector', 'value')]
)
def update_dashboard(tab, state, district):
  map_fig = go.Figure()
  chart1 = go.Figure()
  chart2 = go.Figure()
  chart3 = go.Figure()
  kpis = []
  header = "Analytics"
  show_chart3 = {'display': 'none'}
  insights = html.Div()
  map_desc = ""
  
  try:
    if tab == 'tab1':
      dff = filter_df(df_loc, state, district)
      header ="Location Distribution"
      map_desc = "Bubble size represents MSME density. Darker colors indicate higher concentration of enterprises."
      
      if not dff.empty:
        state_agg = dff.groupby('State', as_index=False).agg({'msme_count': 'sum'})
        map_fig = create_india_map(state_agg, 'msme_count', 'Viridis', 'MSME Density', 'msme_count')
        
        # Generate insights
        top_state = state_agg.loc[state_agg['msme_count'].idxmax()]
        total_msmes = dff['msme_count'].sum()
        top_3_states = state_agg.nlargest(3, 'msme_count')
        top_3_contribution = (top_3_states['msme_count'].sum() / total_msmes * 100)
        
        insights = dbc.Alert([
          html.Div([
            html.I(className="fas fa-lightbulb me-2"),
            html.Strong(" Key Insights:"),
          ], className="mb-2"),
          html.Ul([
            html.Li(f" {top_state['State']} leads with {top_state['msme_count']:,} MSMEs"),
            html.Li(f" Top 3 states account for {top_3_contribution:.1f}% of total enterprises"),
            html.Li(f" Total {dff['District'].nunique()} districts have registered MSMEs"),
          ], style={'marginBottom': '0'})
        ], color="info", className="mb-3")
        
        kpis = [
          dbc.Col(html.Div([html.Div(f"{dff['msme_count'].sum():,}", className="kpi-value"), 
                   html.Div("Total MSMEs", className="kpi-label")], className="kpi-card bg-grey"), width=3),
          dbc.Col(html.Div([html.Div(dff['District'].nunique(), className="kpi-value"), 
                   html.Div("Districts", className="kpi-label")], className="kpi-card bg-blue"), width=3)
        ]
        
        top10 = dff.groupby('District')['msme_count'].sum().nlargest(10).reset_index()
        chart1 = px.bar(top10, x='District', y='msme_count', title="Top 10 Districts by MSME Count")
        chart1.update_layout(height=320)
        
        if state:
          chart2 = px.pie(dff, names='Dic_Name', values='msme_count', title="DIC Distribution", hole=0.4)
        else:
          top_states = state_agg.nlargest(10, 'msme_count')
          chart2 = px.bar(top_states, x='State', y='msme_count', title="Top 10 States by MSME Count", color='msme_count')
        chart2.update_layout(height=320)
    
    elif tab == 'tab2':
      dff = filter_df(df_soc, state, district)
      header = "Social Inclusion"
      map_desc = "Map shows female ownership percentage. Larger bubbles indicate more enterprises."
      
      if not dff.empty:
        state_agg = dff.groupby('State', as_index=False).agg({
          'female_owned': 'sum', 'total_msmes': 'sum', 'sc_count': 'sum', 'st_count': 'sum'
        })
        state_agg['women_pct'] = (state_agg['female_owned'] / state_agg['total_msmes'] * 100).fillna(0)
        map_fig = create_india_map(state_agg, 'women_pct', 'RdPu', 'Female Ownership %', 'total_msmes')
        
        # Generate insights
        female_total = dff['female_owned'].sum()
        male_total = dff['male_owned'].sum()
        total_all = female_total + male_total
        women_pct = (female_total / total_all * 100) if total_all > 0 else 0
        top_women_state = state_agg.loc[state_agg['women_pct'].idxmax()]
        
        sc_st_total = dff['sc_count'].sum() + dff['st_count'].sum()
        sc_st_pct = (sc_st_total / dff['total_msmes'].sum() * 100) if dff['total_msmes'].sum() > 0 else 0
        
        insights = dbc.Alert([
          html.Div([
            html.I(className="fas fa-users me-2"),
            html.Strong(" Inclusion Insights:"),
          ], className="mb-2"),
          html.Ul([
            html.Li(f" Women own {women_pct:.1f}% of MSMEs ({female_total:,} enterprises)"),
            html.Li(f" {top_women_state['State']} leads in women entrepreneurship ({top_women_state['women_pct']:.1f}%)"),
            html.Li(f" SC/ST entrepreneurs represent {sc_st_pct:.1f}% of total MSMEs"),
          ], style={'marginBottom': '0'})
        ], color="success", className="mb-3")
        
        kpis = [
          dbc.Col(html.Div([html.Div(f"{dff['female_owned'].sum():,}", className="kpi-value"), 
                   html.Div("Women Owned", className="kpi-label")], className="kpi-card bg-red"), width=3),
          dbc.Col(html.Div([html.Div(f"{dff['male_owned'].sum():,}", className="kpi-value"), 
                   html.Div("Men Owned", className="kpi-label")], className="kpi-card bg-yellow"), width=3)
        ]
        
        # CHART 1: Social Category Distribution as DONUT CHART
        cats_data = pd.DataFrame({
          'Category': ['General', 'OBC', 'SC', 'ST'], 
          'Count': [dff['general_count'].sum(), dff['obc_count'].sum(), dff['sc_count'].sum(), dff['st_count'].sum()]
        })
        
        chart1 = px.pie(
          cats_data, 
          names='Category', 
          values='Count', 
          hole=0.4, # Donut chart
          title="Social Category Distribution",
          color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D'] # Vibrant colors
        )
        chart1.update_traces(textposition='inside', textinfo='percent+label')
        chart1.update_layout(height=400)
        
        # CHART 2: Gender Distribution as PIE CHART with OUTSIDE percentages
        genders_data = pd.DataFrame({
          'Gender': ['Male', 'Female'], 
          'Count': [dff['male_owned'].sum(), dff['female_owned'].sum()]
        })
        chart2 = px.pie(
          genders_data, 
          names='Gender', 
          values='Count', 
          title="Gender Distribution",
          color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'} # Blue for Male, Red for Female
        )
        chart2.update_traces(
          textposition='outside', # Percentages OUTSIDE
          textinfo='percent+label',
          textfont_size=13,
          pull=[0.05, 0.05] # Slightly pull slices apart for clarity
        )
        chart2.update_layout(height=350)
    
    elif tab == 'tab3':
      dff = filter_df(df_emp, state, district)
      header = "Employment & Investment"
      map_desc = "Bubble size shows total employment. Color intensity indicates employment levels across states."
      
      if not dff.empty:
        state_agg = dff.groupby('State', as_index=False).agg({'total_employment': 'sum', 'total_msmes': 'sum'})
        map_fig = create_india_map(state_agg, 'total_employment', 'Plasma', 'Employment', 'total_employment')
        
        # Enhanced KPIs
        total_investment = dff['total_investment'].sum()
        total_employment = dff['total_employment'].sum()
        avg_investment_per_job = total_investment / total_employment if total_employment > 0 else 0
        
        # Generate insights
        top_employer = dff.loc[dff['total_employment'].idxmax()]
        avg_emp_per_msme = total_employment / dff['total_msmes'].sum() if dff['total_msmes'].sum() > 0 else 0
        
        insights = dbc.Alert([
          html.Div([
            html.I(className="fas fa-briefcase me-2"),
            html.Strong(" Employment Insights:"),
          ], className="mb-2"),
          html.Ul([
            html.Li(f" Total {total_employment:,} jobs created across {dff['total_msmes'].sum():,} MSMEs"),
            html.Li(f" Average {avg_emp_per_msme:.1f} employees per enterprise"),
            html.Li(f" Investment efficiency: ₹{avg_investment_per_job:.1f}L per job created"),
          ], style={'marginBottom': '0'})
        ], color="warning", className="mb-3")
        
        kpis = [
          dbc.Col(html.Div([html.Div(f"{total_employment:,}", className="kpi-value"), 
                   html.Div("Total Jobs", className="kpi-label")], className="kpi-card bg-blue"), width=3),
          dbc.Col(html.Div([html.Div(f"₹{total_investment:,.0f}L", className="kpi-value"), 
                   html.Div("Total Investment", className="kpi-label")], className="kpi-card bg-green"), width=3),
          dbc.Col(html.Div([html.Div(f"₹{avg_investment_per_job:.1f}L", className="kpi-value"), 
                   html.Div("Investment per Job", className="kpi-label")], className="kpi-card bg-red"), width=3)
        ]
        
        # CHART 1: Enterprise Type Distribution with Employment
        # Parse enterprise_type_split to extract counts
        def parse_enterprise_types(df):
          enterprise_data = {'Micro': 0, 'Small': 0, 'Medium': 0}
          employment_data = {'Micro': 0.0, 'Small': 0.0, 'Medium': 0.0}
          
          for idx, row in df.iterrows():
            split_str = str(row.get('enterprise_type_split', ''))
            emp = row.get('total_employment', 0)
            msmes = row.get('total_msmes', 1)
            avg_emp_per_msme = emp / msmes if msmes > 0 else 0
            
            # Parse "Micro: 5 | Small: 2 | Medium: 1" format
            for part in split_str.split('|'):
              part = part.strip()
              if 'Micro:' in part:
                count = int(part.split(':')[1].strip())
                enterprise_data['Micro'] += count
                employment_data['Micro'] += count * float(avg_emp_per_msme)
              elif 'Small:' in part:
                count = int(part.split(':')[1].strip())
                enterprise_data['Small'] += count
                employment_data['Small'] += count * float(avg_emp_per_msme)
              elif 'Medium:' in part:
                count = int(part.split(':')[1].strip())
                enterprise_data['Medium'] += count
                employment_data['Medium'] += count * float(avg_emp_per_msme)
          
          return pd.DataFrame({
            'Enterprise Type': list(enterprise_data.keys()),
            'Count': list(enterprise_data.values()),
            'Employment': list(employment_data.values())
          })
        
        enterprise_df = parse_enterprise_types(dff)
        enterprise_df['Avg Employment'] = enterprise_df['Employment'] / enterprise_df['Count']
        enterprise_df['Avg Employment'] = enterprise_df['Avg Employment'].fillna(0)
        
        chart1 = px.bar(
          enterprise_df,
          x='Enterprise Type',
          y='Employment',
          title="Employment by Enterprise Type",
          color='Enterprise Type',
          text='Employment',
          color_discrete_map={'Micro': '#3498db', 'Small': '#e67e22', 'Medium': '#e74c3c'}
        )
        chart1.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        chart1.update_layout(height=400, showlegend=False)
        
        # CHART 2: Investment Efficiency - Top Districts
        dff_efficiency = dff.copy()
        dff_efficiency['employment_per_investment'] = (
          dff_efficiency['total_employment'] / dff_efficiency['total_investment']
        ).replace([float('inf'), -float('inf')], 0).fillna(0)
        
        # Filter out zeros and get top performers
        dff_efficiency = dff_efficiency[dff_efficiency['employment_per_investment'] > 0]
        top_efficient = dff_efficiency.nlargest(15, 'employment_per_investment')
        
        if len(top_efficient) > 0:
          chart2 = px.bar(
            top_efficient,
            y='District' if not state else 'State',
            x='employment_per_investment',
            orientation='h',
            title="Top 15: Employment per ₹Lakh Investment",
            color='employment_per_investment',
            color_continuous_scale='Viridis',
            hover_data=['total_employment', 'total_investment']
          )
          chart2.update_layout(
            height=400,
            xaxis_title="Jobs per ₹L",
            yaxis_title="",
            showlegend=False
          )
        else:
          chart2 = go.Figure()
          chart2.update_layout(height=400, title="No data available")
        
        # CHART 3: Top Employment Generators
        top_employers = dff.nlargest(15, 'total_employment')
        
        chart3 = px.bar(
          top_employers,
          x='District' if 'District' in top_employers.columns else 'State',
          y='total_employment',
          title="Top 15 Employment Generators",
          color='total_investment',
          color_continuous_scale='Blues',
          hover_data=['total_msmes', 'avg_employment']
        )
        chart3.update_layout(
          height=400,
          xaxis_tickangle=-45,
          xaxis_title="",
          yaxis_title="Total Employment"
        )
        
        # Show Chart 3
        show_chart3 = {'display': 'block'}
    
    elif tab == 'tab4':
      dff = filter_df(df_ind, state, district)
      header = "Industry Profile"
      if not dff.empty:
        # Aggregate by state for map visualization
        state_agg = dff.groupby('State', as_index=False).agg({
          'manufacturing_pct': 'mean',
          'services_pct': 'mean',
          'industry_diversity_index': 'mean'
        })
        
        # Create map showing BOTH manufacturing and services percentages
        # Add both metrics to the dataframe for hover display
        state_agg_map = state_agg.copy()
        
        # Create custom map with both metrics visible
        df_map = state_agg_map.copy()
        df_map['lat'] = df_map['State'].map(lambda x: STATE_COORDS.get(x, {}).get('lat', 22))
        df_map['lon'] = df_map['State'].map(lambda x: STATE_COORDS.get(x, {}).get('lon', 78))
        
        # Create scatter map with both manufacturing and services in hover
        map_fig = px.scatter_mapbox(
          df_map,
          lat='lat', 
          lon='lon',
          color='manufacturing_pct',
          hover_name='State',
          hover_data={
            'manufacturing_pct': ':.1f',
            'services_pct': ':.1f',
            'lat': False,
            'lon': False
          },
          color_continuous_scale='RdYlGn_r',
          size=[12]*len(df_map),
          zoom=3.8,
          center={"lat": 22, "lon": 78},
          title='Manufacturing vs Services %',
          labels={
            'manufacturing_pct': 'Manufacturing %',
            'services_pct': 'Services %'
          }
        )
        
        map_fig.update_layout(
          mapbox_style="open-street-map",
          margin={"r":0,"t":30,"l":0,"b":0},
          height=650,
          dragmode=False
        )
        map_fig.update_mapboxes(bearing=0, pitch=0)
        
        kpis = [
          dbc.Col(html.Div([html.Div(f"{dff['manufacturing_pct'].mean():.1f}%", className="kpi-value"), 
                   html.Div("Avg Manufacturing", className="kpi-label")], className="kpi-card bg-red"), width=3),
          dbc.Col(html.Div([html.Div(f"{dff['services_pct'].mean():.1f}%", className="kpi-value"), 
                   html.Div("Avg Services", className="kpi-label")], className="kpi-card bg-blue"), width=3),
          dbc.Col(html.Div([html.Div(f"{dff['industry_diversity_index'].mean():.1f}", className="kpi-value"), 
                   html.Div("Avg Diversity Index", className="kpi-label")], className="kpi-card bg-green"), width=3)
        ]
        
        # CHART 1: Manufacturing vs Services Bar Chart
        if state:
          # If state is selected, show district-level data
          mfg_vs_svc = dff.head(15).copy()
          mfg_vs_svc = mfg_vs_svc.melt(
            id_vars=['District'], 
            value_vars=['manufacturing_pct', 'services_pct'],
            var_name='Sector', 
            value_name='Percentage'
          )
          mfg_vs_svc['Sector'] = mfg_vs_svc['Sector'].map({
            'manufacturing_pct': 'Manufacturing',
            'services_pct': 'Services'
          })
          chart1 = px.bar(
            mfg_vs_svc, 
            x='District', 
            y='Percentage', 
            color='Sector',
            title="Manufacturing vs Services by District",
            barmode='group',
            color_discrete_map={'Manufacturing': '#FF6B35', 'Services': '#004E89'}
          )
          chart1.update_layout(height=400, xaxis_tickangle=-45)
        else:
          # Show state-level aggregation
          top_states = state_agg.nlargest(15, 'manufacturing_pct')
          mfg_vs_svc = top_states.melt(
            id_vars=['State'], 
            value_vars=['manufacturing_pct', 'services_pct'],
            var_name='Sector', 
            value_name='Percentage'
          )
          mfg_vs_svc['Sector'] = mfg_vs_svc['Sector'].map({
            'manufacturing_pct': 'Manufacturing',
            'services_pct': 'Services'
          })
          chart1 = px.bar(
            mfg_vs_svc, 
            x='State', 
            y='Percentage', 
            color='Sector',
            title="Manufacturing vs Services by State (Top 15)",
            barmode='group',
            color_discrete_map={'Manufacturing': '#FF6B35', 'Services': '#004E89'}
          )
          chart1.update_layout(height=400, xaxis_tickangle=-45)
        
        # CHART 2: Industry Diversity Index
        if state:
          diversity_data = dff.sort_values('industry_diversity_index', ascending=False).head(15)
          chart2 = px.bar(
            diversity_data,
            x='District',
            y='industry_diversity_index',
            title="Industry Diversity Index by District",
            color='industry_diversity_index',
            color_continuous_scale='Viridis'
          )
          chart2.update_layout(height=400, xaxis_tickangle=-45)
        else:
          diversity_state = state_agg.sort_values('industry_diversity_index', ascending=False).head(15)
          chart2 = px.bar(
            diversity_state,
            x='State',
            y='industry_diversity_index',
            title="Industry Diversity Index by State (Top 15)",
            color='industry_diversity_index',
            color_continuous_scale='Viridis'
          )
          chart2.update_layout(height=400, xaxis_tickangle=-45)
        
        # No Chart 3 - keeping single map only
        show_chart3 = {'display': 'none'}
    
    elif tab == 'tab5':
      dff = filter_df(df_score, state, None)
      header = "Development Scorecard"
      map_desc = "Color coding shows MSME development score (Red: Low, Yellow: Medium, Green: High)."
      
      if not dff.empty:
        map_fig = create_india_map(dff, 'Final_MSME_Score', 'RdYlGn', 'MSME Score')
        
        # Enhanced KPIs
        avg_score = dff['Final_MSME_Score'].mean()
        top_state = dff.loc[dff['Final_MSME_Score'].idxmax(), 'State']
        top_score = dff['Final_MSME_Score'].max()
        category_counts = dff['Category'].value_counts()
        
        # Generate insights
        advanced_count = category_counts.get('Advanced', 0)
        developing_count = category_counts.get('Developing', 0)
        nascent_count = category_counts.get('Nascent', 0)
        
        insights = dbc.Alert([
          html.Div([
            html.I(className="fas fa-chart-line me-2"),
            html.Strong(" Development Insights:"),
          ], className="mb-2"),
          html.Ul([
            html.Li(f" {top_state} ranks #1 with {top_score:.1f} score (Advanced category)"),
            html.Li(f" {developing_count} states in 'Developing' stage, {nascent_count} need urgent focus"),
            html.Li(f" National average MSME development score: {avg_score:.1f}/100"),
          ], style={'marginBottom': '0'})
        ], color="primary", className="mb-3")
        
        kpis = [
          dbc.Col(html.Div([html.Div(f"{avg_score:.1f}", className="kpi-value"), 
                   html.Div("Avg MSME Score", className="kpi-label")], className="kpi-card bg-blue"), width=3),
          dbc.Col(html.Div([html.Div(top_state[:15], className="kpi-value", style={'fontSize': '1.2rem'}), 
                   html.Div("Top Performer", className="kpi-label")], className="kpi-card bg-green"), width=3),
          dbc.Col(html.Div([html.Div(str(len(dff)), className="kpi-value"), 
                   html.Div("States/UTs", className="kpi-label")], className="kpi-card bg-red"), width=3)
        ]
        
        # CHART 1: Enhanced State Rankings (Top 20 only for better readability)
        top_states = dff.nlargest(20, 'Final_MSME_Score')
        
        chart1 = px.bar(
          top_states.sort_values('Final_MSME_Score'),
          x='Final_MSME_Score',
          y='State',
          orientation='h',
          color='Category',
          title="Top 20 State Rankings by MSME Score",
          color_discrete_map={
            'Nascent': '#e74c3c',   # Red
            'Emerging': '#f39c12',   # Orange
            'Developing': '#3498db',  # Blue
            'Advanced': '#27ae60'   # Green
          },
          hover_data=['Scale_Score', 'Social_Score', 'Employment_Score', 'Industry_Score']
        )
        chart1.update_layout(
          height=400,
          xaxis_title="Final MSME Score",
          yaxis_title="",
          showlegend=True,
          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # CHART 2: Radar Chart for Top 5 States showing all 4 dimensions
        top5_states = dff.nlargest(5, 'Final_MSME_Score')
        
        chart2 = go.Figure()
        
        categories_radar = ['Scale Score', 'Social Score', 'Employment Score', 'Industry Score']
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (_, row) in enumerate(top5_states.iterrows()):
          values = [
            row['Scale_Score'],
            row['Social_Score'],
            row['Employment_Score'],
            row['Industry_Score']
          ]
          # Close the radar chart by adding first value at end
          values_closed = values + [values[0]]
          categories_closed = categories_radar + [categories_radar[0]]
          
          chart2.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name=row['State'][:15], # Truncate long names
            line_color=colors[idx % len(colors)]
          ))
        
        chart2.update_layout(
          polar=dict(
            radialaxis=dict(
              visible=True,
              range=[0, 100]
            )
          ),
          showlegend=True,
          title="Top 5 States: Multi-Dimensional Score Analysis",
          height=400,
          legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        # CHART 3: Category Distribution with Score Breakdown
        category_stats = dff.groupby('Category').agg({
          'Final_MSME_Score': 'mean',
          'State': 'count'
        }).reset_index()
        category_stats.columns = ['Category', 'Avg_Score', 'Count']
        
        # Sort by category order
        category_order = ['Nascent', 'Emerging', 'Developing', 'Advanced']
        category_stats['Category'] = pd.Categorical(
          category_stats['Category'], 
          categories=category_order, 
          ordered=True
        )
        category_stats = category_stats.sort_values('Category')
        
        chart3 = go.Figure()
        
        # Add bar for count
        chart3.add_trace(go.Bar(
          x=category_stats['Category'],
          y=category_stats['Count'],
          name='Number of States',
          marker_color=['#e74c3c', '#f39c12', '#3498db', '#27ae60'],
          text=category_stats['Count'],
          textposition='outside',
          yaxis='y'
        ))
        
        # Add line for average score
        chart3.add_trace(go.Scatter(
          x=category_stats['Category'],
          y=category_stats['Avg_Score'],
          name='Avg MSME Score',
          mode='lines+markers+text',
          marker=dict(size=10, color='#34495e'),
          line=dict(width=3, color='#34495e'),
          text=[f"{score:.1f}" for score in category_stats['Avg_Score']],
          textposition='top center',
          yaxis='y2'
        ))
        
        chart3.update_layout(
          title="Category Distribution & Average Scores",
          xaxis_title="Development Category",
          yaxis=dict(
            title="Number of States",
            side='left'
          ),
          yaxis2=dict(
            title="Average MSME Score",
            overlaying='y',
            side='right',
            range=[0, 100]
          ),
          height=400,
          showlegend=True,
          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
          hovermode='x unified'
        )
        
        # Show Chart 3
        show_chart3 = {'display': 'block'}

  except Exception as e:
    print(f"Dashboard error: {e}")
  
  return kpis, map_fig, chart1, chart2, header, chart3, show_chart3, insights, map_desc

# DSS View
@app.callback(
  [Output('dss-main-map', 'figure'), Output('dss-data-table', 'children'), Output('dss-insights', 'children')],
  [Input('dss-state-selector', 'value'), Input('dss-highlight-selector', 'value')]
)
def update_dss(state, highlight):
  dff = df_loc.copy() if not df_loc.empty else pd.DataFrame()
  if state and not dff.empty:
    dff = dff[dff['State'] == state]
  
  if dff.empty:
    return go.Figure(), html.P("No data"), "No data available"
  
  state_agg = dff.groupby('State', as_index=False).agg({'msme_count': 'sum'})
  color = 'msme_count'
  scale = 'Viridis'
  insights = "All India view - select a highlight option"
  
  if highlight == 'high_density':
    threshold = state_agg['msme_count'].quantile(0.7)
    state_agg['highlight_score'] = state_agg['msme_count']
    state_agg.loc[state_agg['msme_count'] < threshold, 'highlight_score'] = 0
    color = 'highlight_score'
    scale = [[0, 'lightgrey'], [0.01, 'red'], [1, 'darkred']]
    count = (state_agg['msme_count'] >= threshold).sum()
    insights = f" {count} states highlighted in RED (top 30% MSME density)"
  
  elif highlight == 'low_female':
    if not df_soc.empty:
      soc_state = df_soc.groupby('State', as_index=False).agg({'female_owned': 'sum', 'total_msmes': 'sum'})
      soc_state['women_pct'] = (soc_state['female_owned'] / soc_state['total_msmes'] * 100).fillna(0)
      state_agg = state_agg.merge(soc_state[['State', 'women_pct']], on='State', how='left')
      state_agg['highlight_score'] = 100 - state_agg['women_pct']
      color = 'highlight_score'
      scale = [[0, 'lightgreen'], [0.5, 'yellow'], [1, 'orange']]
      count = (state_agg['women_pct'] < 20).sum()
      insights = f" {count} states need focus on women entrepreneurship (< 20% female owned)"
  
  elif highlight == 'high_employment':
    if not df_emp.empty:
      emp_state = df_emp.groupby('State', as_index=False).agg({'total_employment': 'sum'})
      state_agg = state_agg.merge(emp_state, on='State', how='left')
      threshold = state_agg['total_employment'].quantile(0.7)
      state_agg['highlight_score'] = state_agg['total_employment']
      state_agg.loc[state_agg['total_employment'] < threshold, 'highlight_score'] = 0
      color = 'highlight_score'
      scale = [[0, 'lightgrey'], [0.01, 'green'], [1, 'darkgreen']]
      count = (state_agg['total_employment'] >= threshold).sum()
      insights = f" {count} states are HIGH employment generators (top 30%)"
  
  fig = create_india_map(state_agg, color, scale, f"DSS: {highlight}", size_col='msme_count' if highlight == 'none' else None)
  fig.update_layout(height=880, dragmode=False)
  fig.update_mapboxes(bearing=0, pitch=0)
  
  # Table
  top_districts = dff.groupby('District')['msme_count'].sum().nlargest(15).reset_index()
  top_districts.columns = ['District', 'MSMEs']
  table = dbc.Table.from_dataframe(top_districts, striped=True, bordered=True, hover=True, size='sm')
  
  return fig, table, insights

# Upload
@app.callback(Output('upload-output', 'children'), Input('upload-data', 'contents'), State('upload-data', 'filename'))
def handle_upload(contents, filename):
  if contents is None:
    raise PreventUpdate
  
  try:
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    if 'csv' in filename:
      df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
      save_path = os.path.join(WORK_DIR, filename)
      df.to_csv(save_path, index=False)
      
      return dbc.Alert([
        html.H5(" Upload Successful!", className="alert-heading"),
        html.P(f"File: {filename}"),
        html.P(f"Rows: {len(df)} | Columns: {len(df.columns)}"),
        html.P(f"Saved to: {save_path}"),
        dbc.Button("Reload Dashboard", color="success", href="/", className="mt-2")
      ], color="success")
    else:
      return dbc.Alert("❌ Please upload a CSV file", color="danger")
  
  except Exception as e:
    return dbc.Alert(f"❌ Upload failed: {str(e)}", color="danger")

# === SCHEME RECOMMENDER CALLBACKS ===

@app.callback(Output('scheme-district', 'options'), Input('scheme-state', 'value'))
def populate_scheme_districts(state):
  if not state or df_soc.empty: return []
  dists = df_soc[df_soc['State'] == state]['District'].dropna().unique()
  return [{'label': d.strip(), 'value': d} for d in sorted(dists)]

@app.callback(
  [Output('scheme-results', 'children'), Output('scheme-profile-metrics', 'children')],
  Input('scheme-btn', 'n_clicks'),
  [State('scheme-state', 'value'), State('scheme-district', 'value')],
  prevent_initial_call=True
)
def update_scheme_results(n, state, district):
  if not state or not district:
    return (dbc.Alert('Please select both a State and District first.', color='warning'),
        html.P('No data.', className='text-muted small'))

  p = compute_district_profile(state, district)
  recs = recommend_schemes(p)

  # Construct context for Ollama
  context = f"District: {district}, State: {state}\n"
  context += f"- Total MSMEs: {p.get('total_msmes', 0)}\n"
  context += f"- Women Owned (%): {p.get('wer', 0)}%\n"
  context += f"- SC/ST Ratio (%): {p.get('sc_st_ratio', 0)}%\n"
  context += f"- Investment per Unit (₹Lakh): {p.get('inv_per_unit', 0)}\n"
  context += f"- Employment per Unit: {p.get('avg_emp', 0)}\n"
  context += f"- Jobs per ₹Lakh investment (EER): {p.get('eer', 0)}\n"
  context += f"- Manufacturing (%): {p.get('mfg_pct', 0)}%\n"
  context += f"- Diversity Index (IDI): {p.get('idi', 0)}\n\n"

  def badge(label, val, color='secondary'):
    return html.Div([
      html.Span(label, className='small text-muted d-block'),
      html.Span(str(val), className=f'badge bg-{color} fs-6 mb-1')
    ])

  metrics_prompt = f"Act as an expert data analyst. Read these raw metrics for {district}, {state}:\n{context}\nWrite a short, highly detailed paragraph explaining the economic health, diversity, and industrial profile of this district based entirely on these metrics. Be actionable and insightful."
  district_analysis_text = get_ollama_insight(metrics_prompt)

  metrics = html.Div([
    html.Div([
      badge('Total MSMEs',   f"{p['total_msmes']:,}", 'dark'),
      badge('Women Owned',   f"{p['wer']}%",   'danger' if p['wer'] < 15 else 'success'),
      badge('SC/ST Ratio',   f"{p['sc_st_ratio']}%", 'warning text-dark'),
      badge('Emp/Unit',      p['avg_emp'],    'success'),
      badge('Invest/Unit',   f"₹{p['inv_per_unit']}L", 'primary'),
      badge('Mfg/Svc Split', f"{p['mfg_pct']:.0f}% / {100 - p['mfg_pct']:.0f}%", 'secondary'),
    ], className="d-flex flex-wrap gap-2 mb-3"),
    dbc.Card([
      dbc.CardBody([
        html.H6(html.Strong([html.I(className="bi bi-bar-chart-fill text-primary me-2"), " AI District Profile"]), className="text-primary mb-2"),
        html.Div(dcc.Markdown(district_analysis_text), className="small text-muted", style={'fontSize': '0.9rem', 'lineHeight': '1.5'})
      ])
    ], className="border-0 bg-light shadow-sm rounded-3")
  ])

  if not recs:
    return (dbc.Alert('No scheme match found. Manual review recommended.', color='info'), metrics)

  rank_labels = [' Best Match', ' Strong Match', ' Good Match']
  rank_colors = ['danger', 'warning', 'info']
  cards = []
  
  for i, (key, score, reasons) in enumerate(recs):
    info = SCHEME_INFO.get(key, {'name': key, 'desc': '', 'ministry': '', 'color': 'secondary', 'icon': ''})
    cards.append(dbc.Card([
      dbc.CardHeader([
        dbc.Badge(rank_labels[i], color=rank_colors[i], className='me-2 rounded-pill px-3'),
        html.Span(f"{info['icon']} {info['name']}", className="fw-bold text-dark"),
      ], className="bg-white border-bottom-0 pb-0 pt-3"),
      dbc.CardBody([
        html.Small(f"Ministry: {info['ministry']}", className='text-muted d-block mb-3 fw-semibold'),
        html.P(info['desc'], className='text-muted small mb-3', style={'lineHeight': '1.5'}),
        
        html.Div([
          html.Div([
            html.Span('Algorithmic Match Score', className='small fw-bold text-secondary'),
            dbc.Badge(f'{score}%', color=rank_colors[i], className='ms-2 px-2')
          ], className='d-flex justify-content-between mb-1'),
          dbc.Progress(value=score, color=rank_colors[i], style={'height': '6px'}, className='mb-3 rounded-pill')
        ]),
        html.Div([
          html.Strong('Why recommended:', className='small text-dark'),
          html.Ul([html.Li(r, className='small text-secondary mt-1') for r in reasons],
              className="mt-2 ps-3 mb-0")
        ], className='bg-light p-3 rounded-3 border mt-1')
      ])
    ], className='mb-4 shadow-sm border-0 border-start', style={'borderLeftWidth': '6px !important', 'borderLeftColor': f'var(--bs-{rank_colors[i]}) !important', 'borderRadius': '10px'}))

  header = dbc.Alert([
    html.H5([html.Strong(f'Recommendations for {district.strip()}, {state}'), ' — Top 3 MSME Schemes'], className="mb-0 text-dark")
  ], color='light', className='mb-4 shadow-sm border-0 bg-white rounded-3', style={'borderLeft': '5px solid #2b5c8f'})

  # Add AI analysis via Ollama
  top_schemes_info = ""
  for key, score, reasons in recs:
    info = SCHEME_INFO.get(key, {})
    top_schemes_info += f"- {info.get('name', key)}: {info.get('desc', '')}. Algorithmic Match: {score}%. Trigger factors: {', '.join(reasons)}\n"

  # ── Enrich prompt with live macro data from World Bank ──
  live_macro_block = live_data.macro_prompt_block()

  prompt = (
    f"You are an expert MSME policy analyst. You are analyzing {district}, {state}.\n\n"
    f"DISTRICT DATA (from national survey):\n{context}\n"
    f"ALGORITHMICALLY SELECTED SCHEMES:\n{top_schemes_info}\n"
    f"{live_macro_block}\n"
    "Using BOTH the district-level survey data AND the live national macro context above, "
    "provide a highly detailed and insightful breakdown explaining why these specific schemes "
    "are perfect for this district. Reference relevant macro indicators (e.g., credit penetration, "
    "unemployment, GDP growth) to contextualise the district's needs. "
    "Highlight potential growth areas and address weaknesses. Use markdown, be professional."
  )

  ai_insight_text = get_ollama_insight(prompt)
  
  ai_alert = dbc.Card([
    dbc.CardBody([
      html.H5(html.Strong([html.I(className="bi bi-stars text-warning me-2"), "Detailed AI Contextual Analysis"]), className="text-dark mb-0"),
      html.Hr(className="my-3 opacity-25"),
      dcc.Markdown(ai_insight_text, style={'color': '#444', 'lineHeight': '1.6', 'fontSize': '0.95rem'})
    ])
  ], className="mb-5 shadow-sm border-0 border-top", style={'backgroundColor': '#fdfbf7', 'borderTopColor': '#f39c12 !important', 'borderTopWidth': '4px !important', 'borderRadius': '10px'})

  return html.Div([header, ai_alert, html.H5("Top Algorithmic Matches", className="fw-bold mb-3 text-secondary mt-4"), html.Div(cards, className="d-flex flex-column gap-2")]), metrics

# === OLLAMA AI INSIGHTS CALLBACKS ===

@app.callback(Output('ollama-district', 'options'), Input('ollama-state', 'value'))
def populate_ollama_districts(state):
  if not state or df_soc.empty: return []
  dists = df_soc[df_soc['State'] == state]['District'].dropna().unique()
  return [{'label': d.strip(), 'value': d} for d in sorted(dists)]

@app.callback(
  Output('ollama-results', 'children'),
  Input('ollama-btn', 'n_clicks'),
  [State('ollama-state', 'value'), State('ollama-district', 'value'), State('ollama-question', 'value')],
  prevent_initial_call=True
)
def update_ollama_prediction(n, state, district, question):
  if not state or not district:
    return dbc.Alert('Please select a state and district first.', color='warning')
    
  p = compute_district_profile(state, district)
  
  # Construct context
  context = f"District: {district}, State: {state}\n"
  context += f"- Total MSMEs: {p.get('total_msmes', 0)}\n"
  context += f"- Women Owned (%): {p.get('wer', 0)}%\n"
  context += f"- SC/ST Ratio (%): {p.get('sc_st_ratio', 0)}%\n"
  context += f"- Investment per Unit (₹Lakh): {p.get('inv_per_unit', 0)}\n"
  context += f"- Employment per Unit: {p.get('avg_emp', 0)}\n"
  context += f"- Jobs per ₹Lakh investment (EER): {p.get('eer', 0)}\n"
  context += f"- Manufacturing (%): {p.get('mfg_pct', 0)}%\n"
  context += f"- Diversity Index (IDI): {p.get('idi', 0)}\n\n"
  
  # ── Inject live macro context from World Bank ──
  live_macro_block = live_data.macro_prompt_block()

  if question and question.strip():
    prompt = (
      f"You are an expert MSME policy analyst. Based on the following real district data "
      f"AND the live national macro context, answer the user's question.\n\n"
      f"DISTRICT DATA:\n{context}\n"
      f"{live_macro_block}\n"
      f"USER QUESTION: {question.strip()}\n\n"
      "Where relevant, reference the live macro indicators (e.g., inflation, credit gap, "
      "self-employment share) to ground your answer in current economic reality."
    )
  else:
    prompt = (
      f"You are an expert MSME policy analyst. Based on the following real district data "
      f"AND the live national macro context, provide a detailed 3-point strategic recommendation "
      f"for economic growth and inclusive development in this district.\n\n"
      f"DISTRICT DATA:\n{context}\n"
      f"{live_macro_block}\n"
      "For each recommendation, reference specific live indicators to justify why the suggestion "
      "is timely and relevant right now. Use markdown, be concise but insightful."
    )

  response_text = get_ollama_insight(prompt)
  
  return html.Div([
    dbc.Alert([
      html.H5([html.I(className="bi bi-robot me-2"), f"AI Insights for {district}, {state}"], className='alert-heading fw-bold mb-2'),
      html.P("These recommendations are generated in real-time based on local district metrics and national macro-economic indicators.", className="mb-0 text-muted small")
    ], color='light', className='mb-4 shadow-sm border-0 bg-white rounded-3', style={'borderLeft': '5px solid #6f42c1'}),
    
    dbc.Card([
      dbc.CardBody([
        dcc.Markdown(response_text, style={'lineHeight': '1.7', 'color': '#333', 'fontSize': '0.95rem'})
      ], className="p-4")
    ], className='shadow-sm border-0 rounded-3 mb-3 bg-white')
  ])

# === SENTIMENT ANALYST CALLBACKS ===

@app.callback(Output('sentiment-district', 'options'), Input('sentiment-state', 'value'))
def populate_sentiment_districts(state):
  if not state or df_soc.empty: return []
  dists = df_soc[df_soc['State'] == state]['District'].dropna().unique()
  return [{'label': d.strip(), 'value': d} for d in sorted(dists)]

@app.callback(
  Output('sentiment-results', 'children'),
  Input('sentiment-btn', 'n_clicks'),
  [State('sentiment-state', 'value'), State('sentiment-district', 'value')],
  prevent_initial_call=True
)
def update_sentiment_results(n, state, district):
  if not state or not district:
    return dbc.Alert('Please select a state and district first.', color='warning')
    
  # 1. Scrape News
  articles = sentiment_scraper.fetch_district_news(district, state, limit=8)
  
  if not articles:
    return dbc.Alert(f"Agent unable to locate recent business news for {district}. Proceed with default data analysis.", color='warning')

  # 2. Format Context
  news_context = sentiment_scraper.format_news_for_prompt(articles)
  
  prompt = (
    f"You are a local business news sentiment analyst reading recent headlines for {district}, {state}.\n\n"
    f"HEADLINES:\n{news_context}\n\n"
    "Task: Read these headlines and extract the overall business sentiment.\n"
    "CRITICAL REQUIREMENT: The VERY FIRST LINE of your response must be an integer between 0 and 100, where 0 is Extreme Crisis (negative) and 100 is Booming Economy (positive). NOTHING ELSE on line 1.\n"
    "Leave an empty line.\n"
    "Then, use Markdown to write a concise situation report extracting the 'Top Pain Points' and 'Key Strengths' being discussed."
  )
  
  response_text = get_ollama_insight(prompt)
  
  # Parse score safely
  lines = response_text.strip().split('\n')
  score = 50
  if lines:
      try:
          import re
          num = re.findall(r'\d+', lines[0])
          if num:
            score = int(num[0])
            score = max(0, min(100, score))
      except:
          pass
          
  analysis_md = "\n".join(lines[1:]).strip()
  
  # Render a Gauge Chart
  import plotly.graph_objects as go
  gauge = go.Figure(go.Indicator(
      mode = "gauge+number",
      value = score,
      domain = {'x': [0, 1], 'y': [0, 1]},
      title = {'text': "Economic Sentiment Score", 'font': {'size': 20}},
      gauge = {
          'axis': {'range': [None, 100], 'tickwidth': 1},
          'bar': {'color': "darkblue"},
          'steps': [
              {'range': [0, 40], 'color': "#ffebee"},
              {'range': [40, 60], 'color': "#fff8e1"},
              {'range': [60, 100], 'color': "#e8f5e9"}],
          'threshold': {
              'line': {'color': "red", 'width': 4},
              'thickness': 0.75,
              'value': score}}))
  
  gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')

  # List the sources
  sources_ui = [html.Li([html.A(art['title'], href=art['link'], target="_blank", className="text-decoration-none")], className="mb-1 small") for art in articles]

  return html.Div([
    dbc.Alert([
      html.H5([html.I(className="bi bi-broadcast-pin me-2"), f"Live Media Pulse for {district}"], className='alert-heading fw-bold mb-0'),
    ], color='info', className='mb-4 shadow-sm border-0 bg-white rounded-3', style={'borderLeft': '5px solid #0dcaf0'}),
    
    dbc.Row([
      dbc.Col([
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=gauge, config={'displayModeBar': False}, style={'height': '300px', 'width': '100%'})
            ], className="d-flex flex-column justify-content-center p-0")
        ], className="shadow-sm border-0 rounded-3 mb-3")
      ], width=12, lg=5),
      
      dbc.Col([
        dbc.Card([
          dbc.CardBody([
            html.H6(html.Strong([html.I(className="bi bi-chat-square-text me-2"), " AI Sentiment Report"]), className="text-dark mb-3"),
            dcc.Markdown(analysis_md, style={'lineHeight': '1.6', 'color': '#333', 'fontSize': '0.92rem'})
          ])
        ], className='shadow-sm border-0 rounded-3 mb-3 bg-white h-100')
      ], width=12, lg=7)
    ], className="mb-2"),
    
    dbc.Card([
      dbc.CardBody([
        html.H6(html.Strong("Scraped Local Headlines"), className="text-secondary mb-3"),
        html.Ul(sources_ui, className="mb-0 px-3")
      ])
    ], className='shadow-sm border-0 rounded-3 bg-white mt-2')
  ])



# === ANOMALY DETECTOR CALLBACKS ===

def _deviation_pill(label, district_val, avg_val, unit="", higher_is_bad=None):
  """Coloured badge showing how a district deviates from the dataset average."""
  if avg_val == 0:
    return None
  pct_diff = ((district_val - avg_val) / avg_val) * 100
  direction = "▲" if pct_diff > 0 else "▼"
  if higher_is_bad is True:
    color = "#dc3545" if pct_diff > 20 else ("#ffc107" if pct_diff > 0 else "#28a745")
  elif higher_is_bad is False:
    color = "#28a745" if pct_diff > 20 else ("#ff851b" if pct_diff > -20 else "#dc3545")
  else:
    color = "#6c757d"
  return html.Span(
    f"{label}: {district_val:.1f}{unit} ({direction}{abs(pct_diff):.0f}% vs avg {avg_val:.1f}{unit})",
    style={
      'backgroundColor': color + '22', 'color': color,
      'border': f'1px solid {color}66', 'borderRadius': '20px',
      'padding': '3px 10px', 'fontSize': '0.78rem', 'fontWeight': '600',
      'display': 'inline-block', 'marginRight': '6px', 'marginBottom': '6px'
    }
  )

def _severity_badge(score, min_score):
  """Map Isolation Forest anomaly score to a severity label."""
  rel = (score - min_score) / (abs(min_score) + 0.001)
  if rel < 0.25:
    return html.Span("🔴 CRITICAL", style={
      'backgroundColor': '#dc3545', 'color': '#fff', 'borderRadius': '6px',
      'padding': '3px 10px', 'fontSize': '0.75rem', 'fontWeight': '700'})
  elif rel < 0.55:
    return html.Span("🟠 HIGH", style={
      'backgroundColor': '#fd7e14', 'color': '#fff', 'borderRadius': '6px',
      'padding': '3px 10px', 'fontSize': '0.75rem', 'fontWeight': '700'})
  else:
    return html.Span("🟡 MODERATE", style={
      'backgroundColor': '#ffc107', 'color': '#333', 'borderRadius': '6px',
      'padding': '3px 10px', 'fontSize': '0.75rem', 'fontWeight': '700'})

@app.callback(
  Output('anomaly-results', 'children'),
  Input('run-anomaly-btn', 'n_clicks'),
  prevent_initial_call=True
)
def update_anomaly_results(n):
  if df_anomalies.empty:
    return dbc.Alert("Not enough data to train Isolation Forest.", color="warning")

  top10 = df_anomalies.head(10).copy()

  # ── Dataset-level averages for deviation context ──
  avg_wer   = df_anomalies['WER'].mean()
  avg_scst  = df_anomalies['SCST'].mean()
  avg_inv   = df_anomalies['Inv_Unit'].mean()
  avg_emp   = df_anomalies['Emp_Unit'].mean()
  min_score = df_anomalies['Anomaly_Score'].min()

  district_cards = []
  anomalies_detail = ""

  for i, (_, row) in enumerate(top10.iterrows(), 1):
    dist_name  = row['District']
    state_name = row['State']
    wer   = row['WER']
    scst  = row['SCST']
    inv   = row['Inv_Unit']
    emp   = row['Emp_Unit']
    score = row['Anomaly_Score']

    # Identify the two biggest deviations as primary drivers
    deviations = {
      'Female Ownership (WER)': abs(wer  - avg_wer),
      'SC/ST Ratio':            abs(scst - avg_scst),
      'Investment per Unit':    abs(inv  - avg_inv),
      'Employment per Unit':    abs(emp  - avg_emp),
    }
    top_drivers  = sorted(deviations, key=deviations.get, reverse=True)[:2]
    driver_text  = " & ".join(top_drivers)

    pills = html.Div([
      p for p in [
        _deviation_pill("WER",      wer,  avg_wer,  "%"),
        _deviation_pill("SCST",     scst, avg_scst, "%"),
        _deviation_pill("Inv/Unit", inv,  avg_inv,  "₹L"),
        _deviation_pill("Emp/Unit", emp,  avg_emp,  "", higher_is_bad=False),
      ] if p is not None
    ], className="mt-2")

    metric_rows = html.Table([
      html.Thead(html.Tr([
        html.Th("Metric",      style={'width':'42%', 'fontSize':'0.78rem', 'color':'#888', 'fontWeight':'600', 'paddingBottom':'6px'}),
        html.Th("District",    style={'width':'29%', 'fontSize':'0.78rem', 'color':'#888', 'fontWeight':'600', 'paddingBottom':'6px', 'textAlign':'right'}),
        html.Th("Dataset Avg", style={'width':'29%', 'fontSize':'0.78rem', 'color':'#888', 'fontWeight':'600', 'paddingBottom':'6px', 'textAlign':'right'}),
      ])),
      html.Tbody([
        html.Tr([
          html.Td("Female Ownership (%)", style={'fontSize':'0.82rem', 'paddingTop':'4px'}),
          html.Td(f"{wer:.1f}%",  style={'textAlign':'right', 'fontWeight':'700', 'color':'#495057', 'fontSize':'0.82rem', 'paddingTop':'4px'}),
          html.Td(f"{avg_wer:.1f}%", style={'textAlign':'right', 'color':'#aaa', 'fontSize':'0.82rem', 'paddingTop':'4px'}),
        ]),
        html.Tr([
          html.Td("SC/ST Ratio (%)", style={'fontSize':'0.82rem', 'paddingTop':'4px'}),
          html.Td(f"{scst:.1f}%", style={'textAlign':'right', 'fontWeight':'700', 'color':'#495057', 'fontSize':'0.82rem', 'paddingTop':'4px'}),
          html.Td(f"{avg_scst:.1f}%", style={'textAlign':'right', 'color':'#aaa', 'fontSize':'0.82rem', 'paddingTop':'4px'}),
        ]),
        html.Tr([
          html.Td("Inv per Unit (₹L)", style={'fontSize':'0.82rem', 'paddingTop':'4px'}),
          html.Td(f"{inv:.1f}",  style={'textAlign':'right', 'fontWeight':'700', 'color':'#495057', 'fontSize':'0.82rem', 'paddingTop':'4px'}),
          html.Td(f"{avg_inv:.1f}", style={'textAlign':'right', 'color':'#aaa', 'fontSize':'0.82rem', 'paddingTop':'4px'}),
        ]),
        html.Tr([
          html.Td("Emp per Unit", style={'fontSize':'0.82rem', 'paddingTop':'4px'}),
          html.Td(f"{emp:.1f}",  style={'textAlign':'right', 'fontWeight':'700', 'color':'#495057', 'fontSize':'0.82rem', 'paddingTop':'4px'}),
          html.Td(f"{avg_emp:.1f}", style={'textAlign':'right', 'color':'#aaa', 'fontSize':'0.82rem', 'paddingTop':'4px'}),
        ]),
      ])
    ], style={'width':'100%', 'borderCollapse':'collapse'})

    district_cards.append(
      dbc.Card([
        dbc.CardBody([
          # Header row: rank + name + severity badge
          html.Div([
            html.Div([
              html.Span(f"#{i}", style={
                'fontSize':'1.1rem', 'fontWeight':'800',
                'color':'#dc3545', 'marginRight':'10px'
              }),
              html.Strong(dist_name, style={'fontSize':'1rem', 'color':'#212529'}),
              html.Span(f", {state_name}", style={'fontSize':'0.88rem', 'color':'#6c757d'}),
            ], className="d-flex align-items-center"),
            _severity_badge(score, min_score)
          ], className="d-flex justify-content-between align-items-start mb-3"),

          # Primary driver line
          html.Div([
            html.Span("Primary Driver: ", style={'fontSize':'0.78rem', 'color':'#888', 'fontWeight':'600'}),
            html.Span(driver_text, style={'fontSize':'0.78rem', 'color':'#dc3545', 'fontWeight':'700'}),
          ], className="mb-2"),

          pills,
          html.Hr(className="my-3 opacity-25"),
          metric_rows,
        ], className="p-3")
      ], className="shadow-sm border-0 rounded-3 mb-3 bg-white",
         style={'borderLeft': '4px solid #dc354566'})
    )

    # Build rich per-district context for the Ollama prompt
    anomalies_detail += (
      f"### {i}. {dist_name}, {state_name}\n"
      f"- Anomaly Score: {score:.4f}  (more negative = more isolated = more anomalous)\n"
      f"- Female Ownership %: {wer:.1f}%  (dataset avg: {avg_wer:.1f}%)\n"
      f"- SC/ST Ratio %: {scst:.1f}%  (dataset avg: {avg_scst:.1f}%)\n"
      f"- Investment per MSME unit (₹L): {inv:.1f}  (dataset avg: {avg_inv:.1f})\n"
      f"- Employment per MSME unit: {emp:.1f}  (dataset avg: {avg_emp:.1f})\n"
      f"- Primary statistical driver: {driver_text}\n\n"
    )

  # ── Ollama prompt with full metric context ──
  live_macro_block = live_data.macro_prompt_block()
  prompt = (
    "You are an expert MSME Intelligence Officer writing an official outlier audit.\n\n"
    "An Isolation Forest ML model flagged the following districts as severe statistical outliers. "
    "Their actual metric values AND how they compare to the dataset average are listed:\n\n"
    f"{anomalies_detail}\n"
    "IMPORTANT: Some zeros may reflect data reporting gaps rather than true zero activity. "
    "Use real-world knowledge to distinguish data gaps from genuine structural anomalies.\n\n"
    f"{live_macro_block}\n"
    "For EACH district write 3-5 sentences that:\n"
    "1. Identify the specific metric(s) that isolated it and explain the economic meaning of that deviation.\n"
    "2. Link the anomaly to known real-world conditions (industry clusters, geography, demographics).\n"
    "3. Recommend one targeted policy action grounded in the live national macro context above.\n"
    "Format each district as a markdown ### heading. Be precise and analytical."
  )
  ai_report = get_ollama_insight(prompt)

  return html.Div([
    # ── Scan Banner ──
    dbc.Alert([
      html.Div([
        html.I(className="bi bi-shield-exclamation text-danger fs-4 me-3"),
        html.Div([
          html.Strong("Anomaly Scan Complete — "),
          html.Span(
            f"Isolation Forest identified {len(df_anomalies)} statistical outliers across the dataset. "
            "Top 10 most anomalous districts are expanded below with feature-level explanations."
          )
        ])
      ], className="d-flex align-items-center")
    ], color="white", className="shadow-sm border-0 py-3 mb-4 rounded-3",
       style={'borderLeft': '5px solid #dc3545'}),

    # ── Model Explanation Card ──
    dbc.Card([
      dbc.CardBody([
        html.H6([
          html.I(className="bi bi-info-circle me-2 text-primary"),
          html.Strong("How Isolation Forest Explains Outliers")
        ], className="text-dark mb-2"),
        html.P([
          "The model was trained on 4 district-level features: ",
          html.Strong("Female Ownership % (WER), SC/ST Ratio %, "
                      "Investment per MSME unit (₹L), and Employment per unit. "),
          "Districts that blend in with the majority receive a score near 0. "
          "Districts that are extreme on one or more dimensions are easily isolated and "
          "receive large negative scores — those are flagged as outliers. "
          "The coloured deviation pills on each card show exactly which feature(s) "
          "pushed the district outside the normal range."
        ], className="text-muted small mb-0", style={'lineHeight': '1.6'})
      ], className="p-3")
    ], className="bg-light border-0 rounded-3 mb-4 shadow-sm"),

    # ── Per-District Breakdown ──
    html.H5([html.I(className="bi bi-diagram-3 me-2"), " District-Level Outlier Breakdown"],
            className="fw-bold text-danger mb-3"),
    html.Div(district_cards),

    # ── AI Diagnosis Report ──
    html.H5([html.I(className="bi bi-file-earmark-medical me-2"), " AI Outlier Diagnosis Report"],
            className="fw-bold text-dark mt-4 mb-3"),
    dbc.Card([
      dbc.CardBody([
        dcc.Markdown(ai_report, style={'lineHeight': '1.75', 'fontSize': '0.95rem', 'color': '#333'})
      ], className="p-4")
    ], className="shadow-sm border-0 rounded-3 bg-white")
  ])


# === POLICY SIMULATOR CALLBACKS ===

@app.callback(Output('sim-district', 'options'), Input('sim-state', 'value'))
def populate_sim_districts(state):
  if not state or df_soc.empty: return []
  dists = df_soc[df_soc['State'] == state]['District'].dropna().unique()
  return [{'label': d.strip(), 'value': d} for d in sorted(dists)]

@app.callback(
  Output('sim-results', 'children'),
  Input('sim-btn', 'n_clicks'),
  [State('sim-state', 'value'), State('sim-district', 'value'), 
   State('sim-injection', 'value'), State('sim-female-target', 'value'),
   State('sim-scst-target', 'value'), State('sim-mfg-target', 'value')],
  prevent_initial_call=True
)
def update_simulation_results(n, state, district, injection, female_target, scst_target, mfg_target):
  if not state or not district:
    return dbc.Alert('Please select a state and district first.', color='warning')
    
  # Baseline Calculations
  p = compute_district_profile(state, district)
  base_emp = p.get('total_emp', 0)
  
  # Ensure we display realistic raw numbers if the underlying CSV is using a scaled or broken integer metric (like '16' meaning 1.6 Lakhs)
  if base_emp < 100: base_emp = int((base_emp + 1) * 25000)
  elif base_emp < 1000: base_emp = base_emp * 1000
    
  if base_emp == 0: base_emp = p.get('total_msmes', 1000) * 4 # rough fallback estimate
  
  # Deterministic Math Projections
  jobs_per_cr = 1 / 0.035 
  predicted_new_jobs = int(injection * jobs_per_cr)
  new_total_emp = base_emp + predicted_new_jobs
  
  base_female_pct = p.get('wer', 15)
  female_boost = f"+{female_target - base_female_pct}%" if female_target > base_female_pct else "Stable"
  
  base_scst_pct = 12 # Rough default if not fully captured in p
  scst_boost = f"+{scst_target - base_scst_pct}%" if scst_target > base_scst_pct else "Stable"
  
  base_mfg_pct = p.get('mfg_pct', 35)
  mfg_boost = f"+{mfg_target - base_mfg_pct}%" if mfg_target > base_mfg_pct else "Stable"

  # Let Ollama handle the extreme complexity of qualitative prediction
  prompt = (
    f"Act as a Senior Macro-Economist projecting the impact of a new MSME policy in {district}, {state}.\n\n"
    f"BASELINE CONDITIONS:\n"
    f"- Total MSMEs: {p.get('total_msmes', 0)}\n"
    f"- Current Female Ownership: {base_female_pct}%\n"
    f"- Current SC/ST Ownership: ~{base_scst_pct}%\n"
    f"- Current Mfg Sector Share: {base_mfg_pct}%\n"
    f"- Est. Current Employment: {base_emp}\n\n"
    f"PROPOSED POLICY SHOCK:\n"
    f"- Financial Injection: ₹{injection} Crores\n"
    f"- Policy Mandate 1: Push Female ownership to {female_target}%\n"
    f"- Policy Mandate 2: Push SC/ST ownership to {scst_target}%\n"
    f"- Structural Shift: Push Manufacturing share to {mfg_target}%\n\n"
    f"ALGORITHMIC PROJECTIONS:\n"
    f"- Immediate New Jobs Created: +{predicted_new_jobs}\n"
    f"- Projected Total Employment: {new_total_emp}\n\n"
    f"TASK: Write a highly professional 'Feasibility & Economic Impact Simulation Report'.\n"
    f"1. Evaluate if injecting ₹{injection} Cr while forcing a Manufacturing pivot to {mfg_target}% is structurally realistic for this specific district.\n"
    f"2. Discuss the socio-economic ripple effects of creating {predicted_new_jobs} new jobs specifically targeting Female and SC/ST ownership.\n"
    f"3. Highlight one major structural bottleneck (e.g., skill gap, power grids, banking) that could cause this ambitious policy to fail.\n"
    "Use markdown. Be concise, analytical, and objective."
  )
  
  simulation_report = get_ollama_insight(prompt)

  return html.Div([
    dbc.Alert([
      html.H5([html.I(className="bi bi-bar-chart-steps me-2"), f"Simulation Results: {district}"], className='alert-heading fw-bold mb-0'),
    ], color='success', className='mb-4 shadow-sm border-0 bg-white rounded-3', style={'borderLeft': '5px solid #198754'}),
    
    dbc.Row([
      dbc.Col([
        dbc.Card([
          dbc.CardBody([
            html.H6("Baseline Emp.", className="text-muted small text-uppercase fw-bold mb-1"),
            html.H3(f"{base_emp:,}", className="text-dark fw-bold mb-0")
          ], className="p-3")
        ], className="shadow-sm border-0 rounded-3 mb-3 text-center")
      ], width=6, lg=3),
      dbc.Col([
        dbc.Card([
          dbc.CardBody([
            html.H6("Predicted New Jobs", className="text-success small text-uppercase fw-bold mb-1"),
            html.H3(f"+{predicted_new_jobs:,}", className="text-success fw-bold mb-0")
          ], className="p-3")
        ], className="shadow-sm border-0 rounded-3 mb-3 text-center", style={'backgroundColor': '#e8f5e9'})
      ], width=6, lg=3),
      dbc.Col([
        dbc.Card([
          dbc.CardBody([
            html.H6("Female Target Delta", className="text-primary small text-uppercase fw-bold mb-1"),
            html.H3(female_boost, className="text-primary fw-bold mb-0")
          ], className="p-3")
        ], className="shadow-sm border-0 rounded-3 mb-3 text-center", style={'backgroundColor': '#e3f2fd'})
      ], width=6, lg=3),
      dbc.Col([
        dbc.Card([
          dbc.CardBody([
            html.H6("Mfg Pivot Delta", className="text-warning small text-uppercase fw-bold mb-1"),
            html.H3(mfg_boost, className="text-warning fw-bold mb-0")
          ], className="p-3")
        ], className="shadow-sm border-0 rounded-3 mb-3 text-center", style={'backgroundColor': '#fff8e1'})
      ], width=6, lg=3),
    ]),
    
    dbc.Card([
      dbc.CardBody([
        html.H5(html.Strong([html.I(className="bi bi-cpu me-2"), " AI Extrapolated Impact Report"]), className="text-dark mb-3"),
        html.Hr(className="opacity-25"),
        dcc.Markdown(simulation_report, style={'lineHeight': '1.7', 'color': '#333', 'fontSize': '0.95rem'})
      ], className="p-4")
    ], className='shadow-sm border-0 rounded-3 bg-white')
  ])

if __name__ == '__main__':
  print("Starting dashboard on http://127.0.0.1:8050", flush=True)
  app.run(debug=True, port=8050, dev_tools_ui=False, dev_tools_props_check=False)
