import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main { background-color: #F8F9FB; }
  [data-testid="stSidebar"] { background: #0A1628; border-right: 1px solid #1E2D3D; }
  [data-testid="stSidebar"] * { color: #C8D6E5 !important; }
  .kpi-card { background: white; border-radius: 12px; padding: 20px 24px; border: 1px solid #E8ECF0; border-left: 4px solid #1A6BFF; margin-bottom: 4px; }
  .kpi-label { font-size: 12px; color: #6B7C93; font-weight: 500; text-transform: uppercase; letter-spacing: 0.8px; }
  .kpi-value { font-size: 28px; font-weight: 600; color: #0D1B2A; font-family: 'DM Serif Display', serif; margin: 4px 0; }
  .kpi-delta { font-size: 12px; color: #16A34A; font-weight: 500; }
  .kpi-delta.red { color: #DC2626; }
  .section-title { font-family: 'DM Serif Display', serif; font-size: 22px; color: #0D1B2A; margin: 0 0 4px 0; }
  .section-sub { font-size: 13px; color: #6B7C93; margin-bottom: 20px; }
  .hero-container { background: linear-gradient(135deg, #0A1628 0%, #0D2E5C 50%, #0A1E3D 100%); border-radius: 16px; padding: 48px 56px; color: white; margin-bottom: 32px; }
  .hero-title { font-family: 'DM Serif Display', serif; font-size: 44px; font-weight: 400; color: white; margin: 0 0 12px 0; line-height: 1.15; }
  .hero-sub { font-size: 16px; color: #93B4D4; max-width: 600px; line-height: 1.6; }
  .hero-badge { display: inline-block; background: rgba(26,107,255,0.25); border: 1px solid rgba(26,107,255,0.5); color: #7BB3FF; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; margin-bottom: 16px; }
  .insight-box { background: #EFF6FF; border-left: 4px solid #1A6BFF; border-radius: 0 8px 8px 0; padding: 14px 18px; font-size: 14px; color: #1E3A5F; margin: 12px 0; }
  .insight-box.orange { background: #FFF7ED; border-left-color: #F97316; color: #7C2D12; }
  .insight-box.green { background: #ECFDF5; border-left-color: #10B981; color: #064E3B; }
  .insight-box.red { background: #FEF2F2; border-left-color: #EF4444; color: #7F1D1D; }
  .pred-card { border-radius: 14px; padding: 32px 24px; text-align: center; }
  .pred-card.green { background: linear-gradient(135deg, #064E3B, #065F46); }
  .pred-card.red { background: linear-gradient(135deg, #7F1D1D, #991B1B); }
  .pred-val { font-family: 'DM Serif Display', serif; font-size: 42px; color: white; }
  .pred-label { font-size: 13px; color: rgba(255,255,255,0.7); text-transform: uppercase; letter-spacing: 0.8px; margin-top: 8px; }
  .pred-prob { font-size: 18px; color: rgba(255,255,255,0.9); margin-top: 8px; font-weight: 600; }
  #MainMenu {visibility:hidden;} footer {visibility:hidden;} .stDeployButton {display:none;}
  h1,h2,h3 { font-family: 'DM Serif Display', serif !important; font-weight: 400 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COORDONNÉES AÉROPORTS
# ─────────────────────────────────────────────
AIRPORTS = {
    'ATL': ('Atlanta', 33.6367, -84.4281), 'DFW': ('Dallas', 32.8998, -97.0403),
    'ORD': ('Chicago', 41.9742, -87.9073), 'DEN': ('Denver', 39.8561, -104.6737),
    'CLT': ('Charlotte', 35.2140, -80.9431), 'LAX': ('Los Angeles', 33.9425, -118.4081),
    'PHX': ('Phoenix', 33.4373, -112.0078), 'SEA': ('Seattle', 47.4502, -122.3088),
    'LAS': ('Las Vegas', 36.0840, -115.1537), 'MCO': ('Orlando', 28.4294, -81.3089),
    'DTW': ('Detroit', 42.2124, -83.3534), 'IAH': ('Houston', 29.9902, -95.3368),
    'SFO': ('San Francisco', 37.6213, -122.3790), 'LGA': ('New York LGA', 40.7772, -73.8726),
    'MSP': ('Minneapolis', 44.8848, -93.2223), 'SLC': ('Salt Lake City', 40.7884, -111.9778),
    'BOS': ('Boston', 42.3656, -71.0096), 'EWR': ('Newark', 40.6895, -74.1745),
    'JFK': ('New York JFK', 40.6413, -73.7781), 'DCA': ('Washington', 38.8521, -77.0377),
    'MIA': ('Miami', 25.7959, -80.2870), 'MDW': ('Chicago Midway', 41.7868, -87.7522),
    'FLL': ('Fort Lauderdale', 26.0726, -80.1527), 'TPA': ('Tampa', 27.9755, -82.5332),
    'BWI': ('Baltimore', 39.1754, -76.6683), 'IAD': ('Washington Dulles', 38.9531, -77.4565),
    'DAL': ('Dallas Love', 32.8471, -96.8517), 'HOU': ('Houston Hobby', 29.6454, -95.2789),
    'OAK': ('Oakland', 37.7213, -122.2208), 'SAN': ('San Diego', 32.7336, -117.1897),
    'PDX': ('Portland', 45.5898, -122.5951), 'STL': ('St. Louis', 38.7487, -90.3700),
    'MCI': ('Kansas City', 39.2976, -94.7139), 'RDU': ('Raleigh', 35.8776, -78.7875),
    'AUS': ('Austin', 30.1975, -97.6664), 'SMF': ('Sacramento', 38.6954, -121.5908),
    'OKC': ('Oklahoma City', 35.3931, -97.6007), 'MSY': ('New Orleans', 29.9934, -90.2580),
    'SJC': ('San Jose', 37.3626, -121.9290), 'HNL': ('Honolulu', 21.3187, -157.9224),
}

# ─────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, 'vols_sample.csv'))

    # Nettoyage
    cols_to_drop = ['CANCELLATION_CODE','DELAY_DUE_CARRIER','DELAY_DUE_WEATHER',
        'DELAY_DUE_NAS','DELAY_DUE_SECURITY','DELAY_DUE_LATE_AIRCRAFT',
        'AIRLINE_DOT','DOT_CODE','FL_NUMBER','WHEELS_OFF','WHEELS_ON',
        'ELAPSED_TIME','AIR_TIME','TAXI_OUT','TAXI_IN','DEP_TIME','ARR_TIME']
    df = df.drop(columns=cols_to_drop)
    df = df.dropna(subset=['ARR_DELAY','DEP_DELAY'])
    df = df[(df['ARR_DELAY']>=-60)&(df['ARR_DELAY']<=600)&
            (df['DEP_DELAY']>=-40)&(df['DEP_DELAY']<=600)]

    # Features
    df['RETARDE']     = (df['ARR_DELAY'] > 15).astype(int)
    df['FL_DATE']     = pd.to_datetime(df['FL_DATE'])
    df['MONTH']       = df['FL_DATE'].dt.month
    df['YEAR']        = df['FL_DATE'].dt.year
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
    df['HOUR_DEP']    = (df['CRS_DEP_TIME'] // 100).astype(int)
    df['IS_WEEKEND']  = (df['DAY_OF_WEEK'] >= 5).astype(int)

    def saison(m):
        if m in [12,1,2]: return 'Hiver'
        if m in [3,4,5]:  return 'Printemps'
        if m in [6,7,8]:  return 'Été'
        return 'Automne'
    df['SAISON'] = df['MONTH'].apply(saison)

    # Raccourcir noms compagnies
    df['AIRLINE_SHORT'] = df['AIRLINE'].str.replace(' Airlines','').str.replace(
        ' Air Lines','').str.replace(' Inc.','').str.replace(
        ' Co.','').str.replace(' Airways','').str.strip()

    return df

@st.cache_resource
def load_model():
    base = os.path.join(os.path.dirname(__file__), 'models')
    rf         = joblib.load(os.path.join(base, 'random_forest.pkl'))
    le_airline = joblib.load(os.path.join(base, 'le_airline.pkl'))
    le_origin  = joblib.load(os.path.join(base, 'le_origin.pkl'))
    le_dest    = joblib.load(os.path.join(base, 'le_dest.pkl'))
    with open(os.path.join(base, 'model_info.json')) as f:
        info = json.load(f)
    return rf, le_airline, le_origin, le_dest, info

df = load_data()
rf, le_airline, le_origin, le_dest, model_info = load_model()

MOIS = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
JOURS = ['Lun','Mar','Mer','Jeu','Ven','Sam','Dim']

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px 0;'>
      <div style='font-size:20px;font-weight:600;color:white;'>✈️ Flight Delay</div>
      <div style='font-size:11px;color:#4A6B8A;margin-top:4px;'>US Flights · 2019–2023</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("Navigation", [
        "Accueil",
        "Exploration",
        "Analyse Approfondie",
        "Prédiction ML"
    ], label_visibility="collapsed")
    st.markdown("<hr style='border-color:#1E2D3D;margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size:11px;color:#3A5470;line-height:1.8;'>
    Dataset : US Bureau of Transportation<br>
    50 000 vols · 18 compagnies<br>
    Modèle : Random Forest<br>
    Accuracy : 93% · AUC : 0.913
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ═══════════════════════════════════════════════
if "Accueil" in page:

    st.markdown("""
    <div class="hero-container">
      <div class="hero-badge">US BUREAU OF TRANSPORTATION · 50 000 VOLS · 2019–2023</div>
      <div class="hero-title">Flight Delay<br>Intelligence Platform</div>
      <div class="hero-sub">Explorez les retards aériens aux États-Unis, analysez les performances
      des compagnies et prédisez en temps réel si votre vol sera retardé grâce au Machine Learning.</div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total_vols    = len(df)
    taux_retard   = df['RETARDE'].mean() * 100
    retard_moyen  = df[df['ARR_DELAY'] > 0]['ARR_DELAY'].mean()
    nb_compagnies = df['AIRLINE'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Vols analysés</div>
          <div class="kpi-value">{total_vols:,}</div>
          <div class="kpi-delta">▲ 2019 – 2023</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#EF4444;">
          <div class="kpi-label">Taux de retard</div>
          <div class="kpi-value">{taux_retard:.1f}%</div>
          <div class="kpi-delta red">▲ Retard > 15 min</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#F59E0B;">
          <div class="kpi-label">Retard moyen</div>
          <div class="kpi-value">{retard_moyen:.0f} min</div>
          <div class="kpi-delta">▲ Sur vols en retard</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#10B981;">
          <div class="kpi-label">Compagnies</div>
          <div class="kpi-value">{nb_compagnies}</div>
          <div class="kpi-delta">▲ Compagnies US</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Carte des aéroports
    st.markdown('<p class="section-title">Carte des aéroports</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Taille des bulles proportionnelle au nombre de vols</p>', unsafe_allow_html=True)

    airport_counts = df['ORIGIN'].value_counts().reset_index()
    airport_counts.columns = ['code','count']
    airport_counts['lat']  = airport_counts['code'].map(lambda x: AIRPORTS.get(x, (None,None,None))[1])
    airport_counts['lon']  = airport_counts['code'].map(lambda x: AIRPORTS.get(x, (None,None,None))[2])
    airport_counts['name'] = airport_counts['code'].map(lambda x: AIRPORTS.get(x, (x,None,None))[0])
    airport_counts = airport_counts.dropna(subset=['lat','lon'])

    # Taux retard par aéroport
    retard_by_airport = df.groupby('ORIGIN')['RETARDE'].mean().reset_index()
    retard_by_airport.columns = ['code','taux_retard']
    airport_counts = airport_counts.merge(retard_by_airport, on='code', how='left')
    airport_counts['taux_pct'] = (airport_counts['taux_retard']*100).round(1)

    fig_map = go.Figure()
    fig_map.add_trace(go.Scattergeo(
        lat=airport_counts['lat'], lon=airport_counts['lon'],
        text=airport_counts.apply(lambda r: f"{r['code']} — {r['name']}<br>Vols: {r['count']:,}<br>Retards: {r['taux_pct']}%", axis=1),
        mode='markers',
        marker=dict(
            size=airport_counts['count']/airport_counts['count'].max()*40+8,
            color=airport_counts['taux_pct'],
            colorscale=[[0,'#2196F3'],[0.5,'#F59E0B'],[1,'#EF4444']],
            colorbar=dict(title='Taux retard %', thickness=12),
            line=dict(color='white', width=1), opacity=0.85
        ),
        hoverinfo='text'
    ))
    fig_map.update_layout(
        geo=dict(scope='usa', showland=True, landcolor='#F1F5F9',
                 showocean=True, oceancolor='#EFF6FF',
                 showlakes=True, lakecolor='#EFF6FF',
                 showcoastlines=True, coastlinecolor='#CBD5E1',
                 projection_type='albers usa'),
        height=420, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Évolution taux retard par année
    st.markdown('<p class="section-title">Évolution des retards 2019–2023</p>', unsafe_allow_html=True)
    year_stats = df.groupby('YEAR').agg(
        taux=('RETARDE','mean'), total=('RETARDE','count')
    ).reset_index()
    year_stats['taux_pct'] = (year_stats['taux']*100).round(1)

    fig_year = go.Figure()
    fig_year.add_trace(go.Bar(
        x=year_stats['YEAR'], y=year_stats['total'],
        name='Nombre de vols', marker_color='#BFDBFE', yaxis='y2', opacity=0.6
    ))
    fig_year.add_trace(go.Scatter(
        x=year_stats['YEAR'], y=year_stats['taux_pct'],
        name='Taux de retard (%)', line=dict(color='#EF4444', width=3),
        mode='lines+markers', marker=dict(size=8)
    ))
    fig_year.update_layout(
        height=300, margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
        yaxis=dict(title='Taux retard (%)', showgrid=True, gridcolor='#F1F5F9'),
        yaxis2=dict(title='Nombre de vols', overlaying='y', side='right'),
        legend=dict(orientation='h', y=1.15),
        xaxis=dict(showgrid=False, tickvals=year_stats['YEAR'].tolist())
    )
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown("""<div class="insight-box orange">
    ⚠️ <strong>Observation :</strong> 2020 marque une chute du nombre de vols due au Covid-19,
    mais paradoxalement un taux de retard plus élevé sur les vols maintenus.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 2 — EXPLORATION
# ═══════════════════════════════════════════════
elif "Exploration" in page:

    st.markdown('<p class="section-title">Exploration des données</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">50 000 vols US · 18 compagnies · 2019–2023</p>', unsafe_allow_html=True)

    # Filtres
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        annees = sorted(df['YEAR'].unique().tolist())
        sel_annees = st.multiselect("Année(s)", annees, default=annees)
    with col_f2:
        compagnies = ["Toutes"] + sorted(df['AIRLINE_SHORT'].unique().tolist())
        sel_compagnie = st.selectbox("Compagnie", compagnies)
    with col_f3:
        saisons = ["Toutes"] + ['Hiver','Printemps','Été','Automne']
        sel_saison = st.selectbox("Saison", saisons)

    # Appliquer filtres
    dff = df[df['YEAR'].isin(sel_annees)].copy()
    if sel_compagnie != "Toutes":
        dff = dff[dff['AIRLINE_SHORT'] == sel_compagnie]
    if sel_saison != "Toutes":
        dff = dff[dff['SAISON'] == sel_saison]

    # KPIs filtrés
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Vols</div>
          <div class="kpi-value">{len(dff):,}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#EF4444;"><div class="kpi-label">Taux retard</div>
          <div class="kpi-value">{dff['RETARDE'].mean()*100:.1f}%</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#F59E0B;"><div class="kpi-label">Retard moyen</div>
          <div class="kpi-value">{dff['ARR_DELAY'].mean():.1f} min</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#10B981;"><div class="kpi-label">Distance moyenne</div>
          <div class="kpi-value">{dff['DISTANCE'].mean():.0f} mi</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # VIZ 1 — Retards par compagnie
    airline_stats = dff.groupby('AIRLINE_SHORT').agg(
        taux=('RETARDE','mean'), total=('RETARDE','count'),
        retard_moy=('ARR_DELAY','mean')
    ).reset_index().sort_values('taux', ascending=True)
    airline_stats['taux_pct'] = (airline_stats['taux']*100).round(1)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure(go.Bar(
            x=airline_stats['taux_pct'], y=airline_stats['AIRLINE_SHORT'],
            orientation='h',
            marker=dict(color=airline_stats['taux_pct'],
                       colorscale=[[0,'#2196F3'],[0.5,'#F59E0B'],[1,'#EF4444']]),
            text=airline_stats['taux_pct'].apply(lambda x: f'{x}%'),
            textposition='outside'
        ))
        fig1.add_vline(x=airline_stats['taux_pct'].mean(), line_dash='dot',
                       line_color='gray', annotation_text='Moyenne')
        fig1.update_layout(height=420, title='Taux de retard par compagnie',
            margin=dict(l=0,r=40,t=40,b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
            xaxis=dict(title='Taux retard (%)', showgrid=True, gridcolor='#F1F5F9'),
            yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # VIZ 2 — Distribution retards
        fig2 = px.histogram(dff[dff['ARR_DELAY'].between(-60,120)],
            x='ARR_DELAY', nbins=60, color='RETARDE',
            color_discrete_map={0:'#2196F3', 1:'#EF4444'},
            labels={'ARR_DELAY':'Retard arrivée (min)','RETARDE':'Retardé'},
            title='Distribution des retards à l\'arrivée')
        fig2.add_vline(x=15, line_dash='dash', line_color='black',
                       annotation_text='Seuil 15 min')
        fig2.update_layout(height=420, margin=dict(l=0,r=0,t=40,b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
            legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

    # VIZ 3 — Carte des routes
    st.markdown('<p class="section-title">Carte des routes aériennes</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Top 30 routes les plus fréquentes — couleur = taux de retard</p>', unsafe_allow_html=True)

    routes = dff.groupby(['ORIGIN','DEST']).agg(
        count=('RETARDE','count'), taux=('RETARDE','mean')
    ).reset_index().sort_values('count', ascending=False).head(30)

    fig_routes = go.Figure()
    # Fond aéroports
    for _, row in routes.iterrows():
        o, d = row['ORIGIN'], row['DEST']
        if o in AIRPORTS and d in AIRPORTS:
            lat_o, lon_o = AIRPORTS[o][1], AIRPORTS[o][2]
            lat_d, lon_d = AIRPORTS[d][1], AIRPORTS[d][2]
            color = f'rgba(239,68,68,{min(row["taux"]*2, 0.8)})' if row['taux'] > 0.2 \
                    else f'rgba(33,150,243,{min(row["count"]/routes["count"].max()*0.8+0.1, 0.8)})'
            fig_routes.add_trace(go.Scattergeo(
                lat=[lat_o, lat_d, None], lon=[lon_o, lon_d, None],
                mode='lines',
                line=dict(width=1.5, color=color),
                hoverinfo='skip', showlegend=False
            ))

    # Points aéroports
    ap_in_routes = set(routes['ORIGIN'].tolist() + routes['DEST'].tolist())
    ap_data = [(k, v) for k, v in AIRPORTS.items() if k in ap_in_routes]
    fig_routes.add_trace(go.Scattergeo(
        lat=[v[1] for k,v in ap_data],
        lon=[v[2] for k,v in ap_data],
        text=[f"{k} — {v[0]}" for k,v in ap_data],
        mode='markers+text',
        marker=dict(size=8, color='#0D1B2A', line=dict(color='white', width=1)),
        textposition='top center', textfont=dict(size=9, color='#0D1B2A'),
        showlegend=False
    ))
    fig_routes.update_layout(
        geo=dict(scope='usa', showland=True, landcolor='#F1F5F9',
                 showocean=True, oceancolor='#EFF6FF',
                 showcoastlines=True, coastlinecolor='#CBD5E1',
                 projection_type='albers usa'),
        height=440, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_routes, use_container_width=True)

    # VIZ 4 & 5 — Heure et jour
    col3, col4 = st.columns(2)
    with col3:
        hour_stats = dff.groupby('HOUR_DEP')['RETARDE'].mean().reset_index()
        hour_stats['taux_pct'] = hour_stats['RETARDE']*100
        fig4 = go.Figure(go.Scatter(
            x=hour_stats['HOUR_DEP'], y=hour_stats['taux_pct'],
            mode='lines+markers', fill='tozeroy',
            line=dict(color='#1A6BFF', width=2.5),
            fillcolor='rgba(26,107,255,0.1)',
            marker=dict(size=7, color='#1A6BFF')
        ))
        fig4.update_layout(height=300, title='Taux de retard par heure de départ',
            margin=dict(l=0,r=0,t=40,b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
            xaxis=dict(title='Heure', showgrid=False),
            yaxis=dict(title='Taux retard (%)', showgrid=True, gridcolor='#F1F5F9'))
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        day_stats = dff.groupby('DAY_OF_WEEK')['RETARDE'].mean().reset_index()
        day_stats['jour'] = day_stats['DAY_OF_WEEK'].map(lambda x: JOURS[x])
        day_stats['taux_pct'] = day_stats['RETARDE']*100
        colors_day = ['#EF4444' if t > day_stats['taux_pct'].mean() else '#2196F3'
                      for t in day_stats['taux_pct']]
        fig5 = go.Figure(go.Bar(
            x=day_stats['jour'], y=day_stats['taux_pct'],
            marker_color=colors_day,
            text=day_stats['taux_pct'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        fig5.update_layout(height=300, title='Taux de retard par jour de la semaine',
            margin=dict(l=0,r=0,t=40,b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(title='Taux retard (%)', showgrid=True, gridcolor='#F1F5F9'))
        st.plotly_chart(fig5, use_container_width=True)

    # Données brutes
    st.markdown('<p class="section-title" style="margin-top:8px;">Données brutes</p>', unsafe_allow_html=True)
    show = dff[['FL_DATE','AIRLINE_SHORT','ORIGIN','DEST','DEP_DELAY',
                'ARR_DELAY','DISTANCE','RETARDE']].copy()
    show.columns = ['Date','Compagnie','Départ','Arrivée','Retard départ',
                    'Retard arrivée','Distance (mi)','Retardé']
    st.dataframe(show.sort_values('Retard arrivée', ascending=False).head(500),
                 use_container_width=True, height=300)


# ═══════════════════════════════════════════════
# PAGE 3 — ANALYSE APPROFONDIE
# ═══════════════════════════════════════════════
elif "Analyse" in page:

    st.markdown('<p class="section-title">Analyse Approfondie</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Question de recherche : Quels facteurs influencent le plus les retards aériens ?</p>', unsafe_allow_html=True)

    st.markdown("""<div class="insight-box">
     <strong>Hypothèse centrale :</strong> Le retard au départ est le facteur le plus prédictif du retard à l'arrivée.
    Mais d'autres facteurs structurels — compagnie, heure, saison — jouent un rôle significatif
    indépendamment du retard initial.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # VIZ 1 — Scatter DEP_DELAY vs ARR_DELAY
    st.markdown('<p class="section-title">Relation retard départ → retard arrivée</p>', unsafe_allow_html=True)
    sample = df.sample(3000, random_state=42)
    fig1 = px.scatter(sample, x='DEP_DELAY', y='ARR_DELAY',
        color='RETARDE', color_discrete_map={0:'#2196F3', 1:'#EF4444'},
        opacity=0.5, trendline='ols',
        labels={'DEP_DELAY':'Retard départ (min)','ARR_DELAY':'Retard arrivée (min)',
                'RETARDE':'Retardé'},
        title='Corrélation retard départ vs retard arrivée (échantillon 3 000 vols)')
    fig1.add_hline(y=15, line_dash='dash', line_color='black',
                   annotation_text='Seuil retard (15 min)')
    fig1.update_layout(height=380, margin=dict(l=0,r=0,t=40,b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
        xaxis=dict(range=[-50,200], showgrid=True, gridcolor='#F1F5F9'),
        yaxis=dict(range=[-70,300], showgrid=True, gridcolor='#F1F5F9'))
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # VIZ 2 — Box plots par saison
        saison_order = ['Hiver','Printemps','Été','Automne']
        saison_colors = {'Hiver':'#2196F3','Printemps':'#10B981',
                        'Été':'#F59E0B','Automne':'#EF4444'}
        fig2 = px.box(df[df['ARR_DELAY'].between(-30,120)],
            x='SAISON', y='ARR_DELAY', color='SAISON',
            color_discrete_map=saison_colors,
            category_orders={'SAISON': saison_order},
            labels={'ARR_DELAY':'Retard arrivée (min)','SAISON':'Saison'},
            title='Distribution des retards par saison')
        fig2.add_hline(y=15, line_dash='dash', line_color='black')
        fig2.update_layout(height=360, margin=dict(l=0,r=0,t=40,b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # VIZ 3 — Heatmap heure × jour
        heat_data = df.groupby(['DAY_OF_WEEK','HOUR_DEP'])['RETARDE'].mean().reset_index()
        heat_pivot = heat_data.pivot(index='DAY_OF_WEEK', columns='HOUR_DEP', values='RETARDE')
        heat_pivot.index = JOURS[:len(heat_pivot)]

        fig3 = px.imshow(heat_pivot*100, aspect='auto',
            color_continuous_scale=[[0,'#EFF6FF'],[0.4,'#93C5FD'],[0.7,'#F59E0B'],[1,'#EF4444']],
            labels=dict(x='Heure de départ', y='Jour', color='Taux retard (%)'),
            title='Heatmap : Taux de retard par heure et jour')
        fig3.update_layout(height=360, margin=dict(l=0,r=0,t=40,b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(title='%', thickness=12))
        st.plotly_chart(fig3, use_container_width=True)

    # VIZ 4 — Top aéroports les plus retardés
    col3, col4 = st.columns(2)
    with col3:
        airport_delay = df.groupby('ORIGIN').agg(
            taux=('RETARDE','mean'), count=('RETARDE','count')
        ).reset_index()
        airport_delay = airport_delay[airport_delay['count'] > 100]
        airport_delay['taux_pct'] = (airport_delay['taux']*100).round(1)
        airport_delay['name'] = airport_delay['ORIGIN'].map(
            lambda x: AIRPORTS.get(x, (x,))[0])
        top_delayed = airport_delay.nlargest(12, 'taux_pct')

        fig4 = px.bar(top_delayed, x='taux_pct', y='ORIGIN',
            orientation='h', color='taux_pct',
            color_continuous_scale=[[0,'#F59E0B'],[1,'#EF4444']],
            text='taux_pct',
            hover_data={'name': True, 'count': True},
            labels={'taux_pct':'Taux retard (%)','ORIGIN':'Aéroport'},
            title='Top 12 aéroports — Taux de retard')
        fig4.update_traces(texttemplate='%{text}%', textposition='outside')
        fig4.update_layout(height=360, margin=dict(l=0,r=40,t=40,b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
            showlegend=False, coloraxis_showscale=False,
            yaxis=dict(categoryorder='total ascending'))
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        # VIZ 5 — Retard moyen par distance
        df['DISTANCE_CAT'] = pd.cut(df['DISTANCE'],
            bins=[0,500,1000,1500,2000,5000],
            labels=['0-500','500-1000','1000-1500','1500-2000','2000+'])
        dist_stats = df.groupby('DISTANCE_CAT', observed=True).agg(
            taux=('RETARDE','mean'), count=('RETARDE','count')
        ).reset_index()
        dist_stats['taux_pct'] = (dist_stats['taux']*100).round(1)

        fig5 = px.bar(dist_stats, x='DISTANCE_CAT', y='taux_pct',
            color='taux_pct',
            color_continuous_scale=[[0,'#2196F3'],[1,'#EF4444']],
            text='taux_pct',
            labels={'DISTANCE_CAT':'Distance (miles)','taux_pct':'Taux retard (%)'},
            title='Taux de retard par distance de vol')
        fig5.update_traces(texttemplate='%{text}%', textposition='outside')
        fig5.update_layout(height=360, margin=dict(l=0,r=0,t=40,b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9'))
        st.plotly_chart(fig5, use_container_width=True)

    # Insights
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""<div class="insight-box">
        📊 <strong>Résultat principal :</strong> Le retard au départ explique à lui seul plus de
        <strong>85% de la variance</strong> du retard à l'arrivée. C'est la feature la plus importante du modèle ML.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="insight-box orange">
        🕐 <strong>Effet de l'heure :</strong> Les vols du soir (après 18h) sont
        <strong>2× plus retardés</strong> que les vols du matin (avant 9h) — effet cascade des retards accumulés.
        </div>""", unsafe_allow_html=True)
    with col_i2:
        st.markdown("""<div class="insight-box green">
        ✅ <strong>Surprise :</strong> La distance du vol n'est pas un facteur significatif.
        Les vols long-courriers ne sont pas plus retardés que les vols courts.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="insight-box red">
        ⚠️ <strong>Limitation :</strong> Le modèle utilise le retard au départ comme feature.
        Dans la réalité, cette information n'est disponible qu'au moment du départ — pas à l'avance.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 4 — PRÉDICTION ML
# ═══════════════════════════════════════════════
elif "Prédiction" in page:

    st.markdown('<p class="section-title">Prédiction ML — Mon vol sera-t-il retardé ?</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Random Forest · Accuracy 93% · AUC-ROC 0.913</p>', unsafe_allow_html=True)

    # Métriques modèle
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Accuracy</div>
          <div class="kpi-value">{model_info['accuracy']*100:.1f}%</div>
          <div class="kpi-delta">▲ Sur données test</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#10B981;"><div class="kpi-label">AUC-ROC</div>
          <div class="kpi-value">{model_info['auc']:.3f}</div>
          <div class="kpi-delta">▲ Très bon score</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#F59E0B;"><div class="kpi-label">Précision retard</div>
          <div class="kpi-value">92%</div>
          <div class="kpi-delta">▲ Quand il prédit retard</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card" style="border-left-color:#8B5CF6;"><div class="kpi-label">Algorithme</div>
          <div class="kpi-value" style="font-size:18px;">Random Forest</div>
          <div class="kpi-delta">▲ 100 arbres</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("### ✈️ Entrez les informations de votre vol")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**🏢 Compagnie & Route**")
        airlines_list = sorted(le_airline.classes_.tolist())
        sel_airline = st.selectbox("Compagnie aérienne", airlines_list)

        origins_list = sorted([o for o in le_origin.classes_.tolist() if o in AIRPORTS])
        sel_origin = st.selectbox("Aéroport de départ", origins_list,
            format_func=lambda x: f"{x} — {AIRPORTS.get(x,(x,))[0]}")

        dests_list = sorted([d for d in le_dest.classes_.tolist() if d in AIRPORTS])
        sel_dest = st.selectbox("Aéroport d'arrivée", dests_list,
            format_func=lambda x: f"{x} — {AIRPORTS.get(x,(x,))[0]}")

    with col_b:
        st.markdown("**📅 Date & Heure**")
        sel_month = st.selectbox("Mois", range(1,13),
            format_func=lambda x: MOIS[x-1])
        sel_dow = st.selectbox("Jour de la semaine", range(7),
            format_func=lambda x: JOURS[x])
        sel_hour = st.slider("Heure de départ", 0, 23, 10)

    with col_c:
        st.markdown("**⏱️ Informations vol**")
        sel_dep_delay = st.number_input("Retard au départ (min)",
            min_value=-40, max_value=600, value=0,
            help="0 si départ à l'heure, négatif si en avance")
        sel_distance = st.number_input("Distance (miles)",
            min_value=50, max_value=5000, value=800)
        sel_elapsed = st.number_input("Durée prévue (min)",
            min_value=30, max_value=600, value=120)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col_btn = st.columns([1,2,1])
    with col_btn[1]:
        predict_btn = st.button("🔮 Prédire le retard", use_container_width=True, type="primary")

    if predict_btn:
        # Features dérivées
        def tranche_horaire(h):
            if h < 6:  return 0
            if h < 12: return 1
            if h < 18: return 2
            return 3

        def get_saison(m):
            if m in [12,1,2]: return 0
            if m in [3,4,5]:  return 1
            if m in [6,7,8]:  return 2
            return 3

        # Encodage
        try:
            airline_enc = int(np.where(le_airline.classes_ == sel_airline)[0][0])
        except:
            airline_enc = 0
        try:
            origin_enc = int(np.where(le_origin.classes_ == sel_origin)[0][0])
        except:
            origin_enc = 0
        try:
            dest_enc = int(np.where(le_dest.classes_ == sel_dest)[0][0])
        except:
            dest_enc = 0

        X_pred = pd.DataFrame([[
            sel_dep_delay,
            airline_enc,
            origin_enc,
            dest_enc,
            sel_distance,
            sel_elapsed,
            sel_hour,
            tranche_horaire(sel_hour),
            sel_dow,
            1 if sel_dow >= 5 else 0,
            sel_month,
            get_saison(sel_month),
            1 if sel_distance > 1500 else 0
        ]], columns=model_info['features'])

        proba    = rf.predict_proba(X_pred)[0][1]
        pred_cls = int(proba >= 0.5)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("### 📊 Résultat de la prédiction")

        col_res1, col_res2, col_res3 = st.columns([2,1,1])

        with col_res1:
            if pred_cls == 1:
                st.markdown(f"""<div class="pred-card red">
                  <div class="pred-val">⚠️ Retardé</div>
                  <div class="pred-prob">Probabilité : {proba*100:.1f}%</div>
                  <div class="pred-label">Le modèle prédit un retard supérieur à 15 min</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="pred-card green">
                  <div class="pred-val">✅ À l'heure</div>
                  <div class="pred-prob">Probabilité retard : {proba*100:.1f}%</div>
                  <div class="pred-label">Le modèle prédit un vol à l'heure</div>
                </div>""", unsafe_allow_html=True)

        with col_res2:
            # Jauge probabilité
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba*100,
                title={'text': "Probabilité retard"},
                gauge=dict(
                    axis=dict(range=[0,100]),
                    bar=dict(color="#EF4444" if proba > 0.5 else "#2196F3"),
                    steps=[
                        dict(range=[0,30], color="#ECFDF5"),
                        dict(range=[30,60], color="#FFF7ED"),
                        dict(range=[60,100], color="#FEF2F2")
                    ],
                    threshold=dict(line=dict(color="black",width=3), value=50)
                ),
                number=dict(suffix="%", font=dict(size=28))
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=20,r=20,t=40,b=0),
                paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_res3:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            # Résumé vol
            origin_name = AIRPORTS.get(sel_origin, (sel_origin,))[0]
            dest_name   = AIRPORTS.get(sel_dest,   (sel_dest,))[0]
            airline_short = sel_airline.replace(' Airlines','').replace(' Air Lines','').replace(' Inc.','').replace(' Co.','').strip()
            st.markdown(f"""
            <div style='background:white;border-radius:12px;padding:16px;border:1px solid #E8ECF0;'>
              <div style='font-size:11px;color:#6B7C93;text-transform:uppercase;margin-bottom:8px;'>Résumé vol</div>
              <div style='font-size:13px;color:#0D1B2A;margin:4px 0;'>✈️ <b>{airline_short}</b></div>
              <div style='font-size:13px;color:#0D1B2A;margin:4px 0;'>🛫 {sel_origin} → {sel_dest}</div>
              <div style='font-size:13px;color:#0D1B2A;margin:4px 0;'>📅 {MOIS[sel_month-1]} · {JOURS[sel_dow]}</div>
              <div style='font-size:13px;color:#0D1B2A;margin:4px 0;'>🕐 {sel_hour:02d}h00</div>
              <div style='font-size:13px;color:#0D1B2A;margin:4px 0;'>📍 {sel_distance} miles</div>
              <div style='font-size:13px;{"color:#EF4444" if sel_dep_delay > 0 else "color:#10B981"};margin:4px 0;'>
              ⏱️ Retard départ : {sel_dep_delay} min</div>
            </div>
            """, unsafe_allow_html=True)

        # Note méthodologique
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("""<div class="insight-box orange">
        ⚠️ <strong>Note importante :</strong> Le retard au départ est la feature la plus influente du modèle (85% d'importance).
        Si vous ne connaissez pas encore le retard au départ (vol futur), mettez 0 pour une prédiction
        basée uniquement sur les facteurs structurels (compagnie, heure, saison, route).
        </div>""", unsafe_allow_html=True)
