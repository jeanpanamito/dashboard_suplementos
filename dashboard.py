# dashboard.py - Dashboard Avanzado con Análisis Completo
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from config import Config
import logging

# Configurar página
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado minimalista oscuro
st.markdown("""
<style>
    /* Tema global oscuro */
    .stApp {
        background-color: #1a1a1a;
        color: #e0e0e0;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #2d2d2d 0%, #1f1f1f 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #404040;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Contenedores de métricas */
    .metric-container {
        background: linear-gradient(135deg, #2a2a2a 0%, #1e1e1e 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #404040;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        border-left: 4px solid #666666;
    }
    
    .metric-container h3 {
        color: #b0b0b0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-container h2 {
        color: #ffffff;
        font-size: 1.8rem;
        margin: 0.5rem 0;
    }
    
    .metric-container p {
        color: #888888;
        font-size: 0.8rem;
        margin: 0;
    }
    
    /* Cajas de insights */
    .insight-box {
        background: linear-gradient(135deg, #2a2a2a 0%, #1e1e1e 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #404040;
        border-left: 4px solid #666666;
    }
    
    .insight-box h4, .insight-box h5 {
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .insight-box p {
        color: #cccccc;
    }
    
    /* Cajas de oportunidades */
    .opportunity-box {
        background: linear-gradient(135deg, #2a2a2a 0%, #1e1e1e 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #404040;
        border-left: 4px solid #888888;
    }
    
    .opportunity-box h4, .opportunity-box h5 {
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .opportunity-box p {
        color: #cccccc;
    }
    
    /* Cajas de advertencias */
    .warning-box {
        background: linear-gradient(135deg, #2a2a2a 0%, #1e1e1e 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #404040;
        border-left: 4px solid #999999;
    }
    
    .warning-box h4, .warning-box h5 {
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .warning-box p {
        color: #cccccc;
    }
    
    /* Tabs estilo oscuro */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 6px;
        padding: 10px 20px;
        color: #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #353535;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #404040;
        border-color: #666666;
        color: #ffffff;
    }
    
    /* Sidebar oscuro */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    /* Headers de secciones */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Texto general */
    .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Métricas de Streamlit */
    .metric-container .css-1xarl3l {
        color: #ffffff;
    }
    
    /* Botones */
    .stButton > button {
        background-color: #2a2a2a;
        border: 1px solid #404040;
        color: #e0e0e0;
        border-radius: 6px;
    }
    
    .stButton > button:hover {
        background-color: #353535;
        border-color: #666666;
    }
    
    /* Selectbox y inputs */
    .stSelectbox > div > div {
        background-color: #2a2a2a;
        border: 1px solid #404040;
        color: #e0e0e0;
    }
    
    .stMultiSelect > div > div {
        background-color: #2a2a2a;
        border: 1px solid #404040;
    }
    
    .stSlider > div > div > div {
        background-color: #404040;
    }
    
    /* Footer personalizado */
    .footer-dark {
        background: linear-gradient(135deg, #2a2a2a 0%, #1e1e1e 100%);
        border: 1px solid #404040;
        border-radius: 10px;
        margin-top: 2rem;
        color: #cccccc;
    }
    
    .footer-dark h4 {
        color: #ffffff;
    }
    
    .footer-dark .badge {
        background: #404040;
        color: #ffffff;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 0 5px;
        border: 1px solid #666666;
    }
    
    /* Expander oscuro */
    .streamlit-expanderHeader {
        background-color: #2a2a2a;
        border: 1px solid #404040;
        color: #e0e0e0;
    }
    
    .streamlit-expanderContent {
        background-color: #1e1e1e;
        border: 1px solid #404040;
        border-top: none;
    }
</style>
""", unsafe_allow_html=True)

def show_connection_status():
    """Muestra el estado de conexión en la sidebar"""
    st.sidebar.header("🔗 Estado de Conexión")
    
    with st.sidebar:
        status = Config.test_connection()
        
        if status['connected']:
            st.success("✅ MongoDB Conectado")
        else:
            st.error("❌ MongoDB Desconectado")
            if status['error']:
                st.error(f"Error: {status['error']}")
        
        # Mostrar detalles
        with st.expander("📊 Detalles de Conexión"):
            st.write(f"**Base de datos:** {Config.DATABASE_NAME}")
            st.write(f"**Colección:** {Config.COLLECTION_NAME}")
            st.write(f"**BD existe:** {'✅' if status['database_exists'] else '❌'}")
            st.write(f"**Colección existe:** {'✅' if status['collection_exists'] else '❌'}")
            st.write(f"**Documentos:** {status['document_count']}")
        
        # Botón para reconectar
        if st.button("🔄 Reconectar"):
            st.cache_data.clear()
            st.rerun()
        
        return status

@st.cache_data(ttl=Config.CACHE_TTL)
def load_data():
    """Carga datos con cache mejorado"""
    try:
        collection = Config.get_mongo_collection()
        if collection is None:
            return pd.DataFrame(), "Error: No se pudo conectar a MongoDB"
        
        # Obtener datos
        cursor = collection.find({})
        data = list(cursor)
        
        if not data:
            return pd.DataFrame(), "Warning: No hay datos en la colección"
        
        # Convertir a DataFrame
        df = pd.DataFrame(data)
        
        # Procesar datos según tu estructura
        df = process_dataframe(df)
        
        return df, f"Success: {len(df)} productos cargados"
        
    except Exception as e:
        logging.error(f"Error cargando datos: {e}")
        return pd.DataFrame(), f"Error: {str(e)}"

def process_dataframe(df):
    """Procesa el DataFrame según la estructura de MongoDB"""
    df_processed = df.copy()
    
    # Mapear campos según tu estructura
    column_mapping = {
        'search_category': 'categoria',
        'title': 'producto',
        'price': 'precio_str',
        'extracted_price': 'precio',
        'source': 'fuente',
        'rating': 'rating',
        'reviews': 'reviews'
    }
    
    # Renombrar columnas
    for old_col, new_col in column_mapping.items():
        if old_col in df_processed.columns:
            df_processed[new_col] = df_processed[old_col]
    
    # Procesar precios
    if 'precio' in df_processed.columns:
        df_processed['precio_num'] = pd.to_numeric(df_processed['precio'], errors='coerce')
    elif 'extracted_price' in df_processed.columns:
        df_processed['precio_num'] = pd.to_numeric(df_processed['extracted_price'], errors='coerce')
    
    # Procesar ratings
    if 'rating' in df_processed.columns:
        df_processed['rating_num'] = pd.to_numeric(df_processed['rating'], errors='coerce')
    
    # Procesar reviews
    if 'reviews' in df_processed.columns:
        df_processed['reviews_num'] = pd.to_numeric(df_processed['reviews'], errors='coerce')
    
    # Agregar timestamp de procesamiento
    if 'metadata' in df_processed.columns:
        df_processed['timestamp'] = pd.to_datetime(df_processed['metadata'].apply(
            lambda x: x.get('processed_at', datetime.now()) if isinstance(x, dict) else datetime.now()
        ))
    else:
        df_processed['timestamp'] = datetime.now()
    
    # Calcular métricas adicionales
    if 'precio_num' in df_processed.columns and 'rating_num' in df_processed.columns:
        df_processed['valor_score'] = df_processed['rating_num'] / (df_processed['precio_num'] / 10)
        df_processed['valor_score'] = df_processed['valor_score'].replace([np.inf, -np.inf], np.nan)
    
    # Calcular ranking de popularidad (reviews + rating)
    if 'reviews_num' in df_processed.columns and 'rating_num' in df_processed.columns:
        df_processed['popularidad_score'] = (
            df_processed['reviews_num'].fillna(0) * 0.7 + 
            df_processed['rating_num'].fillna(0) * 1000 * 0.3
        )
    
    # Categorizar precios
    if 'precio_num' in df_processed.columns:
        df_processed['rango_precio'] = pd.cut(
            df_processed['precio_num'], 
            bins=[0, 20, 40, 60, 100, float('inf')],
            labels=['Económico (<$20)', 'Accesible ($20-40)', 'Medio ($40-60)', 'Premium ($60-100)', 'Lujo (>$100)']
        )
    
    return df_processed

def create_advanced_metrics(df):
    """Crea métricas avanzadas para el dashboard"""
    metrics = {}
    
    if not df.empty:
        # Métricas básicas
        metrics['total_productos'] = len(df)
        metrics['precio_promedio'] = df['precio_num'].mean() if 'precio_num' in df.columns else 0
        metrics['rating_promedio'] = df['rating_num'].mean() if 'rating_num' in df.columns else 0
        metrics['total_fuentes'] = df['fuente'].nunique() if 'fuente' in df.columns else 0
        metrics['total_categorias'] = df['categoria'].nunique() if 'categoria' in df.columns else 0
        
        # Métricas avanzadas
        if 'precio_num' in df.columns:
            metrics['precio_mediano'] = df['precio_num'].median()
            metrics['precio_std'] = df['precio_num'].std()
            metrics['producto_mas_caro'] = df.loc[df['precio_num'].idxmax(), 'producto'] if not df['precio_num'].isna().all() else "N/A"
            metrics['producto_mas_barato'] = df.loc[df['precio_num'].idxmin(), 'producto'] if not df['precio_num'].isna().all() else "N/A"
        
        if 'rating_num' in df.columns:
            metrics['productos_alta_calidad'] = len(df[df['rating_num'] >= 4.5])
            metrics['productos_baja_calidad'] = len(df[df['rating_num'] < 3.0])
        
        if 'reviews_num' in df.columns:
            metrics['total_reviews'] = df['reviews_num'].sum()
            metrics['promedio_reviews'] = df['reviews_num'].mean()
        
        # Oportunidades de mercado
        if 'valor_score' in df.columns:
            metrics['mejor_valor'] = df.loc[df['valor_score'].idxmax(), 'producto'] if not df['valor_score'].isna().all() else "N/A"
    
    return metrics

def show_advanced_kpis(df):
    """Muestra KPIs avanzados"""
    metrics = create_advanced_metrics(df)
    
    st.header("📈 KPIs Ejecutivos")
    
    # Primera fila de métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>📦 Total Productos</h3>
            <h2>{:,}</h2>
            <p>En {} categorías</p>
        </div>
        """.format(metrics.get('total_productos', 0), metrics.get('total_categorias', 0)), 
        unsafe_allow_html=True)
    
    with col2:
        precio_prom = metrics.get('precio_promedio', 0)
        precio_med = metrics.get('precio_mediano', 0)
        diferencia = ((precio_prom - precio_med) / precio_med * 100) if precio_med > 0 else 0
        st.markdown("""
        <div class="metric-container">
            <h3>💰 Precio Promedio</h3>
            <h2>${:.2f}</h2>
            <p>Mediano: ${:.2f} ({:+.1f}%)</p>
        </div>
        """.format(precio_prom, precio_med, diferencia), 
        unsafe_allow_html=True)
    
    with col3:
        rating_prom = metrics.get('rating_promedio', 0)
        alta_calidad = metrics.get('productos_alta_calidad', 0)
        st.markdown("""
        <div class="metric-container">
            <h3>⭐ Rating Promedio</h3>
            <h2>{:.1f}/5.0</h2>
            <p>{} productos premium (≥4.5★)</p>
        </div>
        """.format(rating_prom, alta_calidad), 
        unsafe_allow_html=True)
    
    with col4:
        total_reviews = metrics.get('total_reviews', 0)
        promedio_reviews = metrics.get('promedio_reviews', 0)
        st.markdown("""
        <div class="metric-container">
            <h3>📊 Total Reviews</h3>
            <h2>{:,.0f}</h2>
            <p>Promedio: {:,.0f} por producto</p>
        </div>
        """.format(total_reviews, promedio_reviews), 
        unsafe_allow_html=True)

def create_market_analysis_charts(df):
    """Crea gráficas de análisis de mercado"""
    
    # 1. Distribución de precios por categoría (Box Plot)
    if 'precio_num' in df.columns and 'categoria' in df.columns:
        fig_box = px.box(
            df, 
            x='categoria', 
            y='precio_num',
            title="📊 Distribución de Precios por Categoría",
            color='categoria',
            points="outliers"
        )
        fig_box.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # 2. Matriz de correlación Rating vs Precio vs Reviews
    col1, col2 = st.columns(2)
    
    with col1:
        if all(col in df.columns for col in ['precio_num', 'rating_num', 'reviews_num']):
            fig_scatter = px.scatter(
                df, 
                x='precio_num', 
                y='rating_num',
                size='reviews_num',
                color='categoria',
                title="💎 Relación Precio-Rating-Popularidad",
                labels={'precio_num': 'Precio ($)', 'rating_num': 'Rating'},
                hover_data=['producto', 'fuente']
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # 3. Top 10 fuentes por número de productos
        if 'fuente' in df.columns:
            top_fuentes = df['fuente'].value_counts().head(10)
            fig_fuentes = px.bar(
                x=top_fuentes.values,
                y=top_fuentes.index,
                orientation='h',
                title="🏪 Top 10 Fuentes por Volumen",
                labels={'x': 'Número de Productos', 'y': 'Fuente'},
                color=top_fuentes.values,
                color_continuous_scale='viridis'
            )
            fig_fuentes.update_layout(height=400)
            st.plotly_chart(fig_fuentes, use_container_width=True)

def create_competitive_analysis(df):
    """Análisis competitivo avanzado"""
    st.header("🎯 Análisis Competitivo")
    
    tab1, tab2, tab3 = st.tabs(["📊 Panorama General", "💰 Estrategia de Precios", "⭐ Calidad y Satisfacción"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cuota de mercado por fuente
            if 'fuente' in df.columns:
                cuota_mercado = df['fuente'].value_counts()
                fig_pie = px.pie(
                    values=cuota_mercado.values,
                    names=cuota_mercado.index,
                    title="🏆 Cuota de Mercado por Fuente",
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Productos por rango de precio
            if 'rango_precio' in df.columns:
                rango_dist = df['rango_precio'].value_counts()
                fig_rango = px.bar(
                    x=rango_dist.index,
                    y=rango_dist.values,
                    title="💵 Distribución por Rango de Precio",
                    color=rango_dist.values,
                    color_continuous_scale='plasma'
                )
                fig_rango.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_rango, use_container_width=True)
    
    with tab2:
        # Análisis de precios por fuente
        if all(col in df.columns for col in ['fuente', 'precio_num']):
            precio_por_fuente = df.groupby('fuente').agg({
                'precio_num': ['mean', 'median', 'std', 'count']
            }).round(2)
            precio_por_fuente.columns = ['Precio_Promedio', 'Precio_Mediano', 'Desviación', 'Cantidad_Productos']
            precio_por_fuente = precio_por_fuente.reset_index()
            precio_por_fuente = precio_por_fuente[precio_por_fuente['Cantidad_Productos'] >= 3]  # Filtrar fuentes con pocos productos
            
            # Gráfica de precios promedio por fuente
            fig_precios = px.bar(
                precio_por_fuente,
                x='fuente',
                y='Precio_Promedio',
                title="💰 Precio Promedio por Fuente",
                color='Precio_Promedio',
                color_continuous_scale='reds',
                hover_data=['Precio_Mediano', 'Cantidad_Productos']
            )
            fig_precios.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_precios, use_container_width=True)
            
            # Tabla de análisis de precios
            st.subheader("📋 Análisis Detallado de Precios por Fuente")
            st.dataframe(precio_por_fuente, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating promedio por fuente
            if all(col in df.columns for col in ['fuente', 'rating_num']):
                rating_por_fuente = df.groupby('fuente').agg({
                    'rating_num': 'mean',
                    'producto': 'count'
                }).reset_index()
                rating_por_fuente.columns = ['Fuente', 'Rating_Promedio', 'Cantidad_Productos']
                rating_por_fuente = rating_por_fuente[rating_por_fuente['Cantidad_Productos'] >= 3]
                
                fig_rating = px.scatter(
                    rating_por_fuente,
                    x='Cantidad_Productos',
                    y='Rating_Promedio',
                    size='Cantidad_Productos',
                    color='Rating_Promedio',
                    title="⭐ Rating vs Volumen de Productos",
                    hover_data=['Fuente'],
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_rating, use_container_width=True)
        
        with col2:
            # Distribución de ratings
            if 'rating_num' in df.columns:
                fig_hist = px.histogram(
                    df,
                    x='rating_num',
                    nbins=20,
                    title="📈 Distribución de Ratings",
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_hist.update_layout(
                    xaxis_title="Rating",
                    yaxis_title="Número de Productos"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

def show_business_insights(df):
    """Muestra insights de negocio"""
    st.header("💡 Insights de Negocio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4>🎯 Oportunidades Identificadas</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Análisis de gaps de precio
        if 'categoria' in df.columns and 'precio_num' in df.columns:
            precio_stats = df.groupby('categoria')['precio_num'].agg(['min', 'max', 'mean']).reset_index()
            precio_stats['gap'] = precio_stats['max'] - precio_stats['min']
            precio_stats = precio_stats.sort_values('gap', ascending=False).head(5)
            
            st.write("**🔍 Categorías con Mayor Variación de Precios:**")
            for idx, row in precio_stats.iterrows():
                gap_percentage = (row['gap'] / row['mean']) * 100
                st.write(f"• **{row['categoria']}**: ${row['min']:.2f} - ${row['max']:.2f} (Gap: {gap_percentage:.1f}%)")
    
    with col2:
        st.markdown("""
        <div class="opportunity-box">
            <h4>📊 Recomendaciones Estratégicas</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Top productos por valor
        if 'valor_score' in df.columns:
            top_valor = df.nlargest(5, 'valor_score')[['producto', 'precio_num', 'rating_num', 'valor_score']]
            st.write("**🏆 Top 5 Mejor Relación Calidad-Precio:**")
            for idx, row in top_valor.iterrows():
                st.write(f"• **{row['producto'][:40]}...** - Score: {row['valor_score']:.2f}")

def create_filters_sidebar(df):
    """Crea filtros avanzados en la sidebar"""
    st.sidebar.header("🔍 Filtros de Análisis")
    
    # Filtro por categoría
    categorias_disponibles = df['categoria'].unique() if 'categoria' in df.columns else []
    categoria_seleccionada = st.sidebar.multiselect(
        "Categorías de Productos",
        options=categorias_disponibles,
        default=categorias_disponibles
    )
    
    # Filtro por rango de precios
    if 'precio_num' in df.columns:
        precio_min, precio_max = st.sidebar.slider(
            "Rango de Precios ($)",
            min_value=float(df['precio_num'].min()),
            max_value=float(df['precio_num'].max()),
            value=(float(df['precio_num'].min()), float(df['precio_num'].max())),
            step=1.0
        )
    else:
        precio_min, precio_max = 0, 1000
    
    # Filtro por rating mínimo
    if 'rating_num' in df.columns:
        rating_minimo = st.sidebar.slider(
            "Rating Mínimo",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1
        )
    else:
        rating_minimo = 0.0
    
    # Filtro por fuente
    fuentes_disponibles = df['fuente'].unique() if 'fuente' in df.columns else []
    fuentes_seleccionadas = st.sidebar.multiselect(
        "Fuentes/Marketplaces",
        options=fuentes_disponibles,
        default=fuentes_disponibles
    )
    
    # Filtro por número mínimo de reviews
    if 'reviews_num' in df.columns:
        reviews_minimo = st.sidebar.number_input(
            "Reviews Mínimo",
            min_value=0,
            max_value=int(df['reviews_num'].max()) if not df['reviews_num'].isna().all() else 0,
            value=0,
            step=100
        )
    else:
        reviews_minimo = 0
    
    return {
        'categorias': categoria_seleccionada,
        'precio_min': precio_min,
        'precio_max': precio_max,
        'rating_minimo': rating_minimo,
        'fuentes': fuentes_seleccionadas,
        'reviews_minimo': reviews_minimo
    }

def apply_filters(df, filters):
    """Aplica filtros al DataFrame"""
    df_filtrado = df.copy()
    
    if filters['categorias']:
        df_filtrado = df_filtrado[df_filtrado['categoria'].isin(filters['categorias'])]
    
    if 'precio_num' in df_filtrado.columns:
        df_filtrado = df_filtrado[
            (df_filtrado['precio_num'] >= filters['precio_min']) & 
            (df_filtrado['precio_num'] <= filters['precio_max'])
        ]
    
    if 'rating_num' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['rating_num'] >= filters['rating_minimo']]
    
    if filters['fuentes']:
        df_filtrado = df_filtrado[df_filtrado['fuente'].isin(filters['fuentes'])]
    
    if 'reviews_num' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['reviews_num'] >= filters['reviews_minimo']]
    
    return df_filtrado

# ==================== INTERFAZ PRINCIPAL ====================

# Header principal
st.markdown("""
<div class="main-header">
""", unsafe_allow_html=True)

# Logo en esquina superior derecha y título centrado
title_col, logo_col = st.columns([4, 1])
with title_col:
    st.markdown(f"""
    <h1 style="color: white; margin: 0; text-align: center; line-height: 100px;">
        {Config.PAGE_TITLE}
    </h1>
    """, unsafe_allow_html=True)
with logo_col:
    try:
        st.image("gainsight.png", width=100)
    except:
        st.write("🏋️")  # Fallback si no encuentra la imagen

st.markdown("""
    <p style="color: #e0e0e0; margin: 0; text-align: center; font-size: 1.2em;">Dashboard de Inteligencia de Mercado para Suplementos Deportivos</p>
</div>
""", unsafe_allow_html=True)

# Verificar conexión
connection_status = show_connection_status()

if not connection_status['connected']:
    st.error("❌ No se puede conectar a MongoDB. Verifica que esté ejecutándose.")
    st.stop()

if not connection_status['collection_exists'] or connection_status['document_count'] == 0:
    st.warning("⚠️ La colección está vacía o no existe. Verifica que los datos estén cargados.")
    st.stop()

# Cargar datos
with st.spinner("Cargando y procesando datos..."):
    df, message = load_data()

if message.startswith("Success"):
    st.success(message)
elif message.startswith("Warning"):
    st.warning(message)
else:
    st.error(message)
    st.stop()

if df.empty:
    st.error("❌ No se pudieron cargar los datos")
    st.stop()

# Crear filtros
filters = create_filters_sidebar(df)

# Aplicar filtros
df_filtrado = apply_filters(df, filters)

# Mostrar información de filtros aplicados
if len(df_filtrado) != len(df):
    st.info(f"📊 Mostrando {len(df_filtrado):,} de {len(df):,} productos (filtros aplicados)")

# KPIs avanzados
show_advanced_kpis(df_filtrado)

# Análisis de mercado
st.header("📊 Análisis de Mercado")
create_market_analysis_charts(df_filtrado)

# Análisis competitivo
create_competitive_analysis(df_filtrado)

# Insights de negocio
show_business_insights(df_filtrado)

# Tabla detallada
st.header("📋 Explorador de Productos")
if not df_filtrado.empty:
    # Seleccionar columnas para mostrar
    display_cols = []
    col_mapping = {
        'producto': 'Producto',
        'categoria': 'Categoría', 
        'precio_str': 'Precio',
        'precio_num': 'Precio ($)',
        'rating_num': 'Rating',
        'reviews_num': 'Reviews',
        'fuente': 'Fuente',
        'valor_score': 'Score Valor'
    }
    
    for col, label in col_mapping.items():
        if col in df_filtrado.columns:
            display_cols.append(col)
    
    if display_cols:
        # Configurar la tabla con formato
        df_display = df_filtrado[display_cols].copy()
        
        # Formatear columnas
        if 'precio_num' in df_display.columns:
            df_display['precio_num'] = df_display['precio_num'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        
        if 'rating_num' in df_display.columns:
            df_display['rating_num'] = df_display['rating_num'].apply(lambda x: f"{x:.1f}⭐" if pd.notna(x) else "N/A")
        
        if 'reviews_num' in df_display.columns:
            df_display['reviews_num'] = df_display['reviews_num'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
        
        if 'valor_score' in df_display.columns:
            df_display['valor_score'] = df_display['valor_score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        # Renombrar columnas
        df_display.columns = [col_mapping.get(col, col) for col in df_display.columns]
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=400
        )

# Footer mejorado
st.markdown("---")
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; margin-top: 2rem;">
    <h4>📊 {Config.PAGE_TITLE}</h4>
    <p><strong>Última actualización:</strong> {timestamp}</p>
    <p><strong>Datos procesados:</strong> {len(df):,} productos | <strong>Fuentes:</strong> {df['fuente'].nunique() if 'fuente' in df.columns else 0} marketplaces</p>
    <p><strong>Entorno:</strong> {Config.ENVIRONMENT} | <strong>Versión:</strong> 2.0.0</p>
    <div style="margin-top: 15px;">
        <span style="background: #667eea; color: white; padding: 5px 15px; border-radius: 20px; margin: 0 5px;">🔄 Actualización automática</span>
        <span style="background: #4ecdc4; color: white; padding: 5px 15px; border-radius: 20px; margin: 0 5px;">📊 Analytics en tiempo real</span>
        <span style="background: #ff6b6b; color: white; padding: 5px 15px; border-radius: 20px; margin: 0 5px;">🎯 Inteligencia competitiva</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Información adicional en sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Estadísticas de Sesión")
st.sidebar.metric("Productos Analizados", f"{len(df_filtrado):,}")
st.sidebar.metric("Filtros Activos", sum([
    len(filters['categorias']) < len(df['categoria'].unique()) if 'categoria' in df.columns else False,
    filters['precio_min'] > df['precio_num'].min() if 'precio_num' in df.columns else False,
    filters['precio_max'] < df['precio_num'].max() if 'precio_num' in df.columns else False,
    filters['rating_minimo'] > 0,
    len(filters['fuentes']) < len(df['fuente'].unique()) if 'fuente' in df.columns else False,
    filters['reviews_minimo'] > 0
]))

st.sidebar.markdown("### 💡 Acciones Rápidas")
if st.sidebar.button("📊 Exportar Análisis"):
    st.sidebar.success("Funcionalidad de exportación próximamente")

if st.sidebar.button("🔄 Actualizar Datos"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("📋 Reporte Ejecutivo"):
    st.sidebar.info("Generando reporte ejecutivo...")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Acerca del Dashboard")
st.sidebar.info("""
**SuppMarket Intelligence v2.0**

Dashboard especializado para análisis de mercado de suplementos deportivos.

**Características:**
- 📊 Analytics en tiempo real
- 🎯 Inteligencia competitiva
- 💰 Análisis de precios
- ⭐ Métricas de calidad
- 🔍 Filtros avanzados
- 📈 KPIs ejecutivos

**Datos:**
- Fuente: Google Shopping API
- Actualización: Tiempo real
- Cobertura: Multi-marketplace
""")

# Panel de control avanzado (expandible)
with st.expander("🔧 Panel de Control Avanzado"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Configuración de Visualización")
        chart_theme = st.selectbox("Tema de Gráficas", ["plotly", "plotly_white", "plotly_dark"])
        show_trends = st.checkbox("Mostrar Tendencias", value=True)
        show_outliers = st.checkbox("Mostrar Valores Atípicos", value=True)
    
    with col2:
        st.subheader("🎯 Alertas de Mercado")
        price_alert = st.number_input("Alerta de Precio Alto ($)", value=100.0)
        rating_alert = st.number_input("Alerta de Rating Bajo", value=3.0, max_value=5.0)
        review_alert = st.number_input("Alerta de Pocas Reviews", value=100)
    
    with col3:
        st.subheader("⚡ Acciones Avanzadas")
        if st.button("🔍 Análisis Predictivo"):
            st.info("Funcionalidad próximamente")
        if st.button("📧 Configurar Alertas"):
            st.info("Sistema de alertas próximamente")
        if st.button("🎯 Análisis de Competidores"):
            st.info("Análisis competitivo avanzado próximamente")

# Alertas automáticas basadas en los datos con productos específicos
st.markdown("### 🚨 Alertas de Mercado")
alerts_col1, alerts_col2, alerts_col3 = st.columns(3)

with alerts_col1:
    # Productos con precios muy altos
    if 'precio_num' in df_filtrado.columns:
        high_price_products = df_filtrado[df_filtrado['precio_num'] > df_filtrado['precio_num'].quantile(0.95)]
        if len(high_price_products) > 0:
            st.markdown("""
            <div class="warning-box">
                <h5>💰 Productos de Precio Elevado</h5>
                <p><strong>{} productos</strong> con precios en el top 5%</p>
                <p>Precio promedio: <strong>${:.2f}</strong></p>
            </div>
            """.format(len(high_price_products), high_price_products['precio_num'].mean()), 
            unsafe_allow_html=True)
            
            # Expandible con lista de productos
            with st.expander(f"📋 Ver {len(high_price_products)} productos de precio elevado"):
                for idx, (_, product) in enumerate(high_price_products.head(10).iterrows(), 1):
                    precio = product.get('precio_num', 0)
                    producto_nombre = product.get('producto', 'Producto sin nombre')
                    fuente = product.get('fuente', 'N/A')
                    rating = product.get('rating_num', 0)
                    
                    st.markdown(f"""
                    **{idx}. {producto_nombre[:60]}{'...' if len(producto_nombre) > 60 else ''}**  
                    💰 **${precio:.2f}** | ⭐ {rating:.1f} | 🏪 {fuente}
                    """)
                    
                if len(high_price_products) > 10:
                    st.info(f"💡 Mostrando los primeros 10 de {len(high_price_products)} productos")

with alerts_col2:
    # Productos con ratings bajos
    if 'rating_num' in df_filtrado.columns:
        low_rating_products = df_filtrado[df_filtrado['rating_num'] < 3.5]
        if len(low_rating_products) > 0:
            st.markdown("""
            <div class="warning-box">
                <h5>⭐ Productos de Baja Calidad</h5>
                <p><strong>{} productos</strong> con rating < 3.5</p>
                <p>Requieren atención especial</p>
            </div>
            """.format(len(low_rating_products)), 
            unsafe_allow_html=True)
            
            # Expandible con lista de productos
            with st.expander(f"📋 Ver {len(low_rating_products)} productos de baja calidad"):
                # Ordenar por rating más bajo primero
                low_rating_sorted = low_rating_products.sort_values('rating_num', ascending=True)
                
                for idx, (_, product) in enumerate(low_rating_sorted.head(10).iterrows(), 1):
                    precio = product.get('precio_num', 0)
                    producto_nombre = product.get('producto', 'Producto sin nombre')
                    fuente = product.get('fuente', 'N/A')
                    rating = product.get('rating_num', 0)
                    reviews = product.get('reviews_num', 0)
                    
                    # Determinar color de alerta
                    if rating < 2.0:
                        emoji_alert = "🔴"
                    elif rating < 3.0:
                        emoji_alert = "🟡"
                    else:
                        emoji_alert = "🟠"
                    
                    st.markdown(f"""
                    **{idx}. {emoji_alert} {producto_nombre[:60]}{'...' if len(producto_nombre) > 60 else ''}**  
                    ⭐ **{rating:.1f}** | 💰 ${precio:.2f} | 📊 {reviews:,.0f} reviews | 🏪 {fuente}
                    """)
                    
                if len(low_rating_products) > 10:
                    st.info(f"💡 Mostrando los primeros 10 de {len(low_rating_products)} productos")

with alerts_col3:
    # Oportunidades de mercado
    if 'valor_score' in df_filtrado.columns:
        high_value_products = df_filtrado[df_filtrado['valor_score'] > df_filtrado['valor_score'].quantile(0.9)]
        if len(high_value_products) > 0:
            st.markdown("""
            <div class="opportunity-box">
                <h5>💎 Oportunidades Detectadas</h5>
                <p><strong>{} productos</strong> con excelente relación calidad-precio</p>
                <p>Potencial de crecimiento alto</p>
            </div>
            """.format(len(high_value_products)), 
            unsafe_allow_html=True)
            
            # Expandible con lista de productos
            with st.expander(f"📋 Ver {len(high_value_products)} oportunidades de mercado"):
                # Ordenar por mejor valor score
                high_value_sorted = high_value_products.sort_values('valor_score', ascending=False)
                
                for idx, (_, product) in enumerate(high_value_sorted.head(10).iterrows(), 1):
                    precio = product.get('precio_num', 0)
                    producto_nombre = product.get('producto', 'Producto sin nombre')
                    fuente = product.get('fuente', 'N/A')
                    rating = product.get('rating_num', 0)
                    valor_score = product.get('valor_score', 0)
                    categoria = product.get('categoria', 'N/A')
                    
                    # Emoji según el score de valor
                    if valor_score > 1.0:
                        emoji_value = "🌟"
                    elif valor_score > 0.8:
                        emoji_value = "⭐"
                    else:
                        emoji_value = "💎"
                    
                    st.markdown(f"""
                    **{idx}. {emoji_value} {producto_nombre[:60]}{'...' if len(producto_nombre) > 60 else ''}**  
                    📊 **Score: {valor_score:.2f}** | ⭐ {rating:.1f} | 💰 ${precio:.2f}  
                    🏷️ {categoria} | 🏪 {fuente}
                    """)
                    
                if len(high_value_products) > 10:
                    st.info(f"💡 Mostrando las primeras 10 de {len(high_value_products)} oportunidades")

# Sección adicional: Resumen ejecutivo de alertas
st.markdown("### 📊 Resumen Ejecutivo de Alertas")

# Crear resumen en una sola fila
exec_col1, exec_col2, exec_col3, exec_col4 = st.columns(4)

with exec_col1:
    total_alerts = 0
    if 'precio_num' in df_filtrado.columns:
        high_price_count = len(df_filtrado[df_filtrado['precio_num'] > df_filtrado['precio_num'].quantile(0.95)])
        total_alerts += high_price_count
    else:
        high_price_count = 0
    
    st.metric(
        label="🚨 Alertas de Precio Alto",
        value=high_price_count,
        delta=f"{(high_price_count/len(df_filtrado)*100):.1f}% del total" if len(df_filtrado) > 0 else "0%"
    )

with exec_col2:
    if 'rating_num' in df_filtrado.columns:
        low_quality_count = len(df_filtrado[df_filtrado['rating_num'] < 3.5])
        total_alerts += low_quality_count
    else:
        low_quality_count = 0
    
    st.metric(
        label="⭐ Alertas de Calidad",
        value=low_quality_count,
        delta=f"{(low_quality_count/len(df_filtrado)*100):.1f}% del total" if len(df_filtrado) > 0 else "0%"
    )

with exec_col3:
    if 'valor_score' in df_filtrado.columns:
        opportunities_count = len(df_filtrado[df_filtrado['valor_score'] > df_filtrado['valor_score'].quantile(0.9)])
    else:
        opportunities_count = 0
    
    st.metric(
        label="💎 Oportunidades",
        value=opportunities_count,
        delta=f"{(opportunities_count/len(df_filtrado)*100):.1f}% del total" if len(df_filtrado) > 0 else "0%"
    )

with exec_col4:
    st.metric(
        label="📊 Total Alertas",
        value=total_alerts,
        delta=f"De {len(df_filtrado):,} productos analizados"
    )

# Alertas críticas adicionales
if len(df_filtrado) > 0:
    st.markdown("### ⚠️ Alertas Críticas Adicionales")
    
    critical_col1, critical_col2 = st.columns(2)
    
    with critical_col1:
        # Productos sin reviews suficientes
        if 'reviews_num' in df_filtrado.columns:
            low_reviews = df_filtrado[df_filtrado['reviews_num'] < 100]
            if len(low_reviews) > 0:
                st.warning(f"📊 **{len(low_reviews)} productos** tienen menos de 100 reviews (datos insuficientes)")
                
                if st.checkbox("Mostrar productos con pocas reviews"):
                    low_reviews_sorted = low_reviews.sort_values('reviews_num', ascending=True)
                    for _, product in low_reviews_sorted.head(5).iterrows():
                        st.text(f"• {product.get('producto', 'N/A')[:50]}... - {product.get('reviews_num', 0):.0f} reviews")
    
    with critical_col2:
        # Productos con precio extremadamente bajo (posibles problemas de calidad)
        if 'precio_num' in df_filtrado.columns:
            very_cheap = df_filtrado[df_filtrado['precio_num'] < df_filtrado['precio_num'].quantile(0.05)]
            if len(very_cheap) > 0:
                st.info(f"💰 **{len(very_cheap)} productos** tienen precios extremadamente bajos (posible dumping)")
                
                if st.checkbox("Mostrar productos de precio muy bajo"):
                    very_cheap_sorted = very_cheap.sort_values('precio_num', ascending=True)
                    for _, product in very_cheap_sorted.head(5).iterrows():
                        st.text(f"• {product.get('producto', 'N/A')[:50]}... - ${product.get('precio_num', 0):.2f}")

# Acciones recomendadas
st.markdown("### 🎯 Acciones Recomendadas")

actions_col1, actions_col2 = st.columns(2)

with actions_col1:
    st.markdown("""
    <div class="insight-box">
        <h5>📈 Estrategias de Mercado</h5>
        <ul>
            <li><strong>Precios altos:</strong> Analizar justificación de premium pricing</li>
            <li><strong>Baja calidad:</strong> Investigar problemas de productos</li>
            <li><strong>Oportunidades:</strong> Considerar productos similares</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with actions_col2:
    st.markdown("""
    <div class="opportunity-box">
        <h5>🔍 Próximos Pasos</h5>
        <ul>
            <li><strong>Monitoreo:</strong> Seguimiento semanal de alertas</li>
            <li><strong>Análisis:</strong> Deep dive en categorías críticas</li>
            <li><strong>Competencia:</strong> Benchmarking vs competidores</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)