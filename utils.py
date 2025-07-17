# utils.py - Utilidades avanzadas para el dashboard
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st

def create_trend_analysis(df, date_column='timestamp', value_column='precio_num', category_column='categoria'):
    """
    Crea análisis de tendencias temporales
    """
    if not all(col in df.columns for col in [date_column, value_column]):
        return None
    
    # Convertir timestamp a datetime si no lo es
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Agrupar por fecha y categoría
    if category_column in df.columns:
        trend_data = df.groupby([df[date_column].dt.date, category_column])[value_column].mean().reset_index()
        
        fig = px.line(
            trend_data, 
            x=date_column, 
            y=value_column,
            color=category_column,
            title=f"📈 Tendencia de {value_column} por {category_column}"
        )
    else:
        trend_data = df.groupby(df[date_column].dt.date)[value_column].mean().reset_index()
        
        fig = px.line(
            trend_data, 
            x=date_column, 
            y=value_column,
            title=f"📈 Tendencia de {value_column}"
        )
    
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title=value_column.replace('_', ' ').title(),
        hovermode='x unified'
    )
    
    return fig

def create_heatmap_correlation(df, numeric_columns=None):
    """
    Crea mapa de calor de correlaciones
    """
    if numeric_columns is None:
        numeric_columns = ['precio_num', 'rating_num', 'reviews_num', 'valor_score', 'popularidad_score']
    
    # Filtrar solo columnas que existen
    available_columns = [col for col in numeric_columns if col in df.columns]
    
    if len(available_columns) < 2:
        return None
    
    # Calcular matriz de correlación
    corr_matrix = df[available_columns].corr()
    
    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="🔥 Mapa de Correlaciones",
        width=600,
        height=500
    )
    
    return fig

def create_market_share_sunburst(df, category_col='categoria', source_col='fuente', value_col='precio_num'):
    """
    Crea gráfico sunburst para cuota de mercado
    """
    if not all(col in df.columns for col in [category_col, source_col]):
        return None
    
    # Preparar datos para sunburst
    df_grouped = df.groupby([category_col, source_col]).agg({
        value_col: 'count' if value_col not in df.columns else 'sum',
        'producto': 'count'
    }).reset_index()
    
    df_grouped.columns = [category_col, source_col, 'value', 'count']
    
    # Crear etiquetas y valores
    labels = []
    parents = []
    values = []
    
    # Nivel 1: Categorías
    categories = df_grouped[category_col].unique()
    for cat in categories:
        labels.append(cat)
        parents.append("")
        values.append(df_grouped[df_grouped[category_col] == cat]['count'].sum())
    
    # Nivel 2: Fuentes por categoría
    for _, row in df_grouped.iterrows():
        labels.append(f"{row[source_col]} ({row[category_col]})")
        parents.append(row[category_col])
        values.append(row['count'])
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total"
    ))
    
    fig.update_layout(
        title="☀️ Cuota de Mercado por Categoría y Fuente",
        height=600
    )
    
    return fig

def create_radar_chart(df, categories, metrics=['precio_num', 'rating_num', 'reviews_num']):
    """
    Crea gráfico radar para comparar categorías
    """
    if 'categoria' not in df.columns:
        return None
    
    # Filtrar métricas disponibles
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) < 3:
        return None
    
    # Normalizar datos (0-1)
    df_norm = df.copy()
    for metric in available_metrics:
        df_norm[f'{metric}_norm'] = (df_norm[metric] - df_norm[metric].min()) / (df_norm[metric].max() - df_norm[metric].min())
    
    fig = go.Figure()
    
    for category in categories[:5]:  # Limitar a 5 categorías
        cat_data = df_norm[df_norm['categoria'] == category]
        if len(cat_data) == 0:
            continue
        
        values = []
        for metric in available_metrics:
            values.append(cat_data[f'{metric}_norm'].mean())
        
        values.append(values[0])  # Cerrar el radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_metrics + [available_metrics[0]],
            fill='toself',
            name=category
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="📡 Comparación Radar por Categorías",
        showlegend=True,
        height=500
    )
    
    return fig

def create_price_distribution_violin(df, category_col='categoria', price_col='precio_num'):
    """
    Crea gráfico violin para distribución de precios
    """
    if not all(col in df.columns for col in [category_col, price_col]):
        return None
    
    fig = go.Figure()
    
    categories = df[category_col].unique()
    colors = px.colors.qualitative.Set3
    
    for i, category in enumerate(categories):
        cat_data = df[df[category_col] == category][price_col].dropna()
        
        if len(cat_data) > 0:
            fig.add_trace(go.Violin(
                y=cat_data,
                name=category,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                opacity=0.7,
                x0=category
            ))
    
    fig.update_layout(
        title="🎻 Distribución de Precios por Categoría (Violin Plot)",
        yaxis_title="Precio ($)",
        height=500,
        showlegend=False
    )
    
    return fig

def create_bubble_chart_3d(df, x_col='precio_num', y_col='rating_num', z_col='reviews_num', 
                          size_col='reviews_num', color_col='categoria'):
    """
    Crea gráfico de burbujas 3D
    """
    required_cols = [x_col, y_col, z_col]
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Limpiar datos
    df_clean = df.dropna(subset=required_cols)
    
    if len(df_clean) == 0:
        return None
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df_clean[x_col],
        y=df_clean[y_col],
        z=df_clean[z_col],
        mode='markers',
        marker=dict(
            size=np.sqrt(df_clean[size_col]) / 10 if size_col in df_clean.columns else 5,
            color=df_clean[color_col].astype('category').cat.codes if color_col in df_clean.columns else 'blue',
            colorscale='viridis',
            showscale=True,
            opacity=0.7
        ),
        text=df_clean['producto'] if 'producto' in df_clean.columns else df_clean.index,
        hovertemplate='<b>%{text}</b><br>' +
                      f'{x_col}: %{{x}}<br>' +
                      f'{y_col}: %{{y}}<br>' +
                      f'{z_col}: %{{z}}<extra></extra>'
    )])
    
    fig.update_layout(
        title="🌐 Análisis 3D: Precio vs Rating vs Reviews",
        scene=dict(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            zaxis_title=z_col.replace('_', ' ').title()
        ),
        height=600
    )
    
    return fig

def calculate_market_insights(df):
    """
    Calcula insights automáticos del mercado
    """
    insights = []
    
    if 'precio_num' in df.columns:
        # Análisis de precios
        precio_mean = df['precio_num'].mean()
        precio_median = df['precio_num'].median()
        precio_std = df['precio_num'].std()
        
        if precio_mean > precio_median * 1.2:
            insights.append({
                'type': 'warning',
                'title': '📊 Distribución de Precios Sesgada',
                'message': f'El precio promedio (${precio_mean:.2f}) es {((precio_mean/precio_median-1)*100):.1f}% mayor que la mediana (${precio_median:.2f}), indicando presencia de productos premium que elevan el promedio.'
            })
        
        if precio_std > precio_mean * 0.5:
            insights.append({
                'type': 'info',
                'title': '💰 Alta Variabilidad de Precios',
                'message': f'Gran diversidad de precios en el mercado (σ=${precio_std:.2f}), sugiriendo múltiples segmentos de mercado.'
            })
    
    if 'rating_num' in df.columns:
        # Análisis de ratings
        high_rating = len(df[df['rating_num'] >= 4.5])
        total_products = len(df)
        high_rating_pct = (high_rating / total_products) * 100
        
        if high_rating_pct > 60:
            insights.append({
                'type': 'success',
                'title': '⭐ Mercado de Alta Calidad',
                'message': f'{high_rating_pct:.1f}% de los productos tienen rating ≥4.5, indicando un mercado maduro con productos de calidad.'
            })
        elif high_rating_pct < 30:
            insights.append({
                'type': 'warning',
                'title': '⚠️ Oportunidad de Mejora en Calidad',
                'message': f'Solo {high_rating_pct:.1f}% de productos tienen rating alto. Oportunidad para productos de mayor calidad.'
            })
    
    if 'categoria' in df.columns:
        # Análisis de concentración por categoría
        cat_concentration = df['categoria'].value_counts()
        top_cat_pct = (cat_concentration.iloc[0] / len(df)) * 100
        
        if top_cat_pct > 50:
            insights.append({
                'type': 'info',
                'title': '🎯 Mercado Concentrado',
                'message': f'La categoría "{cat_concentration.index[0]}" representa {top_cat_pct:.1f}% del mercado, indicando alta concentración.'
            })
    
    if 'fuente' in df.columns:
        # Análisis de concentración por fuente
        source_concentration = df['fuente'].value_counts()
        top_source_pct = (source_concentration.iloc[0] / len(df)) * 100
        
        if top_source_pct > 40:
            insights.append({
                'type': 'warning',
                'title': '🏪 Dependencia de Marketplace',
                'message': f'{source_concentration.index[0]} domina con {top_source_pct:.1f}% de los productos. Considerar diversificación.'
            })
    
    return insights

def export_analysis_report(df, filename="market_analysis_report"):
    """
    Exporta reporte de análisis en múltiples formatos
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear resumen ejecutivo
    summary = {
        'Fecha_Analisis': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Total_Productos': len(df),
        'Categorias_Analizadas': df['categoria'].nunique() if 'categoria' in df.columns else 0,
        'Fuentes_Marketplace': df['fuente'].nunique() if 'fuente' in df.columns else 0,
        'Precio_Promedio': df['precio_num'].mean() if 'precio_num' in df.columns else 0,
        'Rating_Promedio': df['rating_num'].mean() if 'rating_num' in df.columns else 0,
        'Total_Reviews': df['reviews_num'].sum() if 'reviews_num' in df.columns else 0
    }
    
    # DataFrame del resumen
    summary_df = pd.DataFrame([summary])
    
    return summary_df, f"{filename}_{timestamp}"

def create_advanced_filters():
    """
    Crea filtros avanzados para el dashboard
    """
    st.sidebar.markdown("### 🔬 Filtros Avanzados")
    
    # Filtro de tendencia de precios
    price_trend = st.sidebar.selectbox(
        "Tendencia de Precios",
        ["Todos", "Alcista", "Bajista", "Estable"]
    )
    
    # Filtro de popularidad
    popularity_filter = st.sidebar.slider(
        "Índice de Popularidad Mínimo",
        min_value=0,
        max_value=100,
        value=0,
        help="Basado en número de reviews y rating"
    )
    
    # Filtro de novedad
    days_since_added = st.sidebar.number_input(
        "Productos agregados en últimos X días",
        min_value=1,
        max_value=365,
        value=30
    )
    
    # Filtro de competitividad
    competitive_level = st.sidebar.selectbox(
        "Nivel de Competencia",
        ["Todos", "Alta competencia (>10 productos)", "Media competencia (5-10)", "Baja competencia (<5)"]
    )
    
    return {
        'price_trend': price_trend,
        'popularity_filter': popularity_filter,
        'days_since_added': days_since_added,
        'competitive_level': competitive_level
    }

def create_performance_metrics(df):
    """
    Crea métricas de performance del dashboard
    """
    metrics = {}
    
    # Tiempo de carga de datos
    start_time = datetime.now()
    
    # Calcular métricas
    if not df.empty:
        metrics['data_quality_score'] = calculate_data_quality_score(df)
        metrics['market_diversity_index'] = calculate_market_diversity(df)
        metrics['price_competitiveness'] = calculate_price_competitiveness(df)
    
    end_time = datetime.now()
    metrics['processing_time'] = (end_time - start_time).total_seconds()
    
    return metrics

def calculate_data_quality_score(df):
    """
    Calcula score de calidad de datos
    """
    if df.empty:
        return 0
    
    # Porcentaje de datos completos
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    
    # Consistencia de precios (no negativos, no extremos)
    price_consistency = 100
    if 'precio_num' in df.columns:
        invalid_prices = len(df[(df['precio_num'] <= 0) | (df['precio_num'] > 10000)])
        price_consistency = max(0, 100 - (invalid_prices / len(df)) * 100)
    
    # Score final
    quality_score = (completeness * 0.6 + price_consistency * 0.4)
    
    return round(quality_score, 1)

def calculate_market_diversity(df):
    """
    Calcula índice de diversidad del mercado
    """
    if df.empty or 'categoria' not in df.columns:
        return 0
    
    # Usar índice de Shannon
    from scipy.stats import entropy
    
    category_counts = df['categoria'].value_counts()
    diversity_index = entropy(category_counts) / np.log(len(category_counts))
    
    return round(diversity_index * 100, 1)

def calculate_price_competitiveness(df):
    """
    Calcula índice de competitividad de precios
    """
    if df.empty or 'precio_num' not in df.columns:
        return 0
    
    # Coeficiente de variación como medida de competitividad
    price_cv = df['precio_num'].std() / df['precio_num'].mean()
    competitiveness = min(100, price_cv * 50)  # Normalizar a 0-100
    
    return round(competitiveness, 1)