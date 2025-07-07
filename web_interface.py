from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import json
import os
import io
import base64
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from updated_survey_system import SocialMediaHealthPredictor  # Actualizado

app = Flask(__name__)
app.secret_key = 'social_media_health_predictor_2024'

# Instancia global del predictor
predictor = SocialMediaHealthPredictor()

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/survey')
def survey():
    """Página de encuesta"""
    questions = predictor.create_survey_questions()
    return render_template('survey.html', questions=questions)

@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    """Procesa la encuesta enviada"""
    try:
        # Obtener datos del formulario
        responses = {}
        for key, value in request.form.items():
            try:
                # Convertir a float si es posible
                responses[key] = float(value)
            except ValueError:
                responses[key] = value
        
        # Agregar timestamp
        responses['survey_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        responses['survey_id'] = f"web_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Preparar características usando el nuevo método
        prepared_data = predictor.prepare_features_for_model(responses)
        
        # Hacer predicciones
        predictions = predictor.make_predictions(prepared_data)
        
        # Generar recomendaciones
        recommendations = predictor.generate_recommendations(prepared_data, predictions)
        
        # Guardar datos
        predictor.save_survey_data(prepared_data, predictions, recommendations)
        
        # Generar visualizaciones
        horizontal_personal = generate_personal_horizontal(prepared_data, predictions)
        comparison_chart = generate_comparison_chart(prepared_data)
        heatmap_chart = generate_heatmap(prepared_data)
        
        # Preparar resultados para mostrar
        results = {
            'prepared_data': prepared_data,
            'predictions': predictions,
            'recommendations': recommendations,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'horizontal_personal': horizontal_personal,
            'comparison_chart': comparison_chart,
            'heatmap_chart': heatmap_chart,
        }
        
        return render_template('results.html', results=results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para predicciones"""
    try:
        data = request.get_json()
        
        # Preparar características usando el nuevo método
        prepared_data = predictor.prepare_features_for_model(data)
        
        # Hacer predicciones
        predictions = predictor.make_predictions(prepared_data)
        
        # Generar recomendaciones
        recommendations = predictor.generate_recommendations(prepared_data, predictions)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'recommendations': recommendations,
            'addiction_score_calculated': prepared_data.get('Addicted_Score'),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats')
def stats():
    """Página de estadísticas"""
    try:
        # Cargar estadísticas del sistema
        stats_data = get_system_stats()
        
        # Generar gráficas de estadísticas
        global_horizontal = generate_global_horizontal()
        trends_chart = generate_trends_chart()
        
        stats_data['global_horizontal'] = global_horizontal
        stats_data['trends_chart'] = trends_chart
        
        return render_template('stats.html', stats=stats_data)
    except Exception as e:
        return render_template('error.html', error=str(e))

def generate_heatmap(prepared_data):
    """Genera un mapa de calor de correlaciones entre métricas de salud mental"""
    try:
        # Configuracion de estilo
        plt.style.use('default')
        sns.set_palette("viridis")
        
        # Métricas para el mapa de calor
        metrics_data = {
            'Uso Diario': prepared_data.get('Avg_Daily_Usage_Hours', 0),
            'Ansiedad': prepared_data.get('Anxiety_Level', 0),
            'Depresión': prepared_data.get('Depression_Level', 0),
            'FOMO': prepared_data.get('FOMO_Level', 0),
            'Adicción': prepared_data.get('Addicted_Score', 0),
            'Concentración': prepared_data.get('Concentration_Issues', 0),
            'Procrastinación': prepared_data.get('Procrastination', 0),
            'Cambios Humor': prepared_data.get('Mood_Changes', 0),
            'Comparación Social': prepared_data.get('Social_Comparison', 0),
            'Autoestima': prepared_data.get('Self_Esteem', 0),
            'Sueño': prepared_data.get('Sleep_Quality', 0),
            'Validación': prepared_data.get('Validation_Seeking', 0)
        }
        
        # Crear matriz de correlaciones simulada basada en los datos del usuario
        # Esto simula cómo cada métrica se relaciona con las demás
        metrics_names = list(metrics_data.keys())
        n_metrics = len(metrics_names)
        
        # Crear matriz de correlaciones basada en patrones conocidos
        correlation_matrix = np.zeros((n_metrics, n_metrics))
        
        # Rellenar diagonal principal con 1s
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Definir correlaciones conocidas entre métricas de salud mental
        correlations = {
            ('Ansiedad', 'Depresión'): 0.75,
            ('Ansiedad', 'FOMO'): 0.68,
            ('Ansiedad', 'Adicción'): 0.72,
            ('Depresión', 'Autoestima'): -0.65,
            ('Depresión', 'Sueño'): -0.58,
            ('FOMO', 'Comparación Social'): 0.81,
            ('FOMO', 'Validación'): 0.77,
            ('Adicción', 'Uso Diario'): 0.85,
            ('Adicción', 'Concentración'): 0.69,
            ('Adicción', 'Procrastinación'): 0.74,
            ('Concentración', 'Procrastinación'): 0.71,
            ('Comparación Social', 'Autoestima'): -0.63,
            ('Comparación Social', 'Validación'): 0.79,
            ('Cambios Humor', 'Ansiedad'): 0.67,
            ('Cambios Humor', 'Depresión'): 0.72,
            ('Sueño', 'Concentración'): -0.54,
            ('Uso Diario', 'Sueño'): -0.48,
            ('Validación', 'Autoestima'): -0.59
        }
        
        # Aplicar correlaciones a la matriz
        for i, metric1 in enumerate(metrics_names):
            for j, metric2 in enumerate(metrics_names):
                if i != j:
                    # Buscar correlación definida
                    correlation = None
                    for (m1, m2), corr in correlations.items():
                        if (metric1 == m1 and metric2 == m2) or (metric1 == m2 and metric2 == m1):
                            correlation = corr
                            break
                    
                    if correlation is not None:
                        correlation_matrix[i][j] = correlation
                    else:
                        # Correlación base pequeña entre métricas no relacionadas
                        correlation_matrix[i][j] = np.random.uniform(-0.2, 0.3)
        
        # Ajustar correlaciones basadas en los valores reales del usuario
        user_values = np.array(list(metrics_data.values()))
        user_values_normalized = user_values / np.max(user_values) if np.max(user_values) > 0 else user_values
        
        # Intensificar correlaciones si los valores del usuario son altos
        intensity_factor = np.mean(user_values_normalized)
        correlation_matrix = correlation_matrix * (0.7 + intensity_factor * 0.3)
        
        # Asegurar que la matriz sea simétrica
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Crear la figura con diseño moderno
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Crear el mapa de calor con colores atractivos
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        
        # Usar colormap personalizado
        cmap = sns.diverging_palette(250, 10, n=200, center='light', as_cmap=True)
        
        # Generar el heatmap
        sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=False, 
                    cmap=cmap,
                    center=0,
                    square=True,
                    xticklabels=metrics_names,
                    yticklabels=metrics_names,
                    cbar_kws={"shrink": .8, "label": "Correlación"},
                    linewidths=0.5,
                    linecolor='white',
                    ax=ax)
        
        # Personalizar el diseño
        ax.set_title('Mapa de Calor: Correlaciones entre Métricas de Salud Mental', 
                    fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
        
        # Rotar etiquetas para mejor legibilidad
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        # Mejorar la apariencia
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Agregar subtitle con información
        plt.figtext(0.5, 0.02, 
                    'Valores positivos indican correlación directa • Valores negativos indican correlación inversa', 
                    ha='center', fontsize=10, style='italic', color='#7f8c8d')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error generando mapa de calor: {e}")
        return None

def generate_personal_horizontal(prepared_data, predictions):
    """Genera una grafica de barra horizontal personal"""
    try:
        # Configurar estilo
        plt.style.use('default')
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Seleccionar métricas principales
        metrics = {
            'Uso Diario (h)': prepared_data.get('Avg_Daily_Usage_Hours', 0),
            'Ansiedad': prepared_data.get('Anxiety_Level', 0),
            'FOMO': prepared_data.get('FOMO_Level', 0),
            'Adicción': prepared_data.get('Addicted_Score', 0),
            'Concentración': prepared_data.get('Concentration_Issues', 0),
            'Procrastinación': prepared_data.get('Procrastination', 0),
            'Cambios Humor': prepared_data.get('Mood_Changes', 0),
            'Comparación Social': prepared_data.get('Social_Comparison', 0)
        }
        
        # Preparar datos para la gráfica
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        # Definir colores según el nivel de riesgo
        colors = []
        for i, (label, value) in enumerate(metrics.items()):
            if 'Uso Diario' in label:
                # Para uso diario: verde < 4h, amarillo 4-8h, rojo > 8h
                if value < 4:
                    colors.append('#28a745')  # Verde
                elif value < 8:
                    colors.append('#ffc107')  # Amarillo
                else:
                    colors.append('#dc3545')  # Rojo
            else:
                # Para escalas de 1-5 y 1-10
                max_scale = 10 if label in ['Ansiedad', 'Adicción'] else 5
                percentage = value / max_scale
                
                if percentage < 0.4:
                    colors.append('#28a745')  # Verde (bajo riesgo)
                elif percentage < 0.7:
                    colors.append('#ffc107')  # Amarillo (riesgo medio)
                else:
                    colors.append('#dc3545')  # Rojo (alto riesgo)
        
        # Crear gráfica de barras horizontales
        bars = ax.barh(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Agregar valores al final de cada barra
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{value:.1f}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        # Configurar el gráfico
        ax.set_xlabel('Puntuación', fontsize=12, fontweight='bold')
        ax.set_title('Tu Perfil de Salud Mental - Barras de Riesgo', fontsize=14, fontweight='bold', pad=20)
        
        # Establecer límites del eje X
        max_value = max(values)
        ax.set_xlim(0, max_value * 1.2)
        
        # Agregar líneas de referencia
        ax.axvline(x=max_value * 0.4, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=max_value * 0.7, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Mejorar el diseño
        ax.grid(True, axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Invertir el orden de las etiquetas para que se lea mejor
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error generando gráfica de barras horizontales: {e}")
        return None

def generate_comparison_chart(prepared_data):
    """Genera gráfica de comparación con promedio"""
    try:
        # Cargar datos históricos si existen
        df = load_historical_data()
        
        if df is None or len(df) == 0:
            return None
        
        # Métricas a comparar
        comparison_metrics = ['Anxiety_Level', 'Addicted_Score', 'FOMO_Level', 'Concentration_Issues']
        
        # Calcular promedios
        averages = {}
        for metric in comparison_metrics:
            if metric in df.columns:
                averages[metric] = df[metric].mean()
            else:
                averages[metric] = 0
        
        # Datos del usuario actual
        user_data = {}
        for metric in comparison_metrics:
            user_data[metric] = prepared_data.get(metric, 0)
        
        # Crear gráfica
        fig, ax = plt.subplots(figsize=(8, 5))
        
        metrics_labels = ['Ansiedad', 'Adicción', 'FOMO', 'Concentración']
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        avg_values = [averages[metric] for metric in comparison_metrics]
        user_values = [user_data[metric] for metric in comparison_metrics]
        
        bars1 = ax.bar(x - width/2, avg_values, width, label='Promedio General', 
                      color='lightblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, user_values, width, label='Tu Puntuación',
                      color='orange', alpha=0.8)
        
        ax.set_xlabel('Métricas')
        ax.set_ylabel('Puntuación')
        ax.set_title('Tu Perfil vs Promedio General')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error generando gráfica de comparación: {e}")
        return None

def generate_global_horizontal():
    """Genera Grafica horizontal global de todas las respuestas"""
    try:
        df = load_historical_data()
        
        if df is None or len(df) < 5:  # Necesitamos al menos 5 respuestas
            return None
        
        # Seleccionar columnas numéricas relevantes
        numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Anxiety_Level', 'FOMO_Level', 
                       'Addicted_Score', 'Concentration_Issues', 'Procrastination',
                       'Mood_Changes', 'Social_Comparison']
        
        # Filtrar columnas que existen
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 3:
            return None
        
        # Crear matriz de correlación
        correlation_matrix = df[available_cols].corr()
        
        # Crear horizontal
        fig, ax = plt.subplots(figsize=(7,6))
        
        sns.horizontal(
            correlation_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .8},
            ax=ax
        )
        
        plt.title('Correlaciones entre Métricas de Salud Mental', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error generando horizontal global: {e}")
        return None

def generate_trends_chart():
    """Genera gráfica de tendencias temporales"""
    try:
        df = load_historical_data()
        
        if df is None or len(df) < 3:
            return None
        
        # Convertir fecha
        if 'survey_date' in df.columns:
            df['survey_date'] = pd.to_datetime(df['survey_date'], errors='coerce')
            df = df.dropna(subset=['survey_date'])
            df = df.sort_values('survey_date')
        else:
            return None
        
        # Crear gráfica de tendencias
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))
        
        # Gráfica 1: Salud Mental Promedio por día
        if 'Mental_Health_Score' in df.columns:
            daily_mental = df.groupby(df['survey_date'].dt.date)['Mental_Health_Score'].mean()
            ax1.plot(daily_mental.index, daily_mental.values, marker='o', color='green')
            ax1.set_title('Tendencia: Salud Mental Promedio')
            ax1.set_ylabel('Puntuación')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        
        # Gráfica 2: Nivel de Adicción
        if 'Addicted_Score' in df.columns:
            daily_addiction = df.groupby(df['survey_date'].dt.date)['Addicted_Score'].mean()
            ax2.plot(daily_addiction.index, daily_addiction.values, marker='s', color='red')
            ax2.set_title('Tendencia: Nivel de Adicción')
            ax2.set_ylabel('Puntuación')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        # Gráfica 3: Uso Diario Promedio
        if 'Avg_Daily_Usage_Hours' in df.columns:
            daily_usage = df.groupby(df['survey_date'].dt.date)['Avg_Daily_Usage_Hours'].mean()
            ax3.plot(daily_usage.index, daily_usage.values, marker='^', color='blue')
            ax3.set_title('Tendencia: Uso Diario (Horas)')
            ax3.set_ylabel('Horas')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
        
        # Gráfica 4: Ansiedad Promedio
        if 'Anxiety_Level' in df.columns:
            daily_anxiety = df.groupby(df['survey_date'].dt.date)['Anxiety_Level'].mean()
            ax4.plot(daily_anxiety.index, daily_anxiety.values, marker='d', color='orange')
            ax4.set_title('Tendencia: Nivel de Ansiedad')
            ax4.set_ylabel('Puntuación')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Tendencias Temporales de Métricas', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error generando gráfica de tendencias: {e}")
        return None

def load_historical_data():
    """Carga datos históricos"""
    try:
        if os.path.exists(predictor.data_file):
            df = pd.read_excel(predictor.data_file)
            return df
        return None
    except Exception as e:
        print(f"Error cargando datos históricos: {e}")
        return None

def get_system_stats():
    """Obtiene estadísticas del sistema"""
    try:
        import pandas as pd
        
        stats = {
            'total_surveys': 0,
            'avg_mental_health_score': 0,
            'academic_impact_percentage': 0,
            'most_common_issues': [],
            'recent_surveys': 0
        }
        
        if os.path.exists(predictor.data_file):
            df = pd.read_excel(predictor.data_file)
            stats['total_surveys'] = len(df)
            
            if 'Mental_Health_Score' in df.columns:
                stats['avg_mental_health_score'] = df['Mental_Health_Score'].mean()
            
            if 'Affects_Academic_Performance' in df.columns:
                stats['academic_impact_percentage'] = (df['Affects_Academic_Performance'].sum() / len(df)) * 100
            
            # Encuestas recientes (último mes)
            if 'survey_date' in df.columns:
                df['survey_date'] = pd.to_datetime(df['survey_date'], errors='coerce')
                recent_date = datetime.now() - pd.Timedelta(days=30)
                recent_surveys = df[df['survey_date'] >= recent_date]
                stats['recent_surveys'] = len(recent_surveys)
        
        return stats
        
    except Exception as e:
        print(f"Error obteniendo estadísticas: {e}")
        return {
            'total_surveys': 0,
            'avg_mental_health_score': 0,
            'academic_impact_percentage': 0,
            'most_common_issues': [],
            'recent_surveys': 0
        }

# Crear templates si no existen
def create_templates():
    """Crea los templates HTML necesarios"""
    
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # Template base (mismo que antes)
    base_template = '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Sistema de Evaluación de Salud Mental{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .bg-gradient { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-hover:hover { transform: translateY(-5px); transition: all 0.3s; }
        .result-card { border-left: 4px solid #007bff; }
        .recommendation { border-left: 3px solid #28a745; background-color: #f8f9fa; }
        .chart-container { max-width: 100%; overflow-x: auto; }
        .chart-image { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-gradient">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="fas fa-brain"></i> Evaluación Salud Mental
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="{{ url_for('index') }}">Inicio</a>
                <a class="nav-link" href="{{ url_for('survey') }}">Encuesta</a>
                <a class="nav-link" href="{{ url_for('stats') }}">Estadísticas</a>
            </div>
        </div>
    </nav>

    <main class="container mt-4">
        {% block content %}{% endblock %}
    </main>

    <footer class="bg-light mt-5 py-3">
        <div class="container text-center">
            <small class="text-muted">© 2024 Sistema de Evaluación de Salud Mental - IA</small>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
    
    # Template resultados actualizado con gráficas
    results_template = '''{% extends "base.html" %}

{% block title %}Resultados - {{ super() }}{% endblock %}

{% block content %}
<div class="row">
    <div class="col-lg-10 mx-auto">
        <h2 class="text-center mb-4">
            <i class="fas fa-chart-pie"></i>
            Tus Resultados con Análisis Visual
        </h2>
        
        <!-- Resumen -->
        <div class="card result-card mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0"><i class="fas fa-user"></i> Resumen Personal</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3">
                        <strong>Edad:</strong> 
                        {{ results.prepared_data.get('Age', 'N/A') }} años
                    </div>
                    <div class="col-md-3">
                        <strong>Uso diario:</strong> 
                        {{ "%.1f"|format(results.prepared_data.get('Avg_Daily_Usage_Hours', 0)) }} horas
                    </div>
                    <div class="col-md-3">
                        <strong>Horas de sueño:</strong> 
                        {{ "%.1f"|format(results.prepared_data.get('Sleep_Hours_Per_Night', 0)) }} horas
                    </div>
                    <div class="col-md-3">
                        <strong>Ansiedad:</strong> 
                        {{ "%.0f"|format(results.prepared_data.get('Anxiety_Level', 0)) }}/10
                    </div>
                </div>
                <hr>
                <div class="row">
                    <div class="col-md-3">
                        <strong>Adicción (calculada):</strong> 
                        <span class="badge 
                        {% if results.prepared_data.get('Addicted_Score', 0) >= 8 %}bg-danger
                        {% elif results.prepared_data.get('Addicted_Score', 0) >= 6 %}bg-warning
                        {% else %}bg-success{% endif %}">
                        {{ "%.1f"|format(results.prepared_data.get('Addicted_Score', 0)) }}/10
                        </span>
                    </div>
                    <div class="col-md-3">
                        <strong>FOMO:</strong> 
                        {{ "%.0f"|format(results.prepared_data.get('FOMO_Level', 0)) }}/5
                    </div>
                    <div class="col-md-3">
                        <strong>Concentración:</strong> 
                        {{ "%.0f"|format(results.prepared_data.get('Concentration_Issues', 0)) }}/5
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Predicciones -->
        <div class="card result-card mb-4">
            <div class="card-header bg-success text-white">
                <h5 class="mb-0"><i class="fas fa-brain"></i> Análisis IA</h5>
            </div>
            <div class="card-body">
                {% if results.predictions.get('mental_health_score') %}
                <div class="row mb-3">
                    <div class="col-md-6">
                        <h6>Puntuación de Salud Mental</h6>
                        <div class="progress">
                            {% set score = results.predictions.mental_health_score %}
                            <div class="progress-bar 
                                {% if score >= 7 %}bg-success
                                {% elif score >= 5 %}bg-warning
                                {% else %}bg-danger{% endif %}" 
                                style="width: {{ (score/10)*100 }}%">
                                {{ "%.1f"|format(score) }}/10
                            </div>
                        </div>
                        {% if score >= 8 %}
                        <small class="text-success">Excelente salud mental</small>
                        {% elif score >= 6 %}
                        <small class="text-warning">Buena salud mental</small>
                        {% elif score >= 4 %}
                        <small class="text-warning">Regular - considera mejoras</small>
                        {% else %}
                        <small class="text-danger">Preocupante - busca ayuda</small>
                        {% endif %}
                    </div>
                    <div class="col-md-6">
                        <h6>Impacto Académico</h6>
                        {% if results.predictions.get('affects_academic_performance') == 1 %}
                        <span class="badge bg-warning fs-6">SÍ afecta</span>
                        <small class="d-block text-muted">
                            {{ "%.1f"|format(results.predictions.get('academic_impact_probability', 0)*100) }}% probabilidad
                        </small>
                        {% else %}
                        <span class="badge bg-success fs-6">NO afecta</span>
                        <small class="d-block text-muted">
                            {{ "%.1f"|format(results.predictions.get('academic_impact_probability', 0)*100) }}% probabilidad
                        </small>
                        {% endif %}
                    </div>
                </div>
                {% endif %}
                
                <!-- Gráficas Desplegables con Details/Summary -->
                {% if results.horizontal_personal or results.comparison_chart or results.heatmap_chart %}
                <div class="mt-3">
                    <details class="chart-details">
                        <summary class="chart-summary">
                            <i class="fas fa-chart-bar me-2"></i>
                            <strong>Ver Análisis Visual de mis Resultados</strong>
                            <i class="fas fa-chevron-down ms-2 summary-icon"></i>
                        </summary>
                        
                        <div class="charts-content mt-4">
                            <div class="row">
                                <!-- Grafica Horizontal -->
                                {% if results.horizontal_personal %}
                                <div class="col-lg-6 mb-4">
                                    <div class="card h-100 border-0 shadow-sm">
                                        <div class="card-header bg-danger text-white">
                                            <h6 class="mb-0">
                                                <i class="fas fa-fire"></i> 
                                                Tu Grafica Horizontal
                                            </h6>
                                        </div>
                                        <div class="card-body text-center">
                                            <div class="chart-container">
                                                <img src="data:image/png;base64,{{ results.horizontal_personal }}" 
                                                     class="chart-image" alt="Grafica horizontal Personal"
                                                     style="max-height: 300px; width: 100%; object-fit: contain;">
                                            </div>
                                            <small class="text-muted mt-2 d-block">
                                                <strong>Interpretación:</strong> Verde = Bajo riesgo, Amarillo = Medio, Rojo = Alto riesgo
                                            </small>
                                        </div>
                                    </div>
                                </div>
                                {% endif %}

                                <!-- Gráfica de Comparación -->
                                {% if results.comparison_chart %}
                                <div class="col-lg-6 mb-4">
                                    <div class="card h-100 border-0 shadow-sm">
                                        <div class="card-header bg-info text-white">
                                            <h6 class="mb-0">
                                                <i class="fas fa-chart-bar"></i> 
                                                Comparación vs Otros Usuarios
                                            </h6>
                                        </div>
                                        <div class="card-body text-center">
                                            <div class="chart-container">
                                                <img src="data:image/png;base64,{{ results.comparison_chart }}" 
                                                     class="chart-image" alt="Gráfica de Comparación"
                                                     style="max-height: 300px; width: 100%; object-fit: contain;">
                                            </div>
                                            <small class="text-muted mt-2 d-block">
                                                <strong>Comparación:</strong> Azul = Promedio, Naranja = Tu puntuación
                                            </small>
                                        </div>
                                    </div>
                                </div>
                                {% endif %}
                            </div>
                            
                            <!-- Panel de Interpretación Detallada -->
                            <div class="card border-0 bg-light">
                                <div class="card-body">
                                    <h6 class="text-center mb-3">
                                        <i class="fas fa-info-circle text-primary"></i> 
                                        Guía de Interpretación
                                    </h6>
                                    <div class="row">
                                        <div class="col-md-6">
                                            <h6><i class="fas fa-chart-bar text-primary"></i> Barras de Riesgo</h6>
                                            <ul class="small mb-0">
                                                <li><span class="badge bg-success">Verde</span> = Bajo riesgo (saludable)</li>
                                                <li><span class="badge bg-warning">Amarillo</span> = Riesgo moderado</li>
                                                <li><span class="badge bg-danger">Rojo</span> = Alto riesgo (requiere atención)</li>
                                            </ul>
                                        </div>
                                        <div class="col-md-6">
                                            <h6><i class="fas fa-chart-bar text-info"></i> Gráfica de Comparación</h6>
                                            <ul class="small mb-0">
                                                <li><span class="badge bg-primary">Azul</span> = Promedio de todos los usuarios</li>
                                                <li><span class="badge bg-warning">Naranja</span> = Tu puntuación actual</li>
                                                <li>Barras más altas = Mayor nivel del problema</li>
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {% if results.heatmap_chart %}
                            <div class="col-12 mb-4">
                                <div class="card h-100 border-0 shadow-lg">
                                    <div class="card-header bg-gradient-purple text-white">
                                        <h6 class="mb-0 text-black">
                                            <i class="fas fa-fire"></i> 
                                            Mapa de Calor: Correlaciones Mentales
                                        </h6>
                                    </div>
                                    <div class="card-body text-center">
                                        <div class="chart-container">
                                            <img src="data:image/png;base64,{{ results.heatmap_chart }}" 
                                                class="chart-image-heatmap" alt="Mapa de Calor de Correlaciones"
                                                style="max-height: 500px; width: 100%; object-fit: contain;">
                                        </div>
                                        <div class="mt-3 p-3 bg-light rounded">
                                            <small class="text-muted">
                                                <h6 class="text-center mb-3">
                                                    <i class="fas fa-info-circle text-primary"></i> 
                                                    Guía de Interpretación
                                                </h6>
                                                <span class="badge bg-danger me-2">Rojo</span> = Correlación fuerte positiva (cuando una sube, la otra también)<br>
                                                <span class="badge bg-primary me-2">Azul</span> = Correlación fuerte negativa (cuando una sube, la otra baja)<br>
                                                <span class="badge bg-light text-dark">Blanco</span> = Sin correlación significativa
                                            </small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            {% endif %}
                            
                        </div>
                    </details>
                </div>
                {% endif %}
            </div>
        </div>
        
        <!-- Recomendaciones -->
        {% if results.recommendations %}
        <div class="card mb-4">
            <div class="card-header bg-warning text-dark">
                <h5 class="mb-0"><i class="fas fa-lightbulb"></i> Recomendaciones Personalizadas</h5>
            </div>
            <div class="card-body">
                {% for recommendation in results.recommendations %}
                <div class="recommendation p-3 mb-2 rounded border-start border-warning border-3">
                    {{ recommendation }}
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <div class="text-center">
            <a href="{{ url_for('survey') }}" class="btn btn-primary btn-lg">
                <i class="fas fa-redo"></i> Nueva Evaluación
            </a>
            <a href="{{ url_for('index') }}" class="btn btn-secondary btn-lg">
                <i class="fas fa-home"></i> Inicio
            </a>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Template estadísticas actualizado con gráficas
    stats_template = '''{% extends "base.html" %}

{% block title %}Estadísticas - {{ super() }}{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2 class="text-center mb-4">
            <i class="fas fa-chart-bar"></i>
            Estadísticas del Sistema con Análisis Visual
        </h2>
        
        <div class="row">
            <div class="col-md-3 mb-4">
                <div class="card text-center">
                    <div class="card-body">
                        <i class="fas fa-users fa-2x text-primary mb-2"></i>
                        <h3>{{ stats.total_surveys }}</h3>
                        <p class="text-muted">Total Encuestas</p>
                    </div>
                </div>
            </div>
            
            <div class="col-md-3 mb-4">
                <div class="card text-center">
                    <div class="card-body">
                        <i class="fas fa-heart fa-2x text-success mb-2"></i>
                        <h3>{{ "%.1f"|format(stats.avg_mental_health_score) }}</h3>
                        <p class="text-muted">Salud Mental Promedio</p>
                    </div>
                </div>
            </div>
            
            <div class="col-md-3 mb-4">
                <div class="card text-center">
                    <div class="card-body">
                        <i class="fas fa-graduation-cap fa-2x text-warning mb-2"></i>
                        <h3>{{ "%.1f"|format(stats.academic_impact_percentage) }}%</h3>
                        <p class="text-muted">Impacto Académico</p>
                    </div>
                </div>
            </div>
            
            <div class="col-md-3 mb-4">
                <div class="card text-center">
                    <div class="card-body">
                        <i class="fas fa-calendar fa-2x text-info mb-2"></i>
                        <h3>{{ stats.recent_surveys }}</h3>
                        <p class="text-muted">Último Mes</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Grafica horizontal Global -->
        {% if stats.global_horizontal %}
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0"><i class="fas fa-fire"></i> Grafica horizontal Global - Correlaciones</h5>
            </div>
            <div class="card-body text-center">
                <div class="chart-container">
                    <img src="data:image/png;base64,{{ stats.global_horizontal }}" 
                         class="chart-image" alt="Grafica horizontal Global">
                </div>
                <p class="mt-3 text-muted">
                    Correlaciones entre diferentes métricas de salud mental de todos los usuarios.
                    Los colores azules indican correlación positiva, los rojos correlación negativa.
                </p>
            </div>
        </div>
        {% endif %}
        
        <!-- Gráficas de Tendencias -->
        {% if stats.trends_chart %}
        <div class="card mb-4">
            <div class="card-header bg-success text-white">
                <h5 class="mb-0"><i class="fas fa-chart-line"></i> Tendencias Temporales</h5>
            </div>
            <div class="card-body text-center">
                <div class="chart-container">
                    <img src="data:image/png;base64,{{ stats.trends_chart }}" 
                         class="chart-image" alt="Gráficas de Tendencias">
                </div>
                <p class="mt-3 text-muted">
                    Evolución temporal de las métricas principales del sistema.
                </p>
            </div>
        </div>
        {% endif %}
        
        {% if not stats.global_horizontal and not stats.trends_chart %}
        <div class="alert alert-info text-center">
            <i class="fas fa-info-circle"></i>
            <strong>Nota:</strong> Se necesitan más datos para generar visualizaciones. 
            Las gráficas aparecerán cuando haya suficientes encuestas completadas.
        </div>
        {% endif %}
    </div>
</div>
{% endblock %}'''
    
    # Template index (mismo que antes)
    index_template = '''{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-lg-8 mx-auto text-center">
        <div class="bg-gradient text-white rounded p-5 mb-5">
            <h1 class="display-4 mb-3">
                <i class="fas fa-brain"></i>
                Evaluación de Salud Mental
            </h1>
            <p class="lead">
                Sistema inteligente para evaluar el impacto de las redes sociales 
                en tu salud mental y rendimiento académico
            </p>
            <a href="{{ url_for('survey') }}" class="btn btn-light btn-lg">
                <i class="fas fa-clipboard-list"></i> Iniciar Evaluación
            </a>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-4 mb-4">
        <div class="card card-hover h-100">
            <div class="card-body text-center">
                <i class="fas fa-chart-line fa-3x text-primary mb-3"></i>
                <h5 class="card-title">Análisis Inteligente</h5>
                <p class="card-text">
                    Utilizamos algoritmos de machine learning para analizar 
                    tu relación con las redes sociales
                </p>
            </div>
        </div>
    </div>
    
    <div class="col-md-4 mb-4">
        <div class="card card-hover h-100">
            <div class="card-body text-center">
                <i class="fas fa-lightbulb fa-3x text-warning mb-3"></i>
                <h5 class="card-title">Recomendaciones</h5>
                <p class="card-text">
                    Recibe consejos personalizados para mejorar tu bienestar 
                    digital y salud mental
                </p>
            </div>
        </div>
    </div>
    
    <div class="col-md-4 mb-4">
        <div class="card card-hover h-100">
            <div class="card-body text-center">
                <i class="fas fa-shield-alt fa-3x text-success mb-3"></i>
                <h5 class="card-title">Privacidad</h5>
                <p class="card-text">
                    Tus datos son anónimos y se utilizan solo para 
                    mejorar el sistema de predicción
                </p>
            </div>
        </div>
    </div>
    
</div>
{% endblock %}'''
    
    # Template survey (mismo que antes)
    survey_template = '''{% extends "base.html" %}

{% block title %}Encuesta - {{ super() }}{% endblock %}

{% block content %}
<div class="row">
    <div class="col-lg-8 mx-auto">
        <h2 class="text-center mb-4">
            <i class="fas fa-clipboard-list"></i>
            Encuesta de Evaluación (13 preguntas)
        </h2>
        
        <div class="alert alert-info">
            <i class="fas fa-info-circle"></i>
            <strong>Versión optimizada:</strong> Solo 13 preguntas clave con selecciones fáciles.
            Tu puntuación de adicción se calculará automáticamente.
        </div>
        
        <form method="POST" action="{{ url_for('submit_survey') }}" id="surveyForm">
            {% for category, category_questions in questions.items() %}
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0">
                        <i class="fas fa-list"></i>
                        {{ category.replace('_', ' ').title() }}
                    </h5>
                </div>
                <div class="card-body">
                    {% for key, question_data in category_questions.items() %}
                    <div class="mb-4">
                        <label for="{{ key }}" class="form-label fw-bold">
                            {{ question_data.question }}
                        </label>
                        
                        {% if question_data.type == 'select' %}
                            <select class="form-select form-select-lg" id="{{ key }}" name="{{ key }}" required>
                                <option value="">-- Selecciona una opción --</option>
                                {% for option in question_data.options %}
                                    <option value="{{ option.value }}">{{ option.label }}</option>
                                {% endfor %}
                            </select>
                        {% elif question_data.type == 'number' %}
                            <input type="number" 
                                   class="form-control form-control-lg" 
                                   id="{{ key }}" 
                                   name="{{ key }}" 
                                   min="{{ question_data.get('min', 1) }}"
                                   max="{{ question_data.get('max', 100) }}"
                                   step="1"
                                   placeholder="Ingresa un número entre {{ question_data.get('min', 1) }} y {{ question_data.get('max', 100) }}"
                                   required>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
            
            <div class="text-center mb-4">
                <button type="submit" class="btn btn-success btn-lg px-5">
                    <i class="fas fa-brain"></i> Obtener Mi Evaluación Completa
                </button>
            </div>
            
            <div class="alert alert-secondary">
                <small>
                    <i class="fas fa-shield-alt"></i>
                    <strong>Privacidad:</strong> Tus respuestas son anónimas y se usan solo para mejorar el sistema.
                    <br>
                    <i class="fas fa-calculator"></i>
                    <strong>Adicción:</strong> Se calcula automáticamente basándose en tus patrones de uso.
                </small>
            </div>
        </form>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.getElementById('surveyForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Validar que todos los campos estén completos
    const requiredFields = this.querySelectorAll('[required]');
    let allCompleted = true;
    
    requiredFields.forEach(field => {
        if (!field.value) {
            allCompleted = false;
            field.classList.add('is-invalid');
        } else {
            field.classList.remove('is-invalid');
        }
    });
    
    if (!allCompleted) {
        alert('Por favor completa todas las preguntas antes de continuar.');
        return;
    }
    
    // Mostrar loading
    const submitBtn = this.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analizando con IA...';
    submitBtn.disabled = true;
    
    // Agregar clase de carga al formulario
    this.style.opacity = '0.7';
    
    // Enviar formulario después de animación
    setTimeout(() => {
        this.submit();
    }, 1500);
});

// Mejorar UX con animaciones en selects
document.querySelectorAll('select').forEach(select => {
    select.addEventListener('change', function() {
        this.classList.add('border-success');
        setTimeout(() => {
            this.classList.remove('border-success');
        }, 1000);
    });
});
</script>

<style>
<style>
/* Estilos para Details/Summary */
.chart-details {
    width: 100%;
}

.chart-summary {
    font-size: 1.1rem;
    padding: 15px 20px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border: 2px solid #dee2e6;
    border-radius: 8px;
    cursor: pointer;
    user-select: none;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0;
    list-style: none;
}

.chart-summary:hover {
    background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
    border-color: #007bff;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,123,255,0.1);
}

.chart-summary::-webkit-details-marker {
    display: none;
}

.chart-summary::marker {
    display: none;
}

.summary-icon {
    transition: transform 0.3s ease;
}

.chart-details[open] .summary-icon {
    transform: rotate(180deg);
}

.chart-details[open] .chart-summary {
    background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
    color: white;
    border-color: #007bff;
}

.charts-content {
    animation: slideDown 0.4s ease-out;
    border-left: 3px solid #007bff;
    padding-left: 15px;
    margin-left: 10px;
}

@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Mejoras para las gráficas */
.chart-image {
    border-radius: 6px;
    transition: transform 0.2s ease;
}

.chart-image:hover {
    transform: scale(1.02);
}

.card.shadow-sm {
    box-shadow: 0 0.125rem 0.5rem rgba(0, 0, 0, 0.1) !important;
    transition: box-shadow 0.2s ease;
}

.card.shadow-sm:hover {
    box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15) !important;
}

.bg-gradient-purple {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.chart-image-heatmap {
    border-radius: 8px;
    transition: transform 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.chart-image-heatmap:hover {
    transform: scale(1.01);
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}

.card.shadow-lg {
    box-shadow: 0 0.5rem 1.5rem rgba(0, 0, 0, 0.15) !important;
    transition: all 0.3s ease;
}

.card.shadow-lg:hover {
    box-shadow: 0 1rem 2rem rgba(0, 0, 0, 0.2) !important;
    transform: translateY(-2px);
}

/* Animación de entrada para el mapa de calor */
.chart-container {
    animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.bg-light.rounded {
    border-left: 4px solid #667eea;
    transition: border-left 0.3s ease;
}

.bg-light.rounded:hover {
    border-left: 6px solid #667eea;
}

/* Estilos para badges mejorados */
.badge {
    font-size: 0.75rem;
    padding: 0.375rem 0.75rem;
    border-radius: 0.375rem;
}

.badge.bg-danger {
    background: linear-gradient(135deg, #ff6b6b, #ee5a52) !important;
}

.badge.bg-primary {
    background: linear-gradient(135deg, #4facfe, #00f2fe) !important;
}

.badge.bg-light {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef) !important;
    border: 1px solid #dee2e6;
}
</style>
{% endblock %}'''
    
    # Template error
    error_template = '''{% extends "base.html" %}

{% block title %}Error - {{ super() }}{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-6 mx-auto text-center">
        <div class="card">
            <div class="card-body">
                <i class="fas fa-exclamation-triangle fa-3x text-danger mb-3"></i>
                <h3>Error</h3>
                <p class="text-muted">{{ error }}</p>
                <a href="{{ url_for('index') }}" class="btn btn-primary">
                    <i class="fas fa-home"></i> Volver al Inicio
                </a>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''

    # Escribir templates
    templates = {
        'base.html': base_template,
        'index.html': index_template,
        'survey.html': survey_template,
        'results.html': results_template,
        'stats.html': stats_template,
        'error.html': error_template
    }
    
    for filename, content in templates.items():
        with open(f'{templates_dir}/{filename}', 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"✅ Templates creados en {templates_dir}/")

def run_server():
    """Ejecuta el servidor web"""
    print("🌐 Iniciando servidor web...")
    
    # Crear templates si no existen
    create_templates()
    
    print("🚀 Servidor ejecutándose en: http://localhost:5000")
    print("📊 Estadísticas en: http://localhost:5000/stats")
    print("📝 Encuesta en: http://localhost:5000/survey")
    print("\n⚠️ Presiona Ctrl+C para detener el servidor")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    run_server()