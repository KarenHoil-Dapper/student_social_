from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json
import os
import pandas as pd
import numpy as np
from social_media_ml_system import SocialMediaMLSystem
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from data_saver_system import save_user_prediction

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Cambiar en producción

# Inicializar sistema ML
ml_system = None
system_ready = False

def initialize_ml_system():
    """Inicializa el sistema ML con manejo de errores"""
    global ml_system, system_ready
    try:
        ml_system = SocialMediaMLSystem()
        
        if os.path.exists('models/linear_mental.pkl'):
            print("Cargando modelos existentes...")
            success = ml_system.load_models()
            if not success:
                print("Reentrenando modelos...")
                train_new_models()
        else:
            print("Entrenando nuevos modelos...")
            train_new_models()
            
        print("✅ Sistema ML inicializado correctamente")
        system_ready = True
        return True
        
    except Exception as e:
        print(f"❌ Error inicializando sistema ML: {str(e)}")
        system_ready = False
        return False

def train_new_models():
    """Entrena nuevos modelos"""
    global ml_system
    df = ml_system.load_data('Social_Bueno.xlsx')
    df = ml_system.create_target_scores()
    X, y_mental, y_addiction = ml_system.prepare_features()
    ml_system.train_models()
    os.makedirs('models', exist_ok=True)
    ml_system.save_models()

def create_user_position_chart(user_score, all_scores, score_type, model_name):
    """Crea gráfica mostrando posición del usuario vs otros"""
    
    # Crear histograma de todos los usuarios
    fig = go.Figure()
    
    # Histograma de otros usuarios
    fig.add_trace(go.Histogram(
        x=all_scores,
        nbinsx=20,
        name='Otros usuarios',
        marker_color='rgba(102, 126, 234, 0.7)',
        opacity=0.7
    ))
    
    # Línea vertical para el usuario actual
    fig.add_vline(
        x=user_score,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"Tu puntuación: {user_score:.1f}",
        annotation_position="top"
    )
    
    # Personalizar layout
    fig.update_layout(
        title=f'{score_type} - {model_name}<br>Tu posición vs otros usuarios',
        xaxis_title=f'Puntuación {score_type} (1-10)',
        yaxis_title='Número de usuarios',
        template='plotly_white',
        height=400,
        showlegend=True,
        annotations=[
            dict(
                x=user_score,
                y=max(np.histogram(all_scores, bins=20)[0]) * 0.8,
                text=f"🎯 TU AQUÍ<br>{user_score:.1f}/10",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=2
            )
        ]
    )
    
    return fig

def create_comparison_radar_chart(user_scores, avg_scores):
    """Crea gráfica radar comparando usuario con promedio"""
    
    models = list(user_scores.keys())
    user_values = list(user_scores.values())
    avg_values = list(avg_scores.values())
    
    fig = go.Figure()
    
    # Usuario actual
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=models,
        fill='toself',
        name='Tu puntuación',
        line_color='red',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    
    # Promedio general
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=models,
        fill='toself',
        name='Promedio general',
        line_color='blue',
        fillcolor='rgba(0, 0, 255, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        title="Comparación por Modelos ML",
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_model_accuracy_chart():
    """Crea gráfica de precisión de modelos"""
    models = ['Regresión Lineal', 'Random Forest', 'Decision Tree']
    mental_accuracy = [85, 92, 78]  # Valores ejemplo - ajustar con datos reales
    addiction_accuracy = [82, 89, 75]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Salud Mental',
        x=models,
        y=mental_accuracy,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Adicción',
        x=models,
        y=addiction_accuracy,
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Precisión de Modelos de Machine Learning',
        xaxis_title='Modelos',
        yaxis_title='Precisión (%)',
        barmode='group',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_demographic_comparison(user_data, df):
    """Crea gráficas de comparación demográfica"""
    figures = {}
    
    # Comparación por edad
    user_age = user_data['edad']
    age_groups = pd.cut(df['Age'], bins=[0, 18, 25, 35, 100], labels=['<18', '18-25', '26-35', '35+'])
    user_age_group = pd.cut([user_age], bins=[0, 18, 25, 35, 100], labels=['<18', '18-25', '26-35', '35+'])[0]
    
    age_mental = df.groupby(age_groups)['Mental_Health_Score'].mean()
    age_addiction = df.groupby(age_groups)['Addiction_Score'].mean()
    
    fig_age = go.Figure()
    fig_age.add_trace(go.Bar(
        name='Salud Mental',
        x=age_mental.index,
        y=age_mental.values,
        marker_color='lightblue'
    ))
    fig_age.add_trace(go.Bar(
        name='Adicción',
        x=age_addiction.index,
        y=age_addiction.values,
        marker_color='lightcoral'
    ))
    
    # Resaltar grupo del usuario
    fig_age.add_annotation(
        x=user_age_group,
        y=max(max(age_mental.values), max(age_addiction.values)) + 0.5,
        text="🎯 TU GRUPO",
        showarrow=True,
        arrowhead=2,
        bgcolor="yellow",
        bordercolor="orange"
    )
    
    fig_age.update_layout(
        title='Comparación por Grupo de Edad',
        xaxis_title='Grupo de Edad',
        yaxis_title='Puntuación Promedio',
        barmode='group',
        height=400,
        template='plotly_white'
    )
    
    figures['age'] = fig_age
    
    # Comparación por uso diario
    usage_groups = pd.cut(df['Avg_Daily_Usage_Hours'], 
                         bins=[0, 2, 4, 6, 24], 
                         labels=['Bajo (0-2h)', 'Moderado (2-4h)', 'Alto (4-6h)', 'Muy Alto (6h+)'])
    user_usage = user_data['uso_diario']
    user_usage_group = pd.cut([user_usage], 
                             bins=[0, 2, 4, 6, 24], 
                             labels=['Bajo (0-2h)', 'Moderado (2-4h)', 'Alto (4-6h)', 'Muy Alto (6h+)'])[0]
    
    usage_mental = df.groupby(usage_groups)['Mental_Health_Score'].mean()
    usage_addiction = df.groupby(usage_groups)['Addiction_Score'].mean()
    
    fig_usage = go.Figure()
    fig_usage.add_trace(go.Scatter(
        x=usage_mental.index,
        y=usage_mental.values,
        mode='lines+markers',
        name='Salud Mental',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    fig_usage.add_trace(go.Scatter(
        x=usage_addiction.index,
        y=usage_addiction.values,
        mode='lines+markers',
        name='Adicción',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Resaltar grupo del usuario
    fig_usage.add_annotation(
        x=user_usage_group,
        y=max(max(usage_mental.values), max(usage_addiction.values)) + 0.5,
        text="🎯 TU GRUPO",
        showarrow=True,
        arrowhead=2,
        bgcolor="yellow",
        bordercolor="orange"
    )
    
    fig_usage.update_layout(
        title='Tendencias por Nivel de Uso Diario',
        xaxis_title='Nivel de Uso',
        yaxis_title='Puntuación Promedio',
        height=400,
        template='plotly_white'
    )
    
    figures['usage'] = fig_usage
    
    return figures

# Inicializar al arrancar la app
initialize_ml_system()

@app.route('/')
def index():
    """Página principal con formulario mejorado"""
    return render_template('enhanced_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones y redirigir a resultados"""
    global ml_system, system_ready
    
    if not system_ready or ml_system is None:
        return jsonify({
            'success': False,
            'error': 'Sistema ML no está listo. Intenta recargar la página.'
        }), 500
    
    try:
        # Obtener datos del formulario
        user_data = {
            'edad': int(request.form['edad']),
            'genero': request.form['genero'],
            'uso_diario': int(request.form['uso_diario']),
            'horas_sueno': int(request.form['horas_sueno']),
            'conflictos': int(request.form['conflictos']),
            'afecta_academico': request.form['afecta_academico'],
            'nivel_academico': request.form['nivel_academico'],
            'plataforma_mas_usada': request.form['plataforma_mas_usada'],
            'estado_relacion': request.form['estado_relacion']
        }
        
        # Validar datos
        if not (13 <= user_data['edad'] <= 80):
            raise ValueError("La edad debe estar entre 13 y 80 años")
        if not (0 <= user_data['uso_diario'] <= 24):
            raise ValueError("Las horas de uso diario deben estar entre 0 y 24")
        if not (3 <= user_data['horas_sueno'] <= 12):
            raise ValueError("Las horas de sueño deben estar entre 3 y 12")
        if not (0 <= user_data['conflictos'] <= 10):
            raise ValueError("Los conflictos deben estar entre 0 y 10")
        
        # Realizar predicción
        results = ml_system.predict_user_profile(user_data)
        
        # 🆕 GUARDAR DATOS EN EXCEL
        save_result = save_user_prediction(user_data, results)
        if save_result['success']:
            print(f"✅ Usuario guardado como Student_ID: {save_result['user_info']['student_id']}")
        else:
            print(f"⚠️ Error guardando: {save_result['message']}")
        
        # Calcular estadísticas para comparación
        df = ml_system.df
        avg_mental_scores = {
            'linear': df['Mental_Health_Score'].mean(),
            'random_forest': df['Mental_Health_Score'].mean(),
            'decision_tree': df['Mental_Health_Score'].mean()
        }
        avg_addiction_scores = {
            'linear': df['Addiction_Score'].mean(),
            'random_forest': df['Addiction_Score'].mean(),
            'decision_tree': df['Addiction_Score'].mean()
        }
        
        # Guardar datos en sesión para la página de resultados
        session['user_data'] = user_data
        session['results'] = results
        session['avg_scores'] = {
            'mental_health': avg_mental_scores,
            'addiction': avg_addiction_scores
        }
        
        # Redirigir a página de resultados
        return redirect(url_for('results'))
        
    except ValueError as ve:
        return render_template('enhanced_form.html', error=str(ve))
    except Exception as e:
        return render_template('enhanced_form.html', error=f'Error del sistema: {str(e)}')

@app.route('/results')
def results():
    """Página de resultados con gráficas interactivas"""
    if 'results' not in session:
        return redirect(url_for('index'))
    
    try:
        user_data = session['user_data']
        results = session['results']
        avg_scores = session.get('avg_scores', {})
        
        # Crear todas las gráficas
        graphs = {}
        mental_graphs = {}
        addiction_graphs = {}
        
        # Verificar que el sistema ML esté disponible
        if ml_system is None or not hasattr(ml_system, 'df'):
            # Datos de ejemplo si no hay sistema ML
            return render_template('results_page.html', 
                                 user_data=user_data,
                                 results=results,
                                 graphs={},
                                 mental_graphs={},
                                 addiction_graphs={},
                                 stats={'total_users': 705, 'mental_percentile': 50, 
                                       'addiction_percentile': 50, 'better_mental': 350, 'better_addiction': 350})
        
        df = ml_system.df
        
        # 1. Gráficas de posición del usuario para cada modelo
        print("Creando gráficas de salud mental...")
        for model_name, score in results['mental_health'].items():
            try:
                fig = create_user_position_chart(
                    score, 
                    df['Mental_Health_Score'].values, 
                    'Salud Mental', 
                    model_name.replace('_', ' ').title()
                )
                mental_graphs[model_name] = json.dumps(fig, cls=PlotlyJSONEncoder)
                print(f"✅ Gráfica mental {model_name} creada")
            except Exception as e:
                print(f"❌ Error creando gráfica mental {model_name}: {str(e)}")
                mental_graphs[model_name] = json.dumps({"data": [], "layout": {}})
        
        print("Creando gráficas de adicción...")
        for model_name, score in results['addiction'].items():
            try:
                fig = create_user_position_chart(
                    score,
                    df['Addiction_Score'].values,
                    'Adicción',
                    model_name.replace('_', ' ').title()
                )
                addiction_graphs[model_name] = json.dumps(fig, cls=PlotlyJSONEncoder)
                print(f"✅ Gráfica adicción {model_name} creada")
            except Exception as e:
                print(f"❌ Error creando gráfica adicción {model_name}: {str(e)}")
                addiction_graphs[model_name] = json.dumps({"data": [], "layout": {}})
        
        # 2. Gráficas radar de comparación
        print("Creando gráficas radar...")
        try:
            if avg_scores.get('mental_health'):
                radar_mental = create_comparison_radar_chart(
                    results['mental_health'], 
                    avg_scores['mental_health']
                )
                graphs['radar_mental'] = json.dumps(radar_mental, cls=PlotlyJSONEncoder)
            else:
                graphs['radar_mental'] = json.dumps({"data": [], "layout": {}})
        except Exception as e:
            print(f"Error creando radar mental: {str(e)}")
            graphs['radar_mental'] = json.dumps({"data": [], "layout": {}})
        
        try:
            if avg_scores.get('addiction'):
                radar_addiction = create_comparison_radar_chart(
                    results['addiction'],
                    avg_scores['addiction'] 
                )
                graphs['radar_addiction'] = json.dumps(radar_addiction, cls=PlotlyJSONEncoder)
            else:
                graphs['radar_addiction'] = json.dumps({"data": [], "layout": {}})
        except Exception as e:
            print(f"Error creando radar adicción: {str(e)}")
            graphs['radar_addiction'] = json.dumps({"data": [], "layout": {}})
        
        # 3. Gráfica de precisión de modelos
        print("Creando gráfica de precisión...")
        try:
            accuracy_chart = create_model_accuracy_chart()
            graphs['accuracy'] = json.dumps(accuracy_chart, cls=PlotlyJSONEncoder)
        except Exception as e:
            print(f"Error creando gráfica de precisión: {str(e)}")
            graphs['accuracy'] = json.dumps({"data": [], "layout": {}})
        
        # 4. Comparaciones demográficas
        print("Creando gráficas demográficas...")
        try:
            demographic_charts = create_demographic_comparison(user_data, df)
            for key, fig in demographic_charts.items():
                graphs[f'demographic_{key}'] = json.dumps(fig, cls=PlotlyJSONEncoder)
        except Exception as e:
            print(f"Error creando gráficas demográficas: {str(e)}")
            graphs['demographic_age'] = json.dumps({"data": [], "layout": {}})
            graphs['demographic_usage'] = json.dumps({"data": [], "layout": {}})
        
        # Calcular estadísticas adicionales
        try:
            mental_avg = np.mean(list(results['mental_health'].values()))
            addiction_avg = np.mean(list(results['addiction'].values()))
            
            percentile_mental = (df['Mental_Health_Score'] <= mental_avg).mean() * 100
            percentile_addiction = (df['Addiction_Score'] <= addiction_avg).mean() * 100
            
            stats = {
                'mental_percentile': round(percentile_mental, 1),
                'addiction_percentile': round(percentile_addiction, 1),
                'total_users': len(df),
                'better_mental': round((df['Mental_Health_Score'] < mental_avg).sum()),
                'better_addiction': round((df['Addiction_Score'] > addiction_avg).sum())
            }
        except Exception as e:
            print(f"Error calculando estadísticas: {str(e)}")
            stats = {
                'mental_percentile': 50.0,
                'addiction_percentile': 50.0,
                'total_users': 705,
                'better_mental': 350,
                'better_addiction': 350
            }
        
        print("✅ Todas las gráficas creadas exitosamente")
        
        return render_template('results_page.html', 
                             user_data=user_data,
                             results=results,
                             graphs=graphs,
                             mental_graphs=mental_graphs,
                             addiction_graphs=addiction_graphs,
                             stats=stats)
    
    except Exception as e:
        print(f"❌ Error general en /results: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Página de error básica
        return render_template('results_page.html', 
                             user_data=session.get('user_data', {}),
                             results=session.get('results', {'mental_health': {}, 'addiction': {}, 'explanations': {}, 'recommendations': []}),
                             graphs={},
                             mental_graphs={},
                             addiction_graphs={},
                             stats={'total_users': 705, 'mental_percentile': 50, 'addiction_percentile': 50, 'better_mental': 350, 'better_addiction': 350})

@app.route('/about')
def about():
    """Página de información del sistema"""
    return render_template('about.html')

@app.route('/health')
def health():
    """Endpoint para verificar estado del sistema"""
    global ml_system, system_ready
    
    status = {
        'system_ready': system_ready,
        'ml_system_loaded': ml_system is not None,
        'models_available': {},
        'features_ready': False
    }
    
    if ml_system:
        status['models_available'] = {
            'linear': 'linear_mental' in ml_system.models and 'linear_addiction' in ml_system.models,
            'random_forest': 'rf_mental' in ml_system.models and 'rf_addiction' in ml_system.models,
            'decision_tree': 'dt_mental' in ml_system.models and 'dt_addiction' in ml_system.models
        }
        status['features_ready'] = hasattr(ml_system, 'feature_names') and bool(ml_system.feature_names)
        if status['features_ready']:
            status['feature_count'] = len(ml_system.feature_names)
    
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)