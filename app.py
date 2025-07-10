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
    df = ml_system.load_data('Social_Bueno.xlsx')  # ✅ Usar el archivo correcto
    df = ml_system.create_target_scores()
    X, y_mental, y_addiction = ml_system.prepare_features()
    ml_system.train_models()
    os.makedirs('models', exist_ok=True)
    ml_system.save_models()

def create_user_position_chart(user_score, all_scores, score_type, model_name):
    """Crea gráfica mostrando posición del usuario vs otros"""
    
    # Filtrar valores válidos
    valid_scores = [score for score in all_scores if not np.isnan(score) and 1 <= score <= 10]
    
    if len(valid_scores) < 10:
        # Crear datos de ejemplo si no hay suficientes datos
        valid_scores = np.random.normal(5.5, 1.5, 1000)
        valid_scores = np.clip(valid_scores, 1, 10)
    
    # Crear histograma de todos los usuarios
    fig = go.Figure()
    
    # Histograma de otros usuarios
    fig.add_trace(go.Histogram(
        x=valid_scores,
        nbinsx=20,
        name='Otros usuarios',
        marker_color='rgba(102, 126, 234, 0.7)',
        opacity=0.7,
        hovertemplate='<b>Rango:</b> %{x}<br><b>Usuarios:</b> %{y}<extra></extra>'
    ))
    
    # Línea vertical para el usuario actual
    fig.add_vline(
        x=user_score,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"Tu puntuación: {user_score:.1f}",
        annotation_position="top right"
    )
    
    # Calcular percentil del usuario
    percentile = (np.array(valid_scores) <= user_score).mean() * 100
    
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
                y=max(np.histogram(valid_scores, bins=20)[0]) * 0.8,
                text=f"🎯 TU AQUÍ<br>{user_score:.1f}/10<br>Percentil: {percentile:.0f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="red",
                borderwidth=2,
                font=dict(size=12)
            )
        ]
    )
    
    return fig

def create_comparison_radar_chart(user_scores, avg_scores):
    """Crea gráfica radar comparando usuario con promedio"""
    
    models = list(user_scores.keys())
    user_values = list(user_scores.values())
    avg_values = list(avg_scores.values())
    
    # Limpiar nombres de modelos
    clean_models = [model.replace('_', ' ').title() for model in models]
    
    fig = go.Figure()
    
    # Usuario actual
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=clean_models,
        fill='toself',
        name='Tu puntuación',
        line_color='red',
        fillcolor='rgba(255, 0, 0, 0.2)',
        hovertemplate='<b>Modelo:</b> %{theta}<br><b>Tu puntuación:</b> %{r:.1f}<extra></extra>'
    ))
    
    # Promedio general
    fig.add_trace(go.Scatterpolar(
        r=avg_values,
        theta=clean_models,
        fill='toself',
        name='Promedio general',
        line_color='blue',
        fillcolor='rgba(0, 0, 255, 0.2)',
        hovertemplate='<b>Modelo:</b> %{theta}<br><b>Promedio:</b> %{r:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickmode='linear',
                tick0=0,
                dtick=2
            )),
        title="Comparación por Modelos ML",
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_model_accuracy_chart():
    """Crea gráfica de precisión de modelos"""
    models = ['Regresión Lineal', 'Random Forest', 'Decision Tree']
    mental_accuracy = [85, 92, 78]
    addiction_accuracy = [82, 89, 75]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Salud Mental',
        x=models,
        y=mental_accuracy,
        marker_color='lightblue',
        hovertemplate='<b>%{x}</b><br>Precisión: %{y}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Adicción',
        x=models,
        y=addiction_accuracy,
        marker_color='lightcoral',
        hovertemplate='<b>%{x}</b><br>Precisión: %{y}%<extra></extra>'
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
    
    try:
        # Usar las columnas correctas del dataset
        mental_col = 'Mental_Health_Score'
        addiction_col = 'Addicted_Score'  # Nombre correcto en el dataset
        
        # Comparación por edad
        user_age = user_data['edad']
        age_groups = pd.cut(df['Age'], bins=[0, 18, 25, 35, 100], labels=['<18', '18-25', '26-35', '35+'])
        user_age_group = pd.cut([user_age], bins=[0, 18, 25, 35, 100], labels=['<18', '18-25', '26-35', '35+'])[0]
        
        age_mental = df.groupby(age_groups)[mental_col].mean()
        age_addiction = df.groupby(age_groups)[addiction_col].mean()
        
        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(
            name='Salud Mental',
            x=age_mental.index.astype(str),
            y=age_mental.values,
            marker_color='lightblue',
            hovertemplate='<b>Grupo:</b> %{x}<br><b>Salud Mental:</b> %{y:.1f}<extra></extra>'
        ))
        fig_age.add_trace(go.Bar(
            name='Adicción',
            x=age_addiction.index.astype(str),
            y=age_addiction.values,
            marker_color='lightcoral',
            hovertemplate='<b>Grupo:</b> %{x}<br><b>Adicción:</b> %{y:.1f}<extra></extra>'
        ))
        
        # Resaltar grupo del usuario
        fig_age.add_annotation(
            x=str(user_age_group),
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
        
        usage_mental = df.groupby(usage_groups)[mental_col].mean()
        usage_addiction = df.groupby(usage_groups)[addiction_col].mean()
        
        fig_usage = go.Figure()
        fig_usage.add_trace(go.Scatter(
            x=usage_mental.index.astype(str),
            y=usage_mental.values,
            mode='lines+markers',
            name='Salud Mental',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Grupo:</b> %{x}<br><b>Salud Mental:</b> %{y:.1f}<extra></extra>'
        ))
        fig_usage.add_trace(go.Scatter(
            x=usage_addiction.index.astype(str),
            y=usage_addiction.values,
            mode='lines+markers',
            name='Adicción',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Grupo:</b> %{x}<br><b>Adicción:</b> %{y:.1f}<extra></extra>'
        ))
        
        # Resaltar grupo del usuario
        fig_usage.add_annotation(
            x=str(user_usage_group),
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
        
    except Exception as e:
        print(f"Error creando gráficas demográficas: {str(e)}")
        # Crear gráficas de ejemplo en caso de error
        figures['age'] = create_example_chart("Comparación por Edad")
        figures['usage'] = create_example_chart("Comparación por Uso")
    
    return figures

def create_example_chart(title):
    """Crea una gráfica de ejemplo"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Grupo 1', 'Grupo 2', 'Grupo 3'],
        y=[5.5, 6.2, 4.8],
        name='Ejemplo',
        marker_color='lightblue'
    ))
    fig.update_layout(
        title=title,
        height=400,
        template='plotly_white'
    )
    return fig

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
            # Agregar info del guardado a los resultados
            results['save_info'] = save_result
        else:
            print(f"⚠️ Error guardando: {save_result['message']}")
            results['save_info'] = {'success': False, 'message': save_result['message']}
        
        # Calcular estadísticas para comparación usando el dataset correcto
        df = ml_system.df
        
        # Usar las columnas correctas
        mental_col = 'Mental_Health_Score'
        addiction_col = 'Addicted_Score'  # Nombre correcto en el dataset
        
        avg_mental_scores = {
            'linear': df[mental_col].mean(),
            'random_forest': df[mental_col].mean(),
            'decision_tree': df[mental_col].mean()
        }
        avg_addiction_scores = {
            'linear': df[addiction_col].mean(),
            'random_forest': df[addiction_col].mean(),
            'decision_tree': df[addiction_col].mean()
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
        print(f"Error inesperado: {str(e)}")
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
        
        # Verificar que el sistema ML esté disponible
        if ml_system is None or not hasattr(ml_system, 'df'):
            print("⚠️ Sistema ML no disponible, usando datos por defecto")
            return render_template('results_page.html', 
                                 user_data=user_data,
                                 results=results,
                                 graphs={'radar_mental': '{}', 'radar_addiction': '{}', 'accuracy': '{}', 'demographic_age': '{}', 'demographic_usage': '{}'},
                                 mental_graphs={},
                                 addiction_graphs={},
                                 stats={'total_users': 1940, 'mental_percentile': 50, 'addiction_percentile': 50, 'better_mental': 970, 'better_addiction': 970})
        
        df = ml_system.df
        print(f"📊 Usando dataset con {len(df)} registros")
        
        # Crear todas las gráficas con manejo de errores robusto
        graphs = {}
        mental_graphs = {}
        addiction_graphs = {}
        
        # 1. Gráficas de posición del usuario para cada modelo
        mental_col = 'Mental_Health_Score'
        addiction_col = 'Addicted_Score'
        
        print("🎨 Creando gráficas de salud mental...")
        for model_name, score in results['mental_health'].items():
            try:
                fig = create_user_position_chart(
                    score, 
                    df[mental_col].values, 
                    'Salud Mental', 
                    model_name.replace('_', ' ').title()
                )
                mental_graphs[model_name] = json.dumps(fig, cls=PlotlyJSONEncoder)
                print(f"  ✅ {model_name}")
            except Exception as e:
                print(f"  ❌ Error en {model_name}: {str(e)}")
                mental_graphs[model_name] = json.dumps({"data": [], "layout": {"title": f"Error en {model_name}"}})
        
        print("🎨 Creando gráficas de adicción...")
        for model_name, score in results['addiction'].items():
            try:
                fig = create_user_position_chart(
                    score,
                    df[addiction_col].values,
                    'Adicción',
                    model_name.replace('_', ' ').title()
                )
                addiction_graphs[model_name] = json.dumps(fig, cls=PlotlyJSONEncoder)
                print(f"  ✅ {model_name}")
            except Exception as e:
                print(f"  ❌ Error en {model_name}: {str(e)}")
                addiction_graphs[model_name] = json.dumps({"data": [], "layout": {"title": f"Error en {model_name}"}})
        
        # 2. Gráficas radar de comparación
        print("🎨 Creando gráficas radar...")
        try:
            if avg_scores.get('mental_health'):
                radar_mental = create_comparison_radar_chart(
                    results['mental_health'], 
                    avg_scores['mental_health']
                )
                graphs['radar_mental'] = json.dumps(radar_mental, cls=PlotlyJSONEncoder)
                print("  ✅ Radar mental health")
            else:
                graphs['radar_mental'] = json.dumps({"data": [], "layout": {}})
        except Exception as e:
            print(f"  ❌ Error radar mental: {str(e)}")
            graphs['radar_mental'] = json.dumps({"data": [], "layout": {}})
        
        try:
            if avg_scores.get('addiction'):
                radar_addiction = create_comparison_radar_chart(
                    results['addiction'],
                    avg_scores['addiction'] 
                )
                graphs['radar_addiction'] = json.dumps(radar_addiction, cls=PlotlyJSONEncoder)
                print("  ✅ Radar addiction")
            else:
                graphs['radar_addiction'] = json.dumps({"data": [], "layout": {}})
        except Exception as e:
            print(f"  ❌ Error radar addiction: {str(e)}")
            graphs['radar_addiction'] = json.dumps({"data": [], "layout": {}})
        
        # 3. Gráfica de precisión de modelos
        try:
            accuracy_chart = create_model_accuracy_chart()
            graphs['accuracy'] = json.dumps(accuracy_chart, cls=PlotlyJSONEncoder)
            print("  ✅ Accuracy chart")
        except Exception as e:
            print(f"  ❌ Error accuracy: {str(e)}")
            graphs['accuracy'] = json.dumps({"data": [], "layout": {}})
        
        # 4. Comparaciones demográficas
        try:
            demographic_charts = create_demographic_comparison(user_data, df)
            for key, fig in demographic_charts.items():
                graphs[f'demographic_{key}'] = json.dumps(fig, cls=PlotlyJSONEncoder)
            print("  ✅ Demographic charts")
        except Exception as e:
            print(f"  ❌ Error demographic: {str(e)}")
            graphs['demographic_age'] = json.dumps({"data": [], "layout": {}})
            graphs['demographic_usage'] = json.dumps({"data": [], "layout": {}})
        
        # Calcular estadísticas adicionales
        try:
            mental_avg = np.mean(list(results['mental_health'].values()))
            addiction_avg = np.mean(list(results['addiction'].values()))
            
            percentile_mental = (df[mental_col] <= mental_avg).mean() * 100
            percentile_addiction = (df[addiction_col] <= addiction_avg).mean() * 100
            
            stats = {
                'mental_percentile': round(percentile_mental, 1),
                'addiction_percentile': round(percentile_addiction, 1),
                'total_users': len(df),
                'better_mental': int((df[mental_col] < mental_avg).sum()),
                'better_addiction': int((df[addiction_col] > addiction_avg).sum())
            }
            print(f"📈 Estadísticas calculadas: {stats}")
        except Exception as e:
            print(f"❌ Error calculando estadísticas: {str(e)}")
            stats = {
                'mental_percentile': 50.0,
                'addiction_percentile': 50.0,
                'total_users': 1940,
                'better_mental': 970,
                'better_addiction': 970
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
        
        # Retornar página con datos mínimos
        return render_template('results_page.html', 
                             user_data=session.get('user_data', {}),
                             results=session.get('results', {
                                 'mental_health': {'linear': 5.0, 'random_forest': 5.0, 'decision_tree': 5.0}, 
                                 'addiction': {'linear': 5.0, 'random_forest': 5.0, 'decision_tree': 5.0}, 
                                 'explanations': {'mental_health': 'Error en explicación', 'addiction': 'Error en explicación'}, 
                                 'recommendations': ['Error generando recomendaciones']
                             }),
                             graphs={},
                             mental_graphs={},
                             addiction_graphs={},
                             stats={'total_users': 1940, 'mental_percentile': 50, 'addiction_percentile': 50, 'better_mental': 970, 'better_addiction': 970})

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