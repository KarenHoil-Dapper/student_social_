from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
from datetime import datetime
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
        
        # Preparar resultados para mostrar
        results = {
            'prepared_data': prepared_data,
            'predictions': predictions,
            'recommendations': recommendations,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        return render_template('stats.html', stats=stats_data)
    except Exception as e:
        return render_template('error.html', error=str(e))

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
    
    # Template base
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
    
    # Template index
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
    
    # Template encuesta
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
.form-select-lg, .form-control-lg {
    font-size: 1.1rem;
    padding: 0.75rem 1rem;
}

.card {
    box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.05);
    border: none;
}

.card-header {
    border-bottom: 3px solid rgba(255, 255, 255, 0.2);
}

.btn-success {
    background: linear-gradient(45deg, #28a745, #20c997);
    border: none;
    box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
}

.btn-success:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
}

.is-invalid {
    border-color: #dc3545 !important;
    animation: shake 0.5s;
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}
</style>
{% endblock %}'''
    
    # Template resultados
    results_template = '''{% extends "base.html" %}

{% block title %}Resultados - {{ super() }}{% endblock %}

{% block content %}
<div class="row">
    <div class="col-lg-10 mx-auto">
        <h2 class="text-center mb-4">
            <i class="fas fa-chart-pie"></i>
            Tus Resultados
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
                    <div class="col-md-3">
                        <strong>Actividad física:</strong> 
                        {{ "%.0f"|format(results.prepared_data.get('Physical_Activity_Hours', 0)) }}h/sem
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
                
                {% if results.predictions.get('cluster') is defined %}
                <div class="alert alert-info">
                    <i class="fas fa-users"></i> <strong>Perfil de Usuario:</strong> Grupo {{ results.predictions.cluster }}
                    <small class="d-block">Basado en patrones de comportamiento similares</small>
                </div>
                {% endif %}
            </div>
        </div>
        
        <!-- Indicadores Detallados -->
        <div class="card mb-4">
            <div class="card-header bg-info text-white">
                <h5 class="mb-0"><i class="fas fa-chart-line"></i> Indicadores Detallados</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6>Bienestar Psicológico</h6>
                        <ul class="list-unstyled">
                            <li>🔄 Cambios de humor: {{ results.prepared_data.get('Mood_Changes', 0) }}/5</li>
                            <li>👥 Comparación social: {{ results.prepared_data.get('Social_Comparison', 0) }}/5</li>
                            <li>📱 Nivel de FOMO: {{ results.prepared_data.get('FOMO_Level', 0) }}/5</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>Productividad</h6>
                        <ul class="list-unstyled">
                            <li>⏰ Procrastinación: {{ results.prepared_data.get('Procrastination', 0) }}/5</li>
                            <li>🎯 Concentración: {{ results.prepared_data.get('Concentration_Issues', 0) }}/5</li>
                            <li>📊 Impacto productividad: {{ results.prepared_data.get('Productivity_Impact', 0) }}/5</li>
                        </ul>
                    </div>
                </div>
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
        
        <div class="alert alert-info mt-4">
            <i class="fas fa-info-circle"></i>
            <strong>Nota:</strong> La puntuación de adicción se calcula automáticamente basándose en tus patrones de uso, 
            frecuencia de publicación, notificaciones, FOMO y otros indicadores.
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Template estadísticas
    stats_template = '''{% extends "base.html" %}

{% block title %}Estadísticas - {{ super() }}{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h2 class="text-center mb-4">
            <i class="fas fa-chart-bar"></i>
            Estadísticas del Sistema
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
    </div>
</div>
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