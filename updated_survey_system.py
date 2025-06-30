import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json
import warnings
from supabase_conn import supabase
warnings.filterwarnings('ignore')

class SocialMediaHealthPredictor:
    """Sistema completo para encuestas, predicciones y actualización de modelos"""
    
    def __init__(self, model_path="model_advanced", data_file="Social_Bueno.xlsx"):
        self.model_path = model_path
        self.data_file = data_file
        self.models = {}
        self.selectors = {}
        self.features = {}
        self.model_info = {}
        self.load_models()
        
    def load_models(self):
        """Carga todos los modelos entrenados"""
        try:
            # Cargar información del modelo
            if os.path.exists(f"{self.model_path}/model_info.pkl"):
                self.model_info = joblib.load(f"{self.model_path}/model_info.pkl")
                print("📋 Información del modelo cargada")
            
            # Cargar modelos de clustering
            if os.path.exists(f"{self.model_path}/kmeans_advanced.pkl"):
                self.models['clustering'] = joblib.load(f"{self.model_path}/kmeans_advanced.pkl")
                self.models['scaler'] = joblib.load(f"{self.model_path}/scaler_advanced.pkl")
                print("✅ Modelo de clustering cargado")
            
            # Cargar modelos de regresión
            if os.path.exists(f"{self.model_path}/regression_best.pkl"):
                self.models['regression'] = joblib.load(f"{self.model_path}/regression_best.pkl")
                self.selectors['regression'] = joblib.load(f"{self.model_path}/regression_selector.pkl")
                self.features['regression'] = joblib.load(f"{self.model_path}/regression_features.pkl")
                print("✅ Modelo de regresión cargado")
            
            # Cargar modelos de clasificación
            if os.path.exists(f"{self.model_path}/classification_best.pkl"):
                self.models['classification'] = joblib.load(f"{self.model_path}/classification_best.pkl")
                self.selectors['classification'] = joblib.load(f"{self.model_path}/classification_selector.pkl")
                self.features['classification'] = joblib.load(f"{self.model_path}/classification_features.pkl")
                print("✅ Modelo de clasificación cargado")
                
        except Exception as e:
            print(f"⚠️ Error cargando modelos: {e}")
            print("💡 Asegúrate de entrenar los modelos primero ejecutando el script de entrenamiento")
    
    def create_survey_questions(self):
        """Define las preguntas de la encuesta optimizadas para interfaz web (12 preguntas)"""
        questions = {
            'informacion_personal': {
                'Age': {
                    'question': "¿Cuál es tu edad?",
                    'type': 'number',
                    'min': 13,
                    'max': 80
                },
                'Gender': {
                    'question': "¿Cuál es tu género?",
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'Femenino'},
                        {'value': 1, 'label': 'Masculino'},
                        {'value': 2, 'label': 'Otro'}
                    ]
                },
                'Academic_Level': {
                    'question': "¿Cuál es tu nivel académico?",
                    'type': 'select',
                    'options': [
                        {'value': 1, 'label': 'Secundaria'},
                        {'value': 2, 'label': 'Preparatoria/Bachillerato'},
                        {'value': 3, 'label': 'Universidad (Licenciatura)'},
                        {'value': 4, 'label': 'Posgrado (Maestría/Doctorado)'}
                    ]
                },
                'Country': {
                    'question': "¿En qué país vives?",
                    'type': 'select',
                    'options': [
                        {'value': 'Mexico', 'label': 'México'},
                        {'value': 'USA', 'label': 'Estados Unidos'},
                        {'value': 'Spain', 'label': 'España'},
                        {'value': 'Argentina', 'label': 'Argentina'},
                        {'value': 'Colombia', 'label': 'Colombia'},
                        {'value': 'Peru', 'label': 'Perú'},
                        {'value': 'Chile', 'label': 'Chile'},
                        {'value': 'Venezuela', 'label': 'Venezuela'},
                        {'value': 'Ecuador', 'label': 'Ecuador'},
                        {'value': 'Canada', 'label': 'Canadá'},
                        {'value': 'Other', 'label': 'Otro'}
                    ]
                }
            },
            'uso_y_patrones': {
                'Avg_Daily_Usage_Hours': {
                    'question': "¿Cuántas horas al día usas redes sociales?",
                    'type': 'select',
                    'options': [
                        {'value': 0.5, 'label': 'Menos de 1 hora'},
                        {'value': 1.5, 'label': '1-2 horas'},
                        {'value': 3, 'label': '2-4 horas'},
                        {'value': 5, 'label': '4-6 horas'},
                        {'value': 7, 'label': '6-8 horas'},
                        {'value': 9, 'label': 'Más de 8 horas'}
                    ]
                },
                'Main_Social_Platform': {
                    'question': "¿Cuál es tu red social principal?",
                    'type': 'select',
                    'options': [
                        {'value': 'Instagram', 'label': 'Instagram'},
                        {'value': 'TikTok', 'label': 'TikTok'},
                        {'value': 'Facebook', 'label': 'Facebook'},
                        {'value': 'YouTube', 'label': 'YouTube'},
                        {'value': 'WhatsApp', 'label': 'WhatsApp'},
                        {'value': 'Twitter', 'label': 'Twitter/X'},
                        {'value': 'Snapchat', 'label': 'Snapchat'},
                        {'value': 'LinkedIn', 'label': 'LinkedIn'}
                    ]
                },
                'Sleep_Hours_Per_Night': {
                    'question': "¿Cuántas horas duermes por noche?",
                    'type': 'select',
                    'options': [
                        {'value': 4, 'label': '4 horas o menos'},
                        {'value': 5, 'label': '5 horas'},
                        {'value': 6, 'label': '6 horas'},
                        {'value': 7, 'label': '7 horas'},
                        {'value': 8, 'label': '8 horas'},
                        {'value': 9, 'label': '9 horas o más'}
                    ]
                }
            },
            'bienestar_mental': {
                'anxiety_level': {
                    'question': "¿Qué tan ansioso/a te sientes generalmente?",
                    'type': 'select',
                    'options': [
                        {'value': 1, 'label': '1 - Nada ansioso'},
                        {'value': 2, 'label': '2 - Poco ansioso'},
                        {'value': 3, 'label': '3 - Algo ansioso'},
                        {'value': 4, 'label': '4 - Moderadamente ansioso'},
                        {'value': 5, 'label': '5 - Bastante ansioso'},
                        {'value': 6, 'label': '6 - Muy ansioso'},
                        {'value': 7, 'label': '7 - Extremadamente ansioso'}
                    ]
                },
                'social_comparison': {
                    'question': "¿Te comparas con otros en redes sociales?",
                    'type': 'select',
                    'options': [
                        {'value': 1, 'label': 'Nunca'},
                        {'value': 2, 'label': 'Rara vez'},
                        {'value': 3, 'label': 'A veces'},
                        {'value': 4, 'label': 'Frecuentemente'},
                        {'value': 5, 'label': 'Siempre'}
                    ]
                },
                'mood_changes': {
                    'question': "¿Las redes sociales afectan tu estado de ánimo?",
                    'type': 'select',
                    'options': [
                        {'value': 1, 'label': 'Nunca'},
                        {'value': 2, 'label': 'Rara vez'},
                        {'value': 3, 'label': 'A veces'},
                        {'value': 4, 'label': 'Frecuentemente'},
                        {'value': 5, 'label': 'Siempre'}
                    ]
                }
            },
            'productividad_y_habitos': {
                'concentration_issues': {
                    'question': "¿Las redes sociales afectan tu concentración?",
                    'type': 'select',
                    'options': [
                        {'value': 1, 'label': 'Nada'},
                        {'value': 2, 'label': 'Poco'},
                        {'value': 3, 'label': 'Moderadamente'},
                        {'value': 4, 'label': 'Bastante'},
                        {'value': 5, 'label': 'Mucho'}
                    ]
                },
                'procrastination': {
                    'question': "¿Procrastinas debido a las redes sociales?",
                    'type': 'select',
                    'options': [
                        {'value': 1, 'label': 'Nunca'},
                        {'value': 2, 'label': 'Rara vez'},
                        {'value': 3, 'label': 'A veces'},
                        {'value': 4, 'label': 'Frecuentemente'},
                        {'value': 5, 'label': 'Siempre'}
                    ]
                },
                'Conflicts_Over_Social_Media': {
                    'question': "¿Has tenido conflictos por el uso de redes sociales?",
                    'type': 'select',
                    'options': [
                        {'value': 0, 'label': 'Nunca'},
                        {'value': 1, 'label': 'Rara vez'},
                        {'value': 2, 'label': 'A veces'},
                        {'value': 3, 'label': 'Frecuentemente'},
                        {'value': 4, 'label': 'Siempre'}
                    ]
                }
            }
        }
        return questions
    
    def conduct_survey(self):
        """Realiza la encuesta interactiva por consola"""
        print("🎯 ENCUESTA DE SALUD MENTAL Y REDES SOCIALES")
        print("="*50)
        print("Por favor responde las siguientes preguntas.")
        print("Tus datos serán utilizados para mejorar nuestro sistema de predicción.\n")
        
        questions = self.create_survey_questions()
        responses = {}
        
        for category, category_questions in questions.items():
            print(f"\n📋 {category.replace('_', ' ').title()}")
            print("-" * 30)
            
            for key, question_data in category_questions.items():
                question_text = question_data['question']
                question_type = question_data['type']
                
                while True:
                    try:
                        if question_type == 'select':
                            print(f"\n{question_text}")
                            options = question_data['options']
                            for i, option in enumerate(options, 1):
                                print(f"  {i}. {option['label']}")
                            
                            choice = input(f"Selecciona una opción (1-{len(options)}): ").strip()
                            choice_idx = int(choice) - 1
                            
                            if 0 <= choice_idx < len(options):
                                if key == 'Country':
                                    responses[key] = options[choice_idx]['value']  # Para países guardamos el string
                                elif key == 'Main_Social_Platform':
                                    responses[key] = options[choice_idx]['value']  # Para plataformas guardamos el string
                                else:
                                    responses[key] = options[choice_idx]['value']  # Para números guardamos el value
                                break
                            else:
                                print("❌ Opción inválida. Intenta de nuevo.")
                                
                        elif question_type == 'number':
                            min_val = question_data.get('min', 1)
                            max_val = question_data.get('max', 100)
                            response = float(input(f"{question_text} ({min_val}-{max_val}): "))
                            
                            if min_val <= response <= max_val:
                                responses[key] = response
                                break
                            else:
                                print(f"❌ Por favor ingresa un valor entre {min_val} y {max_val}.")
                        
                    except (ValueError, IndexError):
                        print("❌ Por favor ingresa un valor válido.")
        
        # Agregar timestamp
        responses['survey_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        responses['survey_id'] = f"survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return responses
    
    def prepare_features_for_model(self, responses):
        """Convierte las respuestas en el formato que espera el modelo (optimizado para 12 preguntas)"""
        # Crear un registro completo con todas las columnas del dataset
        new_record = {}
        
        # Generar nuevo Student_ID
        try:
            df_existing = pd.read_excel(self.data_file)
            max_id = df_existing['Student_ID'].max() if 'Student_ID' in df_existing.columns else 0
            new_record['Student_ID'] = max_id + 1
        except:
            new_record['Student_ID'] = 1
        
        # Función helper para convertir valores de forma segura
        def safe_float(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    return float(value) if value.replace('.', '').replace('-', '').isdigit() else default
                else:
                    return default
            except:
                return default
        
        def safe_int(value, default=0):
            try:
                return int(float(value))
            except:
                return default
        
        # Información básica directa (con conversión segura)
        new_record['Age'] = safe_float(responses.get('Age'), 20)
        new_record['Gender'] = safe_int(responses.get('Gender'), 1)
        new_record['Avg_Daily_Usage_Hours'] = safe_float(responses.get('Avg_Daily_Usage_Hours'), 3)
        new_record['Sleep_Hours_Per_Night'] = safe_float(responses.get('Sleep_Hours_Per_Night'), 7)
        new_record['Conflicts_Over_Social_Media'] = safe_int(responses.get('Conflicts_Over_Social_Media'), 0)
        
        # Nuevas características de bienestar mental (directas)
        new_record['Anxiety_Level'] = safe_float(responses.get('anxiety_level'), 5)
        new_record['Mood_Changes'] = safe_float(responses.get('mood_changes'), 2)
        new_record['Social_Comparison'] = safe_float(responses.get('social_comparison'), 2)
        
        # Características de productividad (directas)
        new_record['Concentration_Issues'] = safe_float(responses.get('concentration_issues'), 2)
        new_record['Procrastination'] = safe_float(responses.get('procrastination'), 2)
        
        # CALCULAR características derivadas y estimadas
        usage_hours = new_record['Avg_Daily_Usage_Hours']
        concentration = new_record['Concentration_Issues']
        procrastination = new_record['Procrastination']
        anxiety = new_record['Anxiety_Level']
        mood_changes = new_record['Mood_Changes']
        social_comparison = new_record['Social_Comparison']
        
        # Estimar características no preguntadas pero necesarias para el modelo
        new_record['FOMO_Level'] = min(5.0, max(1.0, (social_comparison + anxiety) / 2))
        new_record['Productivity_Impact'] = min(5.0, max(1.0, (concentration + procrastination) / 2))
        new_record['Physical_Activity_Hours'] = max(0.0, 7 - usage_hours)  # Estimación inversa
        new_record['Face_to_Face_Preference'] = max(1.0, min(5.0, 6 - social_comparison))  # Estimación inversa
        new_record['Online_vs_Offline_Friends'] = 1 if social_comparison >= 4 else 0
        
        # Estimar patrones de uso basados en otras respuestas
        new_record['Platforms_Used'] = min(8.0, max(1.0, 2 + usage_hours / 2))
        new_record['Posting_Frequency'] = min(5.0, max(1.0, social_comparison))
        new_record['Scrolling_Before_Bed'] = min(5.0, max(1.0, usage_hours / 2 + anxiety / 3))
        new_record['Notification_Frequency'] = min(5.0, max(1.0, concentration))
        
        # CALCULAR AUTOMÁTICAMENTE EL ADDICTED_SCORE
        addiction_score = (
            (usage_hours * 1.0) +
            (new_record['Posting_Frequency'] * 0.8) +
            (new_record['Notification_Frequency'] * 0.7) +
            (new_record['FOMO_Level'] * 0.9) +
            (new_record['Scrolling_Before_Bed'] * 0.8) +
            (concentration * 0.6)
        ) / 6.0
        
        new_record['Addicted_Score'] = min(10.0, max(1.0, round(addiction_score, 1)))
        
        # Nivel académico (one-hot encoding) - CORREGIDO
        academic_level = safe_int(responses.get('Academic_Level'), 3)  # Por defecto universitario
        new_record['Academic_Level_High School'] = 1 if academic_level == 1 else 0
        new_record['Academic_Level_Undergraduate'] = 1 if academic_level == 3 else 0  # Licenciatura
        new_record['Academic_Level_Graduate'] = 1 if academic_level == 4 else 0  # Posgrado
        
        # Para preparatoria/bachillerato (valor 2), no activamos ninguna de las categorías del modelo
        # ya que el modelo original solo tiene: High School, Undergraduate, Graduate
        
        # Plataforma principal (one-hot encoding)
        platforms = ['Facebook', 'Instagram', 'KakaoTalk', 'LINE', 'LinkedIn', 
                    'Snapchat', 'TikTok', 'Twitter', 'VKontakte', 'WeChat', 
                    'WhatsApp', 'YouTube']
        
        # Inicializar todas las plataformas a 0
        for platform in platforms:
            new_record[f'Most_Used_Platform_{platform}'] = 0
        
        # Activar la plataforma seleccionada
        main_platform = responses.get('Main_Social_Platform', 'Instagram')
        if main_platform and main_platform in platforms:
            new_record[f'Most_Used_Platform_{main_platform}'] = 1
        else:
            new_record['Most_Used_Platform_Instagram'] = 1  # Default
        
        # Estado de relación (estimado como soltero por defecto)
        new_record['Relationship_Status_Single'] = 1
        new_record['Relationship_Status_In Relationship'] = 0
        new_record['Relationship_Status_Complicated'] = 0
        
        # País seleccionado (one-hot encoding)
        countries = ['Afghanistan', 'Albania', 'Andorra', 'Argentina', 'Armenia', 'Australia', 
                    'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 
                    'Belgium', 'Bhutan', 'Bolivia', 'Bosnia', 'Brazil', 'Bulgaria', 'Canada', 
                    'Chile', 'China', 'Colombia', 'Costa Rica', 'Croatia', 'Cyprus', 
                    'Czech Republic', 'Denmark', 'Ecuador', 'Egypt', 'Estonia', 'Finland', 
                    'France', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Hong Kong', 'Hungary', 
                    'Iceland', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Israel', 'Italy', 
                    'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 
                    'Kyrgyzstan', 'Latvia', 'Lebanon', 'Liechtenstein', 'Lithuania', 
                    'Luxembourg', 'Malaysia', 'Maldives', 'Malta', 'Mexico', 'Moldova', 
                    'Monaco', 'Montenegro', 'Morocco', 'Nepal', 'Netherlands', 'New Zealand', 
                    'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama', 
                    'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 
                    'Romania', 'Russia', 'San Marino', 'Serbia', 'Singapore', 'Slovakia', 
                    'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sweden', 
                    'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Trinidad', 
                    'Turkey', 'UAE', 'UK', 'USA', 'Ukraine', 'Uruguay', 'Uzbekistan', 
                    'Vatican City', 'Venezuela', 'Vietnam', 'Yemen']
        
        # Inicializar todos los países a 0
        for country in countries:
            new_record[f'Country_{country}'] = 0
        
        # Activar el país seleccionado
        selected_country = responses.get('Country', 'Mexico')
        if selected_country and selected_country != 'Other' and selected_country in countries:
            new_record[f'Country_{selected_country}'] = 1
        else:
            new_record['Country_Mexico'] = 1  # Default
        
        # Crear características derivadas adicionales que el modelo podría esperar
        if new_record['Avg_Daily_Usage_Hours'] > 0 and new_record['Sleep_Hours_Per_Night'] > 0:
            new_record['usage_sleep_ratio'] = new_record['Avg_Daily_Usage_Hours'] / (new_record['Sleep_Hours_Per_Night'] + 0.1)
            new_record['total_daily_activity'] = new_record['Avg_Daily_Usage_Hours'] + new_record['Sleep_Hours_Per_Night']
            new_record['sleep_deficit'] = max(0, 8 - new_record['Sleep_Hours_Per_Night'])
        
        # Categorías de uso
        new_record['usage_category_low'] = 1 if usage_hours <= 2 else 0
        new_record['usage_category_medium'] = 1 if 2 < usage_hours <= 6 else 0
        new_record['usage_category_high'] = 1 if usage_hours > 6 else 0
        
        # Índices combinados
        if anxiety > 0 and social_comparison > 0:
            new_record['mental_wellness_index'] = (anxiety + mood_changes + social_comparison + new_record['FOMO_Level']) / 4
            new_record['mental_risk_high'] = 1 if new_record['mental_wellness_index'] > 3.5 else 0
        
        if concentration > 0 and procrastination > 0:
            new_record['productivity_impact_index'] = (concentration + procrastination + new_record['Productivity_Impact']) / 3
        
        # Interacciones
        new_record['addiction_usage_interaction'] = new_record['Addicted_Score'] * usage_hours
        
        # Valores por defecto para columnas target (serán predichas)
        new_record['Mental_Health_Score'] = None  # Se predicirá
        new_record['Affects_Academic_Performance'] = None  # Se predicirá
        
        # Agregar metadatos
        new_record['survey_date'] = responses.get('survey_date')
        new_record['survey_id'] = responses.get('survey_id')
        
        print(f"🔍 Registro preparado con {len(new_record)} características")
        
        # IMPORTANTE: Retornar el registro creado
        return new_record
    
    def make_predictions(self, prepared_data):
        """Realiza predicciones usando los modelos entrenados"""
        predictions = {}
        
        # Verificar que prepared_data no sea None
        if prepared_data is None:
            print("❌ Error: prepared_data es None")
            return predictions
        
        try:
            # Verificar que tenemos información del modelo
            if not self.model_info or 'extended_features' not in self.model_info:
                print("⚠️ No hay información del modelo disponible")
                return predictions
            
            # Preparar características para el modelo
            extended_features = self.model_info['extended_features']
            available_features = []
            feature_values = []
            
            # Crear DataFrame con todas las características que el modelo espera
            for feature in extended_features:
                if feature in prepared_data and prepared_data[feature] is not None:
                    available_features.append(feature)
                    feature_values.append(float(prepared_data[feature]))
                else:
                    # Usar valor por defecto si la característica no está disponible
                    available_features.append(feature)
                    feature_values.append(0.0)  # Valor por defecto como float
            
            # Crear DataFrame para las predicciones
            X = pd.DataFrame([feature_values], columns=available_features)
            
            print(f"🔍 Preparando predicción con {len(available_features)} características")
            
            # 1. CLUSTERING
            if 'clustering' in self.models and 'scaler' in self.models:
                try:
                    X_scaled = self.models['scaler'].transform(X)
                    cluster = self.models['clustering'].predict(X_scaled)[0]
                    predictions['cluster'] = int(cluster)
                    print(f"✅ Clustering: Grupo {cluster}")
                except Exception as e:
                    print(f"⚠️ Error en clustering: {e}")
            
            # 2. REGRESIÓN (Mental Health Score)
            if ('regression' in self.models and 'regression' in self.selectors and 
                self.models['regression'] is not None):
                try:
                    # Aplicar selector de características
                    X_reg_selected = self.selectors['regression'].transform(X)
                    
                    # Obtener modelo de regresión
                    model_name, model, poly_features = self.models['regression']
                    
                    # Aplicar transformación polinomial si es necesaria
                    if poly_features is not None:
                        X_reg_selected = poly_features.transform(X_reg_selected)
                    
                    # Hacer predicción
                    mental_health_score = model.predict(X_reg_selected)[0]
                    predictions['mental_health_score'] = float(mental_health_score)
                    print(f"✅ Regresión: Salud mental {mental_health_score:.2f}")
                    
                except Exception as e:
                    print(f"⚠️ Error en regresión: {e}")
                    # Usar estimación básica como fallback
                    usage_hours = prepared_data.get('Avg_Daily_Usage_Hours', 3)
                    anxiety = prepared_data.get('Anxiety_Level', 5)
                    predictions['mental_health_score'] = max(1, min(10, 8 - usage_hours * 0.3 - anxiety * 0.2))
            
            # 3. CLASIFICACIÓN (Academic Performance Impact)
            if ('classification' in self.models and 'classification' in self.selectors and 
                self.models['classification'] is not None):
                try:
                    # Aplicar selector de características
                    X_clf_selected = self.selectors['classification'].transform(X)
                    
                    # Obtener modelo de clasificación
                    model_name, model = self.models['classification']
                    
                    # Hacer predicción
                    academic_impact = model.predict(X_clf_selected)[0]
                    academic_impact_proba = model.predict_proba(X_clf_selected)[0]
                    
                    predictions['affects_academic_performance'] = int(academic_impact)
                    predictions['academic_impact_probability'] = float(max(academic_impact_proba))
                    print(f"✅ Clasificación: Impacto académico {'Sí' if academic_impact else 'No'}")
                    
                except Exception as e:
                    print(f"⚠️ Error en clasificación: {e}")
                    # Usar estimación básica como fallback
                    usage_hours = prepared_data.get('Avg_Daily_Usage_Hours', 3)
                    concentration = prepared_data.get('Concentration_Issues', 2)
                    impact_prob = (usage_hours * 0.15 + concentration * 0.2) / 10
                    predictions['affects_academic_performance'] = 1 if impact_prob > 0.5 else 0
                    predictions['academic_impact_probability'] = float(impact_prob)
            
        except Exception as e:
            print(f"❌ Error general en predicciones: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: usar estimaciones simples
            usage_hours = prepared_data.get('Avg_Daily_Usage_Hours', 3)
            anxiety = prepared_data.get('Anxiety_Level', 5)
            concentration = prepared_data.get('Concentration_Issues', 2)
            
            predictions['mental_health_score'] = max(1, min(10, 8 - usage_hours * 0.3 - anxiety * 0.2))
            impact_prob = (usage_hours * 0.15 + concentration * 0.2) / 10
            predictions['affects_academic_performance'] = 1 if impact_prob > 0.5 else 0
            predictions['academic_impact_probability'] = float(impact_prob)
            predictions['cluster'] = 1  # Cluster por defecto
            predictions['error'] = "Usando estimaciones básicas por error en modelo"
        
        return predictions
    
    def generate_recommendations(self, prepared_data, predictions):
        """Genera recomendaciones personalizadas basadas en predicciones"""
        recommendations = []
        
        # Basado en horas de uso
        usage_hours = prepared_data.get('Avg_Daily_Usage_Hours', 0)
        if usage_hours > 6:
            recommendations.append("🚨 Tu uso de redes sociales es muy alto (>6h/día). Considera establecer límites de tiempo.")
        elif usage_hours > 4:
            recommendations.append("⚠️ Tu uso de redes sociales es considerable. Intenta reducir gradualmente el tiempo.")
        else:
            recommendations.append("✅ Tu uso de redes sociales parece estar en un rango saludable.")
        
        # Basado en sueño
        sleep_hours = prepared_data.get('Sleep_Hours_Per_Night', 7)
        scrolling_bed = prepared_data.get('Scrolling_Before_Bed', 2)
        if sleep_hours < 7:
            recommendations.append("😴 Necesitas más horas de sueño. Intenta dormir al menos 7-8 horas por noche.")
        if scrolling_bed >= 4:
            recommendations.append("📱 Evita usar redes sociales antes de dormir para mejorar la calidad del sueño.")
        elif sleep_hours > 9:
            recommendations.append("😴 Duermes más de lo normal. Considera revisar tu rutina de sueño.")
        
        # Basado en predicción de salud mental
        if 'mental_health_score' in predictions:
            score = predictions['mental_health_score']
            if score < 4:
                recommendations.append("🆘 Tu puntuación de salud mental es preocupante. Considera buscar ayuda profesional.")
            elif score < 6:
                recommendations.append("⚠️ Tu salud mental podría necesitar atención. Practica técnicas de mindfulness.")
            elif score >= 8:
                recommendations.append("😊 ¡Excelente salud mental! Mantén tus hábitos actuales.")
            else:
                recommendations.append("😊 Tu salud mental está en buen estado. Sigue cuidándote.")
        
        # Basado en impacto académico
        if 'affects_academic_performance' in predictions and predictions['affects_academic_performance'] == 1:
            recommendations.append("📚 Las redes sociales podrían estar afectando tu rendimiento académico. Considera usar apps de bloqueo durante estudio.")
        
        # Basado en puntuación de adicción CALCULADA
        addiction_score = prepared_data.get('Addicted_Score', 5)
        if addiction_score >= 8:
            recommendations.append("⚠️ Tu puntuación de adicción es alta. Considera técnicas de desintoxicación digital.")
        elif addiction_score >= 6:
            recommendations.append("🔄 Tu relación con las redes sociales podría mejorar. Intenta pausas regulares.")
        
        # Basado en ansiedad
        anxiety_level = prepared_data.get('Anxiety_Level', 5)
        if anxiety_level >= 8:
            recommendations.append("😰 Tu nivel de ansiedad es alto. Practica técnicas de relajación y considera reducir el uso de redes sociales.")
        elif anxiety_level >= 6:
            recommendations.append("🧘 Considera practicar mindfulness o meditación para manejar la ansiedad.")
        
        # Basado en comparación social
        social_comparison = prepared_data.get('Social_Comparison', 2)
        if social_comparison >= 4:
            recommendations.append("🤔 Te comparas frecuentemente con otros. Recuerda que las redes sociales no muestran la realidad completa.")
        
        # Basado en FOMO
        fomo_level = prepared_data.get('FOMO_Level', 2)
        if fomo_level >= 4:
            recommendations.append("📱 El miedo a perderte algo es alto. Practica el mindfulness y desconéctate regularmente.")
        
        # Basado en concentración
        concentration = prepared_data.get('Concentration_Issues', 2)
        if concentration >= 4:
            recommendations.append("🎯 Las redes sociales están afectando tu concentración. Usa técnicas de enfoque como Pomodoro.")
        
        # Basado en procrastinación
        procrastination = prepared_data.get('Procrastination', 2)
        if procrastination >= 4:
            recommendations.append("⏰ La procrastinación por redes sociales es alta. Establece horarios específicos para revisarlas.")
        
        # Basado en actividad física
        physical_activity = prepared_data.get('Physical_Activity_Hours', 3)
        if physical_activity < 2:
            recommendations.append("🏃‍♂️ Considera aumentar tu actividad física. El ejercicio mejora el bienestar mental.")
        
        # Basado en conflictos
        conflicts = prepared_data.get('Conflicts_Over_Social_Media', 0)
        if conflicts >= 3:
            recommendations.append("💬 Los conflictos por redes sociales son frecuentes. Practica comunicación digital saludable.")
        
        # Basado en notificaciones
        notifications = prepared_data.get('Notification_Frequency', 3)
        if notifications >= 4:
            recommendations.append("🔕 Considera reducir las notificaciones para disminuir distracciones y ansiedad.")
        
        # Basado en preferencia social
        face_to_face = prepared_data.get('Face_to_Face_Preference', 3)
        if face_to_face <= 2:
            recommendations.append("👥 Intenta equilibrar tus interacciones digitales con encuentros cara a cara.")
        
        return recommendations
    
    def save_survey_data(self, prepared_data, predictions, recommendations):
        """Prepara y guarda todos los campos requeridos en Supabase según la estructura completa"""
        try:
            supabase_data = {
                # Datos básicos
                "age": int(prepared_data.get('Age') or 0),
                "gender": int(prepared_data.get('Gender') or 0),
                "avg_daily_usage_hours": float(prepared_data.get('Avg_Daily_Usage_Hours') or 0),
                "affects_academic_performance": int(prepared_data.get('Affects_Academic_Performance') or 0),
                "sleep_hours_per_night": float(prepared_data.get('Sleep_Hours_Per_Night') or 0),
                "mental_health_score": float(predictions.get('mental_health_score') or 0),
                "conflicts_over_social_media": int(prepared_data.get('Conflicts_Over_Social_Media') or 0),
                "addicted_score": float(prepared_data.get('Addicted_Score') or 0),
            }

            # Nivel académico
            academic_levels = {
                "academic_level_graduate": 0,
                "academic_level_high_school": 0,
                "academic_level_undergraduate": 0
            }
            level = prepared_data.get('Academic_Level')
            if level == 4:
                academic_levels["academic_level_graduate"] = 1
            elif level == 1:
                academic_levels["academic_level_high_school"] = 1
            elif level == 3:
                academic_levels["academic_level_undergraduate"] = 1
            supabase_data.update(academic_levels)

            # Países
            country_fields = [
    "afghanistan", "albania", "andorra", "argentina", "armenia", "australia", "austria", "azerbaijan",
    "bahamas", "bahrain", "bangladesh", "belarus", "belgium", "bhutan", "bolivia", "bosnia", "brazil",
    "bulgaria", "canada", "chile", "china", "colombia", "costa_rica", "croatia", "cyprus",
    "czech_republic", "denmark", "ecuador", "egypt", "estonia", "finland", "france", "georgia",
    "germany", "ghana", "greece", "hong_kong", "hungary", "iceland", "india", "indonesia", "iraq",
    "ireland", "israel", "italy", "jamaica", "japan", "jordan", "kazakhstan", "kenya", "kosovo", 
    "kuwait", "kyrgyzstan", "latvia", "lebanon", "liechtenstein", "lithuania", "luxembourg", 
    "malaysia", "maldives", "malta", "mexico", "moldova", "monaco", "montenegro", "morocco",
    "nepal", "netherlands", "new_zealand", "nigeria", "north_macedonia", "norway", "oman", 
    "pakistan", "panama", "paraguay", "peru", "philippines", "poland", "portugal", "qatar",
    "romania", "russia", "san_marino", "serbia", "singapore", "slovakia", "slovenia", 
    "south_africa", "south_korea", "spain", "sri_lanka", "sweden", "switzerland", "syria",
    "taiwan", "tajikistan", "thailand", "trinidad", "turkey", "uae", "uk", "usa", "ukraine", 
    "uruguay", "uzbekistan", "vatican_city", "venezuela", "vietnam", "yemen"
]
            country_values = {f"country_{c}": 0 for c in country_fields}
            user_country = (prepared_data.get("Country") or "mexico").strip().replace(" ", "_").lower()
            selected_key = f"country_{user_country}"
            if selected_key in country_values:
                country_values[selected_key] = 1
            supabase_data.update(country_values)

            # Plataformas sociales
            platforms = [
                "facebook", "instagram", "kakaotalk", "line", "linkedin", "snapchat", "tiktok", 
                "twitter", "vkontakte", "wechat", "whatsapp", "youtube"
            ]
            platform_values = {f"most_used_platform_{p}": 0 for p in platforms}
            user_platform = (prepared_data.get("Main_Social_Platform") or "instagram").strip().lower()
            selected_platform_key = f"most_used_platform_{user_platform}"
            if selected_platform_key in platform_values:
                platform_values[selected_platform_key] = 1
            supabase_data.update(platform_values)

            # Estado de relación
            relationship_status = (prepared_data.get("Relationship_Status") or "Single").strip().lower()
            rel_fields = {
                "relationship_status_complicated": 1 if relationship_status == "complicated" else 0,
                "relationship_status_in_relationship": 1 if relationship_status == "in_relationship" else 0,
                "relationship_status_single": 1 if relationship_status == "single" else 0,
            }
            supabase_data.update(rel_fields)

            # Mostrar para depuración
            print("📄 JSON completo para Supabase:")
            print(json.dumps(supabase_data, indent=2, ensure_ascii=False))

            # Enviar a Supabase
            response = supabase.table("survey_results").insert(supabase_data).execute()

            if hasattr(response, "error") and response.error:
                print(f"❌ Error al guardar en Supabase: {response.error}")
            else:
                print("✅ Datos guardados correctamente en Supabase")

        except Exception as e:
            print(f"❌ Error guardando datos: {e}")
            import traceback
            traceback.print_exc()
            return False


    def display_results(self, prepared_data, predictions, recommendations):
        """Muestra los resultados al usuario de forma amigable"""
        print("\n" + "="*60)
        print("🎯 RESULTADOS DE TU EVALUACIÓN")
        print("="*60)
        
        # Información básica
        print(f"\n📊 RESUMEN DE TUS DATOS:")
        print(f"   👤 Edad: {prepared_data.get('Age', 'N/A')} años")
        print(f"   💻 Uso diario de redes sociales: {prepared_data.get('Avg_Daily_Usage_Hours', 'N/A')} horas")
        print(f"   😴 Horas de sueño: {prepared_data.get('Sleep_Hours_Per_Night', 'N/A')} horas")
        print(f"   📱 Puntuación de adicción (calculada): {prepared_data.get('Addicted_Score', 'N/A')}/10")
        print(f"   😰 Nivel de ansiedad: {prepared_data.get('Anxiety_Level', 'N/A')}/10")
        
        # Información adicional de bienestar
        print(f"\n🧠 INDICADORES DE BIENESTAR:")
        print(f"   🔄 Cambios de humor: {prepared_data.get('Mood_Changes', 'N/A')}/5")
        print(f"   👥 Comparación social: {prepared_data.get('Social_Comparison', 'N/A')}/5")
        print(f"   📱 Nivel de FOMO: {prepared_data.get('FOMO_Level', 'N/A')}/5")
        print(f"   🎯 Problemas de concentración: {prepared_data.get('Concentration_Issues', 'N/A')}/5")
        
        # Información de hábitos
        print(f"\n🔧 PATRONES DE USO:")
        print(f"   📝 Frecuencia de publicación: {prepared_data.get('Posting_Frequency', 'N/A')}/5")
        print(f"   🔔 Frecuencia de notificaciones: {prepared_data.get('Notification_Frequency', 'N/A')}/5")
        print(f"   🏃‍♂️ Actividad física semanal: {prepared_data.get('Physical_Activity_Hours', 'N/A')} horas")
        print(f"   ⏰ Procrastinación: {prepared_data.get('Procrastination', 'N/A')}/5")
        
        # Predicciones
        print(f"\n🔮 PREDICCIONES DEL MODELO:")
        if 'mental_health_score' in predictions:
            score = predictions['mental_health_score']
            print(f"   🧠 Puntuación de Salud Mental: {score:.2f}/10")
            if score >= 8:
                print("      ✅ Excelente salud mental")
            elif score >= 6:
                print("      😊 Buena salud mental")
            elif score >= 4:
                print("      ⚠️ Salud mental regular - considera mejoras")
            else:
                print("      🚨 Salud mental preocupante - busca ayuda")
        
        if 'affects_academic_performance' in predictions:
            impact = predictions['affects_academic_performance']
            probability = predictions.get('academic_impact_probability', 0)
            if impact == 1:
                print(f"   📚 Impacto Académico: SÍ afecta ({probability*100:.1f}% probabilidad)")
            else:
                print(f"   📚 Impacto Académico: NO afecta significativamente ({probability*100:.1f}% probabilidad)")
        
        if 'cluster' in predictions:
            cluster = predictions['cluster']
            print(f"   👥 Perfil de Usuario: Grupo {cluster}")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES PERSONALIZADAS:")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("   ✅ ¡Felicidades! Pareces tener hábitos saludables con las redes sociales.")
        
        # Información adicional
        print(f"\n📈 INFORMACIÓN ADICIONAL:")
        print(f"   🕒 Evaluación realizada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   🔄 Tus datos ayudarán a mejorar el modelo para futuras evaluaciones")
        print(f"   🆔 ID de encuesta: {prepared_data.get('survey_id', 'N/A')}")
        print(f"   🧮 Puntuación de adicción calculada automáticamente basada en tus respuestas")
        
        print("\n" + "="*60)
        print("¡Gracias por participar en nuestra evaluación!")
        print("="*60)
    
    def run_survey_session(self):
        """Ejecuta una sesión completa de encuesta"""
        try:
            print("🚀 Iniciando sistema de evaluación de salud mental y redes sociales...\n")
            
            # Verificar que los modelos estén cargados
            if not self.models:
                print("❌ No se pudieron cargar los modelos.")
                print("💡 Primero necesitas entrenar los modelos:")
                print("   python main_system.py --train")
                return
            
            # Realizar encuesta
            responses = self.conduct_survey()
            
            # Preparar datos para el modelo
            prepared_data = self.prepare_features_for_model(responses)
            
            # Hacer predicciones
            print("\n🔄 Analizando tus respuestas...")
            predictions = self.make_predictions(prepared_data)
            
            # Generar recomendaciones
            recommendations = self.generate_recommendations(prepared_data, predictions)
            
            # Mostrar resultados
            self.display_results(prepared_data, predictions, recommendations)
            
            # Guardar datos
            if self.save_survey_data(prepared_data, predictions, recommendations):
                print("\n✅ Tus datos han sido guardados para mejorar el sistema.")
                print(f"📊 Total de registros en el dataset: {len(pd.read_excel(self.data_file))}")
            
            # Preguntar si quiere otra evaluación
            while True:
                another = input("\n¿Quieres realizar otra evaluación? (s/n): ").strip().lower()
                if another in ['s', 'si', 'sí', 'y', 'yes']:
                    print("\n" + "="*60)
                    self.run_survey_session()
                    break
                elif another in ['n', 'no']:
                    print("\n¡Gracias por usar nuestro sistema! 🎉")
                    break
                else:
                    print("❌ Por favor responde 's' para sí o 'n' para no.")
            
        except KeyboardInterrupt:
            print("\n\n👋 Sesión cancelada por el usuario. ¡Hasta pronto!")
        except Exception as e:
            print(f"\n❌ Error durante la sesión: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Función principal"""
    print("🎯 SISTEMA DE EVALUACIÓN DE SALUD MENTAL Y REDES SOCIALES")
    print("="*60)
    print("Este sistema utiliza inteligencia artificial para evaluar el impacto")
    print("de las redes sociales en tu salud mental y rendimiento académico.")
    print("="*60)
    
    # Crear instancia del predictor
    predictor = SocialMediaHealthPredictor()
    
    # Ejecutar sesión de encuesta
    predictor.run_survey_session()

if __name__ == "__main__":
    main()