import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json
import warnings
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
        """Define las preguntas de la encuesta"""
        questions = {
            'personal_info': {
                'age': "¿Cuál es tu edad?",
                'gender': "¿Cuál es tu género? (1=Masculino, 2=Femenino, 3=Otro)",
                'education_level': "¿Cuál es tu nivel educativo? (1=Primaria, 2=Secundaria, 3=Preparatoria, 4=Universidad, 5=Posgrado)"
            },
            'usage_patterns': {
                'avg_daily_usage_hours': "¿Cuántas horas al día usas redes sociales en promedio?",
                'sleep_hours_per_night': "¿Cuántas horas duermes por noche normalmente?",
                'physical_activity_hours': "¿Cuántas horas de actividad física realizas por semana?",
                'study_work_hours': "¿Cuántas horas dedicas al estudio/trabajo por día?"
            },
            'social_media_behavior': {
                'platforms_used': "¿Cuántas plataformas de redes sociales usas regularmente?",
                'posting_frequency': "¿Con qué frecuencia publicas contenido? (1=Nunca, 2=Rara vez, 3=A veces, 4=Frecuentemente, 5=Muy frecuentemente)",
                'scrolling_before_bed': "¿Usas redes sociales antes de dormir? (1=Nunca, 2=Rara vez, 3=A veces, 4=Frecuentemente, 5=Siempre)",
                'notification_frequency': "¿Con qué frecuencia recibes notificaciones? (1=Muy pocas, 2=Pocas, 3=Normal, 4=Muchas, 5=Demasiadas)"
            },
            'psychological_indicators': {
                'anxiety_level': "Del 1 al 10, ¿qué tan ansioso/a te sientes generalmente?",
                'mood_changes': "¿Has notado cambios de humor relacionados con el uso de redes sociales? (1=Nunca, 2=Rara vez, 3=A veces, 4=Frecuentemente, 5=Siempre)",
                'social_comparison': "¿Te comparas con otros en redes sociales? (1=Nunca, 2=Rara vez, 3=A veces, 4=Frecuentemente, 5=Siempre)",
                'fomo_level': "¿Sientes miedo de perderte algo (FOMO)? (1=Nunca, 2=Rara vez, 3=A veces, 4=Frecuentemente, 5=Siempre)"
            },
            'academic_work_impact': {
                'concentration_issues': "¿Las redes sociales afectan tu concentración? (1=Nada, 2=Poco, 3=Moderadamente, 4=Bastante, 5=Mucho)",
                'procrastination': "¿Procrastinas debido a las redes sociales? (1=Nunca, 2=Rara vez, 3=A veces, 4=Frecuentemente, 5=Siempre)",
                'productivity_impact': "¿Cómo afectan las redes sociales tu productividad? (1=Positivamente, 2=No afectan, 3=Ligeramente negativo, 4=Moderadamente negativo, 5=Muy negativo)"
            },
            'social_relationships': {
                'conflicts_over_social_media': "¿Has tenido conflictos por el uso de redes sociales? (1=Nunca, 2=Rara vez, 3=A veces, 4=Frecuentemente, 5=Siempre)",
                'face_to_face_preference': "¿Prefieres interactuar cara a cara que por redes sociales? (1=Nunca, 2=Rara vez, 3=A veces, 4=Frecuentemente, 5=Siempre)",
                'online_vs_offline_friends': "¿Tienes más amigos online que offline? (0=No, 1=Sí)"
            }
        }
        return questions
    
    def conduct_survey(self):
        """Realiza la encuesta interactiva"""
        print("🎯 ENCUESTA DE SALUD MENTAL Y REDES SOCIALES")
        print("="*50)
        print("Por favor responde las siguientes preguntas honestamente.")
        print("Tus datos serán utilizados para mejorar nuestro sistema de predicción.\n")
        
        questions = self.create_survey_questions()
        responses = {}
        
        for category, category_questions in questions.items():
            print(f"\n📋 {category.replace('_', ' ').title()}")
            print("-" * 30)
            
            for key, question in category_questions.items():
                while True:
                    try:
                        if 'género' in question.lower() or 'gender' in question.lower():
                            response = input(f"{question}: ").strip()
                            responses[key] = int(response) if response.isdigit() else 1
                        else:
                            response = float(input(f"{question}: "))
                            responses[key] = response
                        break
                    except ValueError:
                        print("❌ Por favor ingresa un valor numérico válido.")
        
        # Agregar timestamp
        responses['survey_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        responses['survey_id'] = f"survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return responses
    
    def prepare_features(self, responses):
        """Prepara las características para predicción"""
        # Crear características derivadas como en el entrenamiento
        features = responses.copy()
        
        # Características derivadas básicas
        if 'avg_daily_usage_hours' in features and 'sleep_hours_per_night' in features:
            features['usage_sleep_ratio'] = features['avg_daily_usage_hours'] / (features['sleep_hours_per_night'] + 0.1)
            features['total_daily_activity'] = features['avg_daily_usage_hours'] + features['sleep_hours_per_night']
            features['sleep_deficit'] = max(0, 8 - features['sleep_hours_per_night'])
        
        # Categoría de uso
        if 'avg_daily_usage_hours' in features:
            usage_hours = features['avg_daily_usage_hours']
            if usage_hours <= 2:
                features['usage_category_encoded'] = 0
            elif usage_hours <= 4:
                features['usage_category_encoded'] = 1
            elif usage_hours <= 6:
                features['usage_category_encoded'] = 2
            else:
                features['usage_category_encoded'] = 3
        
        return features
    
    def make_predictions(self, features):
        """Realiza predicciones usando los modelos entrenados"""
        predictions = {}
        
        try:
            # Convertir características a DataFrame
            if 'extended_features' in self.model_info:
                available_features = []
                feature_values = []
                
                for feature in self.model_info['extended_features']:
                    if feature in features:
                        available_features.append(feature)
                        feature_values.append(features[feature])
                    else:
                        # Usar valor por defecto si la característica no está disponible
                        available_features.append(feature)
                        feature_values.append(0)  # Valor por defecto
                
                X = pd.DataFrame([feature_values], columns=available_features)
                
                # Clustering
                if 'clustering' in self.models and 'scaler' in self.models:
                    X_scaled = self.models['scaler'].transform(X)
                    cluster = self.models['clustering'].predict(X_scaled)[0]
                    predictions['cluster'] = int(cluster)
                
                # Regresión (Mental Health Score)
                if 'regression' in self.models and 'regression' in self.selectors:
                    X_reg_selected = self.selectors['regression'].transform(X)
                    
                    model_name, model, poly_features = self.models['regression']
                    if poly_features is not None:  # Modelo polinomial
                        X_reg_selected = poly_features.transform(X_reg_selected)
                    
                    mental_health_score = model.predict(X_reg_selected)[0]
                    predictions['mental_health_score'] = float(mental_health_score)
                
                # Clasificación (Academic Performance Impact)
                if 'classification' in self.models and 'classification' in self.selectors:
                    X_clf_selected = self.selectors['classification'].transform(X)
                    
                    model_name, model = self.models['classification']
                    academic_impact = model.predict(X_clf_selected)[0]
                    academic_impact_proba = model.predict_proba(X_clf_selected)[0]
                    
                    predictions['affects_academic_performance'] = int(academic_impact)
                    predictions['academic_impact_probability'] = float(max(academic_impact_proba))
            
        except Exception as e:
            print(f"⚠️ Error en predicciones: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    def generate_recommendations(self, features, predictions):
        """Genera recomendaciones personalizadas basadas en predicciones"""
        recommendations = []
        
        # Basado en horas de uso
        if 'avg_daily_usage_hours' in features:
            usage_hours = features['avg_daily_usage_hours']
            if usage_hours > 6:
                recommendations.append("🚨 Tu uso de redes sociales es muy alto (>6h/día). Considera establecer límites de tiempo.")
            elif usage_hours > 4:
                recommendations.append("⚠️ Tu uso de redes sociales es considerable. Intenta reducir gradualmente el tiempo.")
            else:
                recommendations.append("✅ Tu uso de redes sociales parece estar en un rango saludable.")
        
        # Basado en sueño
        if 'sleep_hours_per_night' in features:
            sleep_hours = features['sleep_hours_per_night']
            if sleep_hours < 7:
                recommendations.append("😴 Necesitas más horas de sueño. Intenta dormir al menos 7-8 horas por noche.")
            if 'scrolling_before_bed' in features and features['scrolling_before_bed'] >= 4:
                recommendations.append("📱 Evita usar redes sociales antes de dormir para mejorar la calidad del sueño.")
        
        # Basado en predicción de salud mental
        if 'mental_health_score' in predictions:
            score = predictions['mental_health_score']
            if score < 3:
                recommendations.append("🆘 Tu puntuación de salud mental es preocupante. Considera buscar ayuda profesional.")
            elif score < 5:
                recommendations.append("⚠️ Tu salud mental podría necesitar atención. Practica técnicas de mindfulness.")
            else:
                recommendations.append("😊 Tu salud mental parece estar en buen estado. ¡Sigue así!")
        
        # Basado en impacto académico
        if 'affects_academic_performance' in predictions and predictions['affects_academic_performance'] == 1:
            recommendations.append("📚 Las redes sociales podrían estar afectando tu rendimiento académico. Considera usar apps de bloqueo durante estudio.")
        
        # Basado en ansiedad
        if 'anxiety_level' in features and features['anxiety_level'] >= 7:
            recommendations.append("😰 Tu nivel de ansiedad es alto. Practica técnicas de relajación y considera reducir el uso de redes sociales.")
        
        # Basado en comparación social
        if 'social_comparison' in features and features['social_comparison'] >= 4:
            recommendations.append("🤔 Te comparas frecuentemente con otros. Recuerda que las redes sociales no muestran la realidad completa.")
        
        return recommendations
    
    def save_survey_data(self, responses, predictions, recommendations):
        """Guarda los datos de la encuesta en el archivo Excel"""
        try:
            # Combinar respuestas, predicciones y metadatos
            new_data = responses.copy()
            
            # Agregar predicciones
            if 'mental_health_score' in predictions:
                new_data['Mental_Health_Score'] = predictions['mental_health_score']
            if 'affects_academic_performance' in predictions:
                new_data['Affects_Academic_Performance'] = predictions['affects_academic_performance']
            if 'cluster' in predictions:
                new_data['Cluster'] = predictions['cluster']
            
            # Mapear nombres de columnas para consistencia
            column_mapping = {
                'avg_daily_usage_hours': 'Avg_Daily_Usage_Hours',
                'sleep_hours_per_night': 'Sleep_Hours_Per_Night',
                'conflicts_over_social_media': 'Conflicts_Over_Social_Media',
                'anxiety_level': 'Anxiety_Level',
                'mood_changes': 'Mood_Changes',
                'social_comparison': 'Social_Comparison',
                'concentration_issues': 'Concentration_Issues',
                'procrastination': 'Procrastination',
                'physical_activity_hours': 'Physical_Activity_Hours',
                'platforms_used': 'Platforms_Used',
                'posting_frequency': 'Posting_Frequency',
                'notification_frequency': 'Notification_Frequency'
            }
            
            # Aplicar mapeo
            mapped_data = {}
            for key, value in new_data.items():
                mapped_key = column_mapping.get(key, key)
                mapped_data[mapped_key] = value
            
            # Cargar datos existentes o crear nuevo DataFrame
            if os.path.exists(self.data_file):
                existing_df = pd.read_excel(self.data_file)
                new_row_df = pd.DataFrame([mapped_data])
                updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            else:
                updated_df = pd.DataFrame([mapped_data])
            
            # Guardar archivo actualizado
            updated_df.to_excel(self.data_file, index=False)
            print(f"✅ Datos guardados en {self.data_file}")
            
            # Guardar también un log detallado
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'responses': responses,
                'predictions': predictions,
                'recommendations': recommendations
            }
            
            log_file = f"survey_log_{datetime.now().strftime('%Y%m')}.json"
            
            # Cargar log existente o crear nuevo
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
                existing_log.append(log_data)
            else:
                existing_log = [log_data]
            
            # Guardar log actualizado
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_log, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"❌ Error guardando datos: {e}")
            return False
    
    def display_results(self, features, predictions, recommendations):
        """Muestra los resultados al usuario de forma amigable"""
        print("\n" + "="*60)
        print("🎯 RESULTADOS DE TU EVALUACIÓN")
        print("="*60)
        
        # Información básica
        print(f"\n📊 RESUMEN DE TUS DATOS:")
        print(f"   💻 Uso diario de redes sociales: {features.get('avg_daily_usage_hours', 'N/A')} horas")
        print(f"   😴 Horas de sueño: {features.get('sleep_hours_per_night', 'N/A')} horas")
        print(f"   😰 Nivel de ansiedad: {features.get('anxiety_level', 'N/A')}/10")
        
        # Predicciones
        print(f"\n🔮 PREDICCIONES DEL MODELO:")
        if 'mental_health_score' in predictions:
            score = predictions['mental_health_score']
            print(f"   🧠 Puntuación de Salud Mental: {score:.2f}/10")
            if score >= 7:
                print("      ✅ Excelente salud mental")
            elif score >= 5:
                print("      😊 Buena salud mental")
            elif score >= 3:
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
        print(f"   📊 Total de características analizadas: {len(features)}")
        
        print("\n" + "="*60)
        print("¡Gracias por participar en nuestra evaluación!")
        print("="*60)
    
    def run_survey_session(self):
        """Ejecuta una sesión completa de encuesta"""
        try:
            print("🚀 Iniciando sistema de evaluación de salud mental y redes sociales...\n")
            
            # Verificar que los modelos estén cargados
            if not self.models:
                print("❌ No se pudieron cargar los modelos. Ejecuta primero el entrenamiento.")
                return
            
            # Realizar encuesta
            responses = self.conduct_survey()
            
            # Preparar características
            features = self.prepare_features(responses)
            
            # Hacer predicciones
            print("\n🔄 Analizando tus respuestas...")
            predictions = self.make_predictions(features)
            
            # Generar recomendaciones
            recommendations = self.generate_recommendations(features, predictions)
            
            # Mostrar resultados
            self.display_results(features, predictions, recommendations)
            
            # Guardar datos
            if self.save_survey_data(responses, predictions, recommendations):
                print("\n✅ Tus datos han sido guardados para mejorar el sistema.")
            
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