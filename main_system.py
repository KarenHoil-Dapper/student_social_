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
        """Define las preguntas de la encuesta adaptadas a tu dataset"""
        questions = {
            'informacion_personal': {
                'Age': "¿Cuál es tu edad?",
                'Gender': "¿Cuál es tu género? (0=Femenino, 1=Masculino, 2=Otro)"
            },
            'uso_redes_sociales': {
                'Avg_Daily_Usage_Hours': "¿Cuántas horas al día usas redes sociales en promedio?",
                'Sleep_Hours_Per_Night': "¿Cuántas horas duermes por noche normalmente?",
                'Conflicts_Over_Social_Media': "¿Has tenido conflictos por el uso de redes sociales? (0=Nunca, 1=Rara vez, 2=A veces, 3=Frecuentemente, 4=Siempre)",
                'Addicted_Score': "En una escala del 1 al 10, ¿qué tan adicto te consideras a las redes sociales?"
            },
            'nivel_academico': {
                'academic_level': "¿Cuál es tu nivel académico? (1=Secundaria, 2=Universitario, 3=Posgrado)"
            },
            'plataforma_principal': {
                'main_platform': """¿Cuál es tu plataforma principal? 
                1=Facebook, 2=Instagram, 3=TikTok, 4=YouTube, 5=WhatsApp, 
                6=Twitter, 7=Snapchat, 8=LinkedIn, 9=Otra"""
            },
            'estado_relacion': {
                'relationship_status': "¿Cuál es tu estado de relación? (1=Soltero, 2=En relación, 3=Es complicado)"
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
                        response = float(input(f"{question}: "))
                        responses[key] = response
                        break
                    except ValueError:
                        print("❌ Por favor ingresa un valor numérico válido.")
        
        # Agregar timestamp
        responses['survey_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        responses['survey_id'] = f"survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return responses
    
    def prepare_features_for_model(self, responses):
        """Convierte las respuestas en el formato que espera el modelo"""
        # Crear un registro completo con todas las columnas del dataset
        new_record = {}
        
        # Generar nuevo Student_ID
        try:
            df_existing = pd.read_excel(self.data_file)
            max_id = df_existing['Student_ID'].max() if 'Student_ID' in df_existing.columns else 0
            new_record['Student_ID'] = max_id + 1
        except:
            new_record['Student_ID'] = 1
        
        # Información básica
        new_record['Age'] = responses.get('Age', 20)
        new_record['Gender'] = int(responses.get('Gender', 1))
        new_record['Avg_Daily_Usage_Hours'] = responses.get('Avg_Daily_Usage_Hours', 3)
        new_record['Sleep_Hours_Per_Night'] = responses.get('Sleep_Hours_Per_Night', 7)
        new_record['Conflicts_Over_Social_Media'] = int(responses.get('Conflicts_Over_Social_Media', 0))
        new_record['Addicted_Score'] = responses.get('Addicted_Score', 5)
        
        # Nivel académico (one-hot encoding)
        academic_level = int(responses.get('academic_level', 2))
        new_record['Academic_Level_High School'] = 1 if academic_level == 1 else 0
        new_record['Academic_Level_Undergraduate'] = 1 if academic_level == 2 else 0
        new_record['Academic_Level_Graduate'] = 1 if academic_level == 3 else 0
        
        # Plataforma principal (one-hot encoding)
        platform_mapping = {
            1: 'Most_Used_Platform_Facebook',
            2: 'Most_Used_Platform_Instagram', 
            3: 'Most_Used_Platform_TikTok',
            4: 'Most_Used_Platform_YouTube',
            5: 'Most_Used_Platform_WhatsApp',
            6: 'Most_Used_Platform_Twitter',
            7: 'Most_Used_Platform_Snapchat',
            8: 'Most_Used_Platform_LinkedIn'
        }
        
        # Inicializar todas las plataformas a 0
        platforms = ['Facebook', 'Instagram', 'KakaoTalk', 'LINE', 'LinkedIn', 
                    'Snapchat', 'TikTok', 'Twitter', 'VKontakte', 'WeChat', 
                    'WhatsApp', 'YouTube']
        for platform in platforms:
            new_record[f'Most_Used_Platform_{platform}'] = 0
        
        # Activar la plataforma seleccionada
        main_platform = int(responses.get('main_platform', 4))
        if main_platform in platform_mapping:
            new_record[platform_mapping[main_platform]] = 1
        else:
            new_record['Most_Used_Platform_YouTube'] = 1  # Default
        
        # Estado de relación (one-hot encoding)
        relationship = int(responses.get('relationship_status', 1))
        new_record['Relationship_Status_Single'] = 1 if relationship == 1 else 0
        new_record['Relationship_Status_In Relationship'] = 1 if relationship == 2 else 0
        new_record['Relationship_Status_Complicated'] = 1 if relationship == 3 else 0
        
        # Inicializar todas las columnas de países a 0 (asumimos ubicación desconocida)
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
        
        for country in countries:
            new_record[f'Country_{country}'] = 0
        
        # Valores por defecto para columnas target (serán predichas)
        new_record['Mental_Health_Score'] = None  # Se predicirá
        new_record['Affects_Academic_Performance'] = None  # Se predicirá
        
        # Agregar metadatos
        new_record['survey_date'] = responses.get('survey_date')
        new_record['survey_id'] = responses.get('survey_id')
        
        return new_record
    
    def make_predictions(self, prepared_data):
        """Realiza predicciones usando los modelos entrenados"""
        predictions = {}
        
        try:
            # Preparar características para el modelo
            if 'extended_features' in self.model_info:
                available_features = []
                feature_values = []
                
                for feature in self.model_info['extended_features']:
                    if feature in prepared_data and prepared_data[feature] is not None:
                        available_features.append(feature)
                        feature_values.append(prepared_data[feature])
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
        if sleep_hours < 7:
            recommendations.append("😴 Necesitas más horas de sueño. Intenta dormir al menos 7-8 horas por noche.")
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
        
        # Basado en puntuación de adicción
        addiction_score = prepared_data.get('Addicted_Score', 5)
        if addiction_score >= 8:
            recommendations.append("⚠️ Tu puntuación de adicción es alta. Considera técnicas de desintoxicación digital.")
        
        # Basado en conflictos
        conflicts = prepared_data.get('Conflicts_Over_Social_Media', 0)
        if conflicts >= 3:
            recommendations.append("💬 Los conflictos por redes sociales son frecuentes. Practica comunicación digital saludable.")
        
        return recommendations
    
    def save_survey_data(self, prepared_data, predictions, recommendations):
        """Guarda los datos de la encuesta en el archivo Excel"""
        try:
            # Agregar predicciones a los datos preparados
            final_data = prepared_data.copy()
            
            if 'mental_health_score' in predictions:
                final_data['Mental_Health_Score'] = predictions['mental_health_score']
            if 'affects_academic_performance' in predictions:
                final_data['Affects_Academic_Performance'] = predictions['affects_academic_performance']
            
            # Cargar datos existentes
            if os.path.exists(self.data_file):
                existing_df = pd.read_excel(self.data_file)
                
                # Asegurar que el nuevo registro tenga todas las columnas
                for col in existing_df.columns:
                    if col not in final_data:
                        final_data[col] = 0  # Valor por defecto
                
                # Crear DataFrame con el nuevo registro
                new_row_df = pd.DataFrame([final_data])
                
                # Reordenar columnas para que coincidan
                new_row_df = new_row_df.reindex(columns=existing_df.columns, fill_value=0)
                
                # Concatenar con datos existentes
                updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            else:
                updated_df = pd.DataFrame([final_data])
            
            # Guardar archivo actualizado
            updated_df.to_excel(self.data_file, index=False)
            print(f"✅ Datos guardados en {self.data_file}")
            
            # Guardar también un log detallado
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'prepared_data': {k: v for k, v in prepared_data.items() if v is not None},
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
        print(f"   📱 Puntuación de adicción: {prepared_data.get('Addicted_Score', 'N/A')}/10")
        
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
    """Función principal"""git 
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