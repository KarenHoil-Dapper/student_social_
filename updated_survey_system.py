import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class SocialMediaHealthPredictor:
    """Sistema para análisis predictivo de salud mental desde archivo Excel"""
    
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
    
    def analyze_excel_data(self, input_file=None):
        """Analiza datos desde un archivo Excel y devuelve resultados"""
        try:
            # Usar archivo de entrada o el predeterminado
            file_to_analyze = input_file if input_file else self.data_file
            
            if not os.path.exists(file_to_analyze):
                return {"error": f"Archivo no encontrado: {file_to_analyze}"}
            
            # Leer datos del Excel
            df = pd.read_excel(file_to_analyze)
            
            if len(df) == 0:
                return {"error": "El archivo Excel está vacío"}
            
            results = []
            
            # Procesar cada fila del Excel
            for _, row in df.iterrows():
                # Convertir la fila a diccionario
                record = row.to_dict()
                
                # Calcular addicted_score si no existe
                record = self.calculate_addiction_score(record)
                
                # Hacer predicciones
                predictions = self.make_predictions(record)
                
                # Generar recomendaciones
                recommendations = self.generate_recommendations(record, predictions)
                
                # Guardar resultados
                results.append({
                    'record': record,
                    'predictions': predictions,
                    'recommendations': recommendations
                })
            
            # Guardar resultados en un nuevo Excel
            output_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            self.save_results_to_excel(results, output_file)
            
            return {
                "success": True,
                "analyzed_records": len(results),
                "output_file": output_file
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_addiction_score(self, record):
        """Calcula el puntaje de adicción basado en los datos del registro"""
        try:
            # Obtener valores del registro con valores por defecto si no existen
            usage_hours = record.get('Avg_Daily_Usage_Hours', 3)
            posting_freq = record.get('Posting_Frequency', 2)
            notification_freq = record.get('Notification_Frequency', 2)
            fomo_level = record.get('FOMO_Level', 2)
            scrolling_bed = record.get('Scrolling_Before_Bed', 2)
            concentration = record.get('Concentration_Issues', 2)
            
            # Calcular puntaje
            addiction_score = (
                (usage_hours * 1.0) +
                (posting_freq * 0.8) +
                (notification_freq * 0.7) +
                (fomo_level * 0.9) +
                (scrolling_bed * 0.8) +
                (concentration * 0.6)
            ) / 6.0
            
            # Asegurar que esté en el rango 1-10
            record['addicted_score'] = min(10.0, max(1.0, round(addiction_score, 1)))
            
        except Exception as e:
            print(f"Error calculando addicted_score: {e}")
            record['addicted_score'] = 5.0  # Valor por defecto
        
        return record
    
    def make_predictions(self, record):
        """Realiza predicciones usando los modelos entrenados"""
        predictions = {}
        
        try:
            # Verificar que tenemos información del modelo
            if not self.model_info or 'extended_features' not in self.model_info:
                print("⚠️ No hay información del modelo disponible")
                return predictions
            
            # Preparar características para el modelo
            extended_features = self.model_info['extended_features']
            available_features = []
            feature_values = []
            
            # Crear vector de características
            for feature in extended_features:
                if feature in record and record[feature] is not None and not pd.isna(record[feature]):
                    available_features.append(feature)
                    feature_values.append(float(record[feature]))
                else:
                    # Usar valor por defecto si la característica no está disponible
                    available_features.append(feature)
                    feature_values.append(0.0)
            
            # Crear DataFrame para las predicciones
            X = pd.DataFrame([feature_values], columns=available_features)
            
            # 1. CLUSTERING
            if 'clustering' in self.models and 'scaler' in self.models:
                try:
                    X_scaled = self.models['scaler'].transform(X)
                    cluster = self.models['clustering'].predict(X_scaled)[0]
                    predictions['cluster'] = int(cluster)
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
                    
                except Exception as e:
                    print(f"⚠️ Error en regresión: {e}")
                    # Usar estimación básica como fallback
                    usage_hours = record.get('Avg_Daily_Usage_Hours', 3)
                    anxiety = record.get('Anxiety_Level', 5)
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
                    
                except Exception as e:
                    print(f"⚠️ Error en clasificación: {e}")
                    # Usar estimación básica como fallback
                    usage_hours = record.get('Avg_Daily_Usage_Hours', 3)
                    concentration = record.get('Concentration_Issues', 2)
                    impact_prob = (usage_hours * 0.15 + concentration * 0.2) / 10
                    predictions['affects_academic_performance'] = 1 if impact_prob > 0.5 else 0
                    predictions['academic_impact_probability'] = float(impact_prob)
            
        except Exception as e:
            print(f"❌ Error general en predicciones: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    def generate_recommendations(self, record, predictions):
        """Genera recomendaciones personalizadas basadas en predicciones"""
        recommendations = []
        
        # Basado en horas de uso
        usage_hours = record.get('Avg_Daily_Usage_Hours', 0)
        if usage_hours > 6:
            recommendations.append("🚨 Uso de redes sociales muy alto (>6h/día). Considera establecer límites de tiempo.")
        elif usage_hours > 4:
            recommendations.append("⚠️ Uso de redes sociales considerable. Intenta reducir gradualmente el tiempo.")
        
        # Basado en sueño
        sleep_hours = record.get('Sleep_Hours_Per_Night', 7)
        scrolling_bed = record.get('Scrolling_Before_Bed', 2)
        if sleep_hours < 7:
            recommendations.append("😴 Necesitas más horas de sueño. Intenta dormir al menos 7-8 horas por noche.")
        if scrolling_bed >= 4:
            recommendations.append("📱 Evita usar redes sociales antes de dormir para mejorar la calidad del sueño.")
        
        # Basado en predicción de salud mental
        if 'mental_health_score' in predictions:
            score = predictions['mental_health_score']
            if score < 4:
                recommendations.append("🆘 Puntuación de salud mental preocupante. Considera buscar ayuda profesional.")
            elif score < 6:
                recommendations.append("⚠️ Salud mental podría necesitar atención. Practica técnicas de mindfulness.")
        
        # Basado en puntuación de adicción
        addiction_score = record.get('addicted_score', 5)
        if addiction_score >= 8:
            recommendations.append("⚠️ Puntuación de adicción alta. Considera técnicas de desintoxicación digital.")
        elif addiction_score >= 6:
            recommendations.append("🔄 Relación con redes sociales podría mejorar. Intenta pausas regulares.")
        
        return recommendations
    
    def save_results_to_excel(self, results, output_file):
        """Guarda los resultados en un nuevo archivo Excel"""
        try:
            output_data = []
            
            for result in results:
                record = result['record']
                predictions = result['predictions']
                recommendations = result['recommendations']
                
                # Combinar datos
                combined = {
                    **record,
                    **predictions,
                    'Mental_Health_Score': predictions.get('mental_health_score', record.get('mental_health_score')),
                    'Addicted_Score': record.get('addicted_score'),
                    'recommendations': "; ".join(recommendations)
               }
                
                output_data.append(combined)
            
            # Crear DataFrame y guardar
            df = pd.DataFrame(output_data)
            df.to_excel(output_file, index=False)
            print(f"✅ Resultados guardados en: {output_file}")
            
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")

def main():
    """Función principal para análisis por lotes"""
    print("🎯 SISTEMA DE ANÁLISIS DE SALUD MENTAL DESDE EXCEL")
    print("="*60)
    
    # Crear instancia del predictor
    predictor = SocialMediaHealthPredictor()
    
    # Solicitar archivo de entrada
    input_file = input("Ingrese la ruta del archivo Excel a analizar (dejar vacío para usar Social_Bueno.xlsx): ").strip()
    
    # Analizar datos
    print("\n🔄 Analizando datos...")
    result = predictor.analyze_excel_data(input_file if input_file else None)
    
    # Mostrar resultados
    if result.get("success"):
        print(f"\n✅ Análisis completado exitosamente!")
        print(f"📊 Registros analizados: {result['analyzed_records']}")
        print(f"📄 Archivo de resultados: {result['output_file']}")
    else:
        print(f"\n❌ Error en el análisis:")
        print(result.get("error", "Error desconocido"))

if __name__ == "__main__":
    main()