import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class SocialMediaMLSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
    def load_data(self, filepath):
        """Carga y preprocesa los datos"""
        self.df = pd.read_excel(filepath)
        print(f"Datos cargados: {len(self.df)} registros")
        return self.df
    
    def create_target_scores(self):
        """Crea los scores de Mental Health y Addiction basados en las variables"""
        # Calcular Mental Health Score (1-10)
        # Factores que afectan negativamente la salud mental
        mental_health_factors = (
            (10 - self.df['Avg_Daily_Usage_Hours']) +  # Menos horas = mejor
            (self.df['Sleep_Hours_Per_Night']) +       # Más sueño = mejor
            (10 - self.df['Conflicts_Over_Social_Media']) +  # Menos conflictos = mejor
            (10 - self.df['Affects_Academic_Performance'] * 5)  # No afectar = mejor
        )
        
        # Normalizar a escala 1-10
        self.df['Mental_Health_Score'] = np.clip(
            (mental_health_factors - mental_health_factors.min()) / 
            (mental_health_factors.max() - mental_health_factors.min()) * 9 + 1, 
            1, 10
        )
        
        # Calcular Addiction Score (1-10)
        # Más uso diario + menos sueño + más conflictos + afecta académicamente = más adicción
        addiction_factors = (
            self.df['Avg_Daily_Usage_Hours'] +
            (10 - self.df['Sleep_Hours_Per_Night']) +
            self.df['Conflicts_Over_Social_Media'] +
            (self.df['Affects_Academic_Performance'] * 3)
        )
        
        # Normalizar a escala 1-10
        self.df['Addiction_Score'] = np.clip(
            (addiction_factors - addiction_factors.min()) / 
            (addiction_factors.max() - addiction_factors.min()) * 9 + 1, 
            1, 10
        )
        
        print("Scores creados exitosamente")
        return self.df
    
    def prepare_features(self):
        """Prepara las características para el entrenamiento"""
        # Verificar que las columnas existen en el dataset
        print("Verificando columnas del dataset...")
        print(f"Columnas disponibles: {list(self.df.columns)[:10]}...")  # Mostrar primeras 10
        
        # Seleccionar características básicas que sabemos que existen
        basic_features = [
            'Age', 'Gender', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
            'Conflicts_Over_Social_Media', 'Affects_Academic_Performance'
        ]
        
        # Verificar que las características básicas existen
        feature_columns = []
        for feature in basic_features:
            if feature in self.df.columns:
                feature_columns.append(feature)
                print(f"✅ {feature} encontrada")
            else:
                print(f"❌ {feature} NO encontrada")
        
        # Agregar características académicas (verificar que existen)
        academic_cols = [col for col in self.df.columns if col.startswith('Academic_Level_')]
        print(f"Columnas académicas encontradas: {academic_cols}")
        feature_columns.extend(academic_cols)
        
        # Agregar plataformas más usadas (verificar que existen)
        platform_cols = [col for col in self.df.columns if col.startswith('Most_Used_Platform_')]
        print(f"Columnas de plataformas encontradas: {len(platform_cols)} plataformas")
        feature_columns.extend(platform_cols)
        
        # Agregar estado de relación (verificar que existen)
        relationship_cols = [col for col in self.df.columns if col.startswith('Relationship_Status_')]
        print(f"Columnas de relación encontradas: {relationship_cols}")
        feature_columns.extend(relationship_cols)
        
        # Verificar que tenemos al menos las características básicas
        if len(feature_columns) < 6:
            raise ValueError(f"No se encontraron suficientes características. Solo se encontraron: {feature_columns}")
        
        self.feature_names = feature_columns
        self.X = self.df[feature_columns]
        self.y_mental = self.df['Mental_Health_Score']
        self.y_addiction = self.df['Addiction_Score']
        
        print(f"✅ Características preparadas: {len(feature_columns)} variables")
        print(f"Forma de X: {self.X.shape}")
        print(f"Primeras características: {feature_columns[:10]}")
        
        return self.X, self.y_mental, self.y_addiction
    
    def train_models(self):
        """Entrena todos los modelos de ML"""
        X_train, X_test, y_mental_train, y_mental_test, y_addiction_train, y_addiction_test = \
            train_test_split(self.X, self.y_mental, self.y_addiction, test_size=0.2, random_state=42)
        
        # Escalado de características
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # 1. Regresión Lineal para Mental Health Score
        print("Entrenando Regresión Lineal...")
        self.models['linear_mental'] = LinearRegression()
        self.models['linear_mental'].fit(X_train_scaled, y_mental_train)
        
        # 2. Regresión Lineal para Addiction Score
        self.models['linear_addiction'] = LinearRegression()
        self.models['linear_addiction'].fit(X_train_scaled, y_addiction_train)
        
        # 3. Random Forest para Mental Health Score
        print("Entrenando Random Forest...")
        self.models['rf_mental'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['rf_mental'].fit(X_train, y_mental_train)
        
        # 4. Random Forest para Addiction Score
        self.models['rf_addiction'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['rf_addiction'].fit(X_train, y_addiction_train)
        
        # 5. Decision Tree Regression para Mental Health
        print("Entrenando Decision Tree Regression...")
        self.models['dt_mental'] = DecisionTreeRegressor(random_state=42)
        self.models['dt_mental'].fit(X_train, y_mental_train)
        
        # 6. Decision Tree Regression para Addiction
        self.models['dt_addiction'] = DecisionTreeRegressor(random_state=42)
        self.models['dt_addiction'].fit(X_train, y_addiction_train)
        
        # 7. Regresión Logística para clasificación de adicción (Alto/Medio/Bajo)
        print("Entrenando Regresión Logística...")
        y_addiction_class = pd.cut(y_addiction_train, bins=3, labels=['Bajo', 'Medio', 'Alto'])
        self.models['logistic'] = LogisticRegression(random_state=42)
        self.models['logistic'].fit(X_train_scaled, y_addiction_class)
        
        # 8. Decision Tree Classification para Mental Health
        print("Entrenando Decision Tree Classification...")
        y_mental_class = pd.cut(y_mental_train, bins=3, labels=['Bajo', 'Medio', 'Alto'])
        self.models['dt_class_mental'] = DecisionTreeClassifier(random_state=42)
        self.models['dt_class_mental'].fit(X_train, y_mental_class)
        
        # 9. Clustering K-Means
        print("Entrenando K-Means Clustering...")
        self.models['kmeans'] = KMeans(n_clusters=3, random_state=42)
        self.models['kmeans'].fit(X_train_scaled)
        
        print("Todos los modelos entrenados exitosamente!")
        
        # Evaluar modelos
        self.evaluate_models(X_test, X_test_scaled, y_mental_test, y_addiction_test)
        
    def evaluate_models(self, X_test, X_test_scaled, y_mental_test, y_addiction_test):
        """Evalúa el rendimiento de los modelos"""
        print("\n=== EVALUACIÓN DE MODELOS ===")
        
        # Regresión Lineal
        mental_pred = self.models['linear_mental'].predict(X_test_scaled)
        addiction_pred = self.models['linear_addiction'].predict(X_test_scaled)
        print(f"Regresión Lineal - Mental Health MSE: {mean_squared_error(y_mental_test, mental_pred):.2f}")
        print(f"Regresión Lineal - Addiction MSE: {mean_squared_error(y_addiction_test, addiction_pred):.2f}")
        
        # Random Forest
        mental_pred_rf = self.models['rf_mental'].predict(X_test)
        addiction_pred_rf = self.models['rf_addiction'].predict(X_test)
        print(f"Random Forest - Mental Health MSE: {mean_squared_error(y_mental_test, mental_pred_rf):.2f}")
        print(f"Random Forest - Addiction MSE: {mean_squared_error(y_addiction_test, addiction_pred_rf):.2f}")
        
    def predict_user_profile(self, user_data):
        """Predice scores y genera recomendaciones para un usuario"""
        try:
            # Convertir datos del usuario a formato del modelo
            user_features = self.prepare_user_features(user_data)
            
            # Verificar que el vector de características no esté vacío
            if not user_features or len(user_features) == 0:
                raise ValueError("No se pudieron preparar las características del usuario")
            
            # Verificar que el scaler esté disponible
            if 'standard' not in self.scalers:
                raise ValueError("El scaler no está disponible. Entrena el modelo primero.")
            
            # Escalar características
            user_features_array = np.array(user_features).reshape(1, -1)
            print(f"Forma del array de características: {user_features_array.shape}")
            
            user_features_scaled = self.scalers['standard'].transform(user_features_array)
            print(f"Características escaladas correctamente")
            
            results = {}
            
            # Predicciones con diferentes modelos
            results['mental_health'] = {}
            results['addiction'] = {}
            
            # Modelos que requieren escalado
            if 'linear_mental' in self.models:
                results['mental_health']['linear'] = float(self.models['linear_mental'].predict(user_features_scaled)[0])
            if 'linear_addiction' in self.models:
                results['addiction']['linear'] = float(self.models['linear_addiction'].predict(user_features_scaled)[0])
            
            # Modelos que no requieren escalado
            if 'rf_mental' in self.models:
                results['mental_health']['random_forest'] = float(self.models['rf_mental'].predict(user_features_array)[0])
            if 'rf_addiction' in self.models:
                results['addiction']['random_forest'] = float(self.models['rf_addiction'].predict(user_features_array)[0])
            if 'dt_mental' in self.models:
                results['mental_health']['decision_tree'] = float(self.models['dt_mental'].predict(user_features_array)[0])
            if 'dt_addiction' in self.models:
                results['addiction']['decision_tree'] = float(self.models['dt_addiction'].predict(user_features_array)[0])
            
            # Clasificaciones
            classifications = {}
            if 'logistic' in self.models:
                addiction_class = self.models['logistic'].predict(user_features_scaled)[0]
                classifications['addiction_level'] = addiction_class
            else:
                classifications['addiction_level'] = 'No disponible'
                
            if 'dt_class_mental' in self.models:
                mental_class = self.models['dt_class_mental'].predict(user_features_array)[0]
                classifications['mental_health_level'] = mental_class
            else:
                classifications['mental_health_level'] = 'No disponible'
                
            if 'kmeans' in self.models:
                cluster = self.models['kmeans'].predict(user_features_scaled)[0]
                classifications['cluster'] = int(cluster)
            else:
                classifications['cluster'] = 0
            
            results['classifications'] = classifications
            
            # Generar explicaciones y consejos
            results['explanations'] = self.generate_explanations(user_data, results)
            results['recommendations'] = self.generate_recommendations(user_data, results)
            
            print("✅ Predicción completada exitosamente")
            return results
            
        except Exception as e:
            print(f"❌ Error en predicción: {str(e)}")
            # Retornar resultados básicos en caso de error
            return {
                'mental_health': {'linear': 5.0, 'random_forest': 5.0, 'decision_tree': 5.0},
                'addiction': {'linear': 5.0, 'random_forest': 5.0, 'decision_tree': 5.0},
                'classifications': {'addiction_level': 'Moderado', 'mental_health_level': 'Moderado', 'cluster': 1},
                'explanations': {
                    'mental_health': f"Análisis básico: Con {user_data['horas_sueno']} horas de sueño y {user_data['uso_diario']} horas de uso diario.",
                    'addiction': f"Análisis básico: Tu uso de {user_data['uso_diario']} horas diarias sugiere un nivel moderado."
                },
                'recommendations': [
                    "🕐 Considera reducir el tiempo en redes sociales gradualmente",
                    "😴 Mantén un horario regular de sueño de 7-9 horas",
                    "📚 Establece horarios específicos para estudiar sin distracciones",
                    "🧘 Practica técnicas de relajación y mindfulness",
                    "👥 Busca actividades offline con amigos y familia"
                ]
            }
    
    def prepare_user_features(self, user_data):
        """Convierte los datos del usuario al formato de características"""
        if not hasattr(self, 'feature_names') or not self.feature_names:
            raise ValueError("Las características no han sido preparadas. Ejecuta prepare_features() primero.")
        
        features = [0] * len(self.feature_names)
        
        # Mapear datos básicos
        feature_map = {
            'Age': user_data['edad'],
            'Gender': 1 if user_data['genero'] == 'Hombre' else 0,
            'Avg_Daily_Usage_Hours': user_data['uso_diario'],
            'Sleep_Hours_Per_Night': user_data['horas_sueno'],
            'Conflicts_Over_Social_Media': user_data['conflictos'],
            'Affects_Academic_Performance': 1 if user_data['afecta_academico'] == 'Si' else 0
        }
        
        # Aplicar características básicas
        for feature, value in feature_map.items():
            if feature in self.feature_names:
                idx = self.feature_names.index(feature)
                features[idx] = value
                print(f"✅ {feature} = {value} (posición {idx})")
            else:
                print(f"⚠️ {feature} no encontrada en feature_names")
        
        # Mapear nivel académico
        academic_map = {
            'Bachillerato': 'Academic_Level_High School',
            'Licenciatura': 'Academic_Level_Undergraduate',
            'Posgrado': 'Academic_Level_Graduate'
        }
        
        if user_data['nivel_academico'] in academic_map:
            academic_col = academic_map[user_data['nivel_academico']]
            if academic_col in self.feature_names:
                idx = self.feature_names.index(academic_col)
                features[idx] = 1
                print(f"✅ {academic_col} = 1 (posición {idx})")
        
        # Mapear plataforma más usada
        platform_col = f"Most_Used_Platform_{user_data['plataforma_mas_usada']}"
        if platform_col in self.feature_names:
            idx = self.feature_names.index(platform_col)
            features[idx] = 1
            print(f"✅ {platform_col} = 1 (posición {idx})")
        else:
            print(f"⚠️ Plataforma {platform_col} no encontrada")
            # Buscar plataformas disponibles
            available_platforms = [col for col in self.feature_names if col.startswith('Most_Used_Platform_')]
            print(f"Plataformas disponibles: {available_platforms[:5]}...")
        
        # Mapear estado de relación
        relationship_map = {
            'Soltero': 'Relationship_Status_Single',
            'En relación': 'Relationship_Status_In Relationship',
            'Complicado': 'Relationship_Status_Complicated'
        }
        
        if user_data['estado_relacion'] in relationship_map:
            relationship_col = relationship_map[user_data['estado_relacion']]
            if relationship_col in self.feature_names:
                idx = self.feature_names.index(relationship_col)
                features[idx] = 1
                print(f"✅ {relationship_col} = 1 (posición {idx})")
        
        print(f"Vector de características creado: {len(features)} dimensiones")
        print(f"Valores no-cero: {sum(1 for x in features if x != 0)}")
        
        return features
    
    def generate_explanations(self, user_data, results):
        """Genera explicaciones de por qué se obtuvo cada predicción"""
        explanations = {}
        
        # Explicación Mental Health Score
        mental_avg = np.mean(list(results['mental_health'].values()))
        
        if mental_avg >= 7:
            explanations['mental_health'] = f"Tu puntuación de salud mental es ALTA ({mental_avg:.1f}/10). Factores positivos: {user_data['horas_sueno']} horas de sueño, {user_data['conflictos']} conflictos relacionados con redes sociales."
        elif mental_avg >= 4:
            explanations['mental_health'] = f"Tu puntuación de salud mental es MEDIA ({mental_avg:.1f}/10). Hay aspectos a mejorar: {user_data['uso_diario']} horas diarias en redes sociales podrían estar afectando tu bienestar."
        else:
            explanations['mental_health'] = f"Tu puntuación de salud mental es BAJA ({mental_avg:.1f}/10). Factores de riesgo: Alto uso de redes sociales ({user_data['uso_diario']} horas), poco sueño ({user_data['horas_sueno']} horas), y conflictos frecuentes."
        
        # Explicación Addiction Score
        addiction_avg = np.mean(list(results['addiction'].values()))
        
        if addiction_avg >= 7:
            explanations['addiction'] = f"Tu nivel de adicción es ALTO ({addiction_avg:.1f}/10). Indicadores: {user_data['uso_diario']} horas diarias, {user_data['conflictos']} conflictos, y afecta tu rendimiento académico."
        elif addiction_avg >= 4:
            explanations['addiction'] = f"Tu nivel de adicción es MODERADO ({addiction_avg:.1f}/10). Algunas señales de preocupación pero manejables con cambios de hábitos."
        else:
            explanations['addiction'] = f"Tu nivel de adicción es BAJO ({addiction_avg:.1f}/10). Mantienes un uso saludable de las redes sociales."
        
        return explanations
    
    def generate_recommendations(self, user_data, results):
        """Genera recomendaciones personalizadas"""
        recommendations = []
        
        mental_avg = np.mean(list(results['mental_health'].values()))
        addiction_avg = np.mean(list(results['addiction'].values()))
        
        # Recomendaciones basadas en uso diario
        if user_data['uso_diario'] >= 6:
            recommendations.append("🕐 Reduce gradualmente tu tiempo en redes sociales a máximo 4 horas diarias")
            recommendations.append("📱 Usa aplicaciones de control parental para limitar tu tiempo de pantalla")
        
        # Recomendaciones basadas en sueño
        if user_data['horas_sueno'] < 7:
            recommendations.append("😴 Aumenta tus horas de sueño a 7-9 horas diarias para mejorar tu bienestar")
            recommendations.append("🌙 Evita usar dispositivos 1 hora antes de dormir")
        
        # Recomendaciones basadas en conflictos
        if user_data['conflictos'] >= 3:
            recommendations.append("🧘 Practica técnicas de manejo del estrés y comunicación asertiva")
            recommendations.append("👥 Busca apoyo de amigos, familia o un profesional si los conflictos persisten")
        
        # Recomendaciones basadas en rendimiento académico
        if user_data['afecta_academico'] == 'Si':
            recommendations.append("📚 Establece horarios específicos para estudiar sin distracciones digitales")
            recommendations.append("🎯 Usa la técnica Pomodoro: 25 min de estudio, 5 min de descanso")
        
        # Recomendaciones generales según scores
        if mental_avg < 5:
            recommendations.append("💚 Considera hablar con un profesional de salud mental")
            recommendations.append("🌱 Incorpora actividades de bienestar: ejercicio, meditación, hobbies offline")
        
        if addiction_avg >= 7:
            recommendations.append("🚨 Tu nivel de adicción es preocupante. Busca ayuda profesional")
            recommendations.append("🔄 Implementa un 'detox digital' gradual con días sin redes sociales")
        
        # Recomendaciones específicas por plataforma
        platform_tips = {
            'Instagram': "Considera desactivar notificaciones y limitar el scroll infinito",
            'TikTok': "Usa el control de tiempo integrado de la app",
            'Facebook': "Revisa y limita las notificaciones que recibes",
            'YouTube': "Usa listas de reproducción específicas en lugar de navegar sin rumbo"
        }
        
        if user_data['plataforma_mas_usada'] in platform_tips:
            recommendations.append(f"📲 {platform_tips[user_data['plataforma_mas_usada']]}")
        
        return recommendations
    
    def save_models(self, path_prefix='models/'):
        """Guarda todos los modelos entrenados"""
        for name, model in self.models.items():
            joblib.dump(model, f'{path_prefix}{name}.pkl')
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{path_prefix}scaler_{name}.pkl')
        
        print(f"Modelos guardados en {path_prefix}")
    
    def load_models(self, path_prefix='models/'):
        """Carga modelos previamente entrenados"""
        model_names = [
            'linear_mental', 'linear_addiction', 'rf_mental', 'rf_addiction',
            'dt_mental', 'dt_addiction', 'logistic', 'dt_class_mental', 'kmeans'
        ]
        
        for name in model_names:
            try:
                self.models[name] = joblib.load(f'{path_prefix}{name}.pkl')
            except FileNotFoundError:
                print(f"No se encontró el modelo {name}")
        
        try:
            self.scalers['standard'] = joblib.load(f'{path_prefix}scaler_standard.pkl')
        except FileNotFoundError:
            print("No se encontró el scaler")

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del sistema
    ml_system = SocialMediaMLSystem()
    
    # Cargar datos
    df = ml_system.load_data('Social_Bueno.xlsx')
    
    # Crear scores objetivo
    df = ml_system.create_target_scores()
    
    # Preparar características
    X, y_mental, y_addiction = ml_system.prepare_features()
    
    # Entrenar modelos
    ml_system.train_models()
    
    # Ejemplo de predicción
    user_example = {
        'edad': 20,
        'genero': 'Hombre',
        'uso_diario': 5,
        'horas_sueno': 6,
        'conflictos': 3,
        'afecta_academico': 'Si',
        'nivel_academico': 'Licenciatura',
        'plataforma_mas_usada': 'Instagram',
        'estado_relacion': 'Soltero'
    }
    
    results = ml_system.predict_user_profile(user_example)
    
    print("\n=== RESULTADOS DE PREDICCIÓN ===")
    print(f"Mental Health Score: {results['mental_health']}")
    print(f"Addiction Score: {results['addiction']}")
    print(f"Clasificaciones: {results['classifications']}")
    print(f"Explicaciones: {results['explanations']}")
    print(f"Recomendaciones: {results['recommendations']}")