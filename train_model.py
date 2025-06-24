import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ExcelSocialMediaPredictor:
    """
    Predictor entrenado específicamente con tu archivo Excel
    """
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.target_columns = ['Mental_Health_Score', 'Addicted_Score', 'Conflicts_Over_Social_Media']
        self.country_columns = []
        self.platform_columns = []
        self.relationship_columns = []
        self.kmeans_model = None
        
    def load_excel_data(self, excel_path, sheet_name=None):
        """
        Carga el archivo Excel con dummies ya creados
        """
        excel_path='Social_bueno.xlsx'
        try:
            # Leer el archivo Excel
            if sheet_name:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(excel_path)
            
            print(f"✅ Archivo cargado exitosamente: {excel_path}")
            print(f"📊 Dimensiones: {df.shape}")
            print(f"📋 Columnas encontradas: {len(df.columns)}")
            
            # Mostrar información básica
            print("\n🔍 Primeras columnas:")
            for i, col in enumerate(df.columns[:10]):
                print(f"  {i+1}. {col}")
            if len(df.columns) > 10:
                print(f"  ... y {len(df.columns)-10} más")
            
            return df
            
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {excel_path}")
            return None
        except Exception as e:
            print(f"❌ Error al cargar el archivo: {str(e)}")
            return None
    
    def analyze_data_structure(self, df):
        """
        Analiza la estructura de datos y identifica columnas dummy
        """
        print("\n🔎 ANÁLISIS DE ESTRUCTURA DE DATOS")
        print("="*50)
        
        # Identificar columnas por tipo
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"📊 Columnas numéricas: {len(numeric_cols)}")
        print(f"📋 Columnas categóricas: {len(categorical_cols)}")
        
        # Identificar columnas dummy por patrones
        self.country_columns = [col for col in df.columns if 'Country_' in col or 'country_' in col]
        self.platform_columns = [col for col in df.columns if 'Platform_' in col or 'platform_' in col or 'Most_Used_Platform_' in col]
        self.relationship_columns = [col for col in df.columns if 'Relationship_' in col or 'relationship_' in col]
        gender_columns = [col for col in df.columns if 'Gender_' in col or 'gender_' in col]
        academic_columns = [col for col in df.columns if 'Academic_' in col or 'academic_' in col]
        
        print(f"\n🌍 Dummies de países: {len(self.country_columns)}")
        if self.country_columns:
            print(f"   Ejemplos: {self.country_columns[:3]}{'...' if len(self.country_columns) > 3 else ''}")
        
        print(f"📱 Dummies de plataformas: {len(self.platform_columns)}")
        if self.platform_columns:
            print(f"   Ejemplos: {self.platform_columns[:3]}{'...' if len(self.platform_columns) > 3 else ''}")
        
        print(f"💕 Dummies de relaciones: {len(self.relationship_columns)}")
        if self.relationship_columns:
            print(f"   Ejemplos: {self.relationship_columns}")
        
        print(f"👤 Dummies de género: {len(gender_columns)}")
        if gender_columns:
            print(f"   Ejemplos: {gender_columns}")
        
        print(f"🎓 Dummies académicos: {len(academic_columns)}")
        if academic_columns:
            print(f"   Ejemplos: {academic_columns}")
        
        # Verificar targets
        available_targets = [col for col in self.target_columns if col in df.columns]
        print(f"\n🎯 Variables objetivo disponibles: {available_targets}")
        
        return available_targets
    
    def prepare_features_and_targets(self, df):
        """
        Prepara las características y variables objetivo
        """
        # Excluir columnas que no son features
        exclude_columns = ['Student_ID', 'ID', 'Index'] + self.target_columns
        
        # Seleccionar todas las columnas excepto las excluidas
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Verificar que las características son numéricas (ya dummificadas)
        non_numeric = []
        for col in feature_columns:
            if df[col].dtype == 'object':
                print(f"⚠️  Advertencia: {col} no es numérica. Valores únicos: {df[col].unique()[:5]}")
                non_numeric.append(col)
        
        # Remover columnas no numéricas si las hay
        feature_columns = [col for col in feature_columns if col not in non_numeric]
        
        self.feature_columns = feature_columns
        
        X = df[feature_columns].copy()
        y = df[self.target_columns].copy()
        
        print(f"\n✅ DATOS PREPARADOS:")
        print(f"   Features: {len(feature_columns)} columnas")
        print(f"   Targets: {len(self.target_columns)} variables")
        print(f"   Muestras: {len(X)} filas")
        
        # Verificar valores faltantes
        missing_features = X.isnull().sum().sum()
        missing_targets = y.isnull().sum().sum()
        
        if missing_features > 0:
            print(f"⚠️  Valores faltantes en features: {missing_features}")
            X = X.fillna(0)  # Rellenar con 0 para dummies
        
        if missing_targets > 0:
            print(f"⚠️  Valores faltantes en targets: {missing_targets}")
            y = y.fillna(y.median())  # Rellenar con mediana para targets
        
        return X, y
    
    def train_models(self, df, test_size=0.2):
        """
        Entrena los modelos con validación cruzada
        """
        print("\n🚀 INICIANDO ENTRENAMIENTO")
        print("="*50)
        
        # Preparar datos
        X, y = self.prepare_features_and_targets(df)
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Escalar características
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Entrenar modelo para cada target
        for target in self.target_columns:
            if target in y.columns:
                print(f"\n🎯 Entrenando modelo para: {target}")
                
                # Configurar modelo con mejores hiperparámetros
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Entrenar modelo
                model.fit(X_train_scaled, y_train[target])
                
                # Predicciones
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Métricas
                train_mse = mean_squared_error(y_train[target], y_pred_train)
                test_mse = mean_squared_error(y_test[target], y_pred_test)
                train_r2 = r2_score(y_train[target], y_pred_train)
                test_r2 = r2_score(y_test[target], y_pred_test)
                
                # Validación cruzada
                cv_scores = cross_val_score(model, X_train_scaled, y_train[target], 
                                          cv=5, scoring='neg_mean_squared_error')
                cv_mse = -cv_scores.mean()
                
                # Guardar modelo y resultados
                self.models[target] = model
                results[target] = {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'cv_mse': cv_mse,
                    'feature_importance': dict(zip(self.feature_columns, model.feature_importances_))
                }
                
                print(f"   📊 Train MSE: {train_mse:.4f}")
                print(f"   📊 Test MSE:  {test_mse:.4f}")
                print(f"   📊 Train R²:  {train_r2:.4f}")
                print(f"   📊 Test R²:   {test_r2:.4f}")
                print(f"   📊 CV MSE:    {cv_mse:.4f}")
        
        # Clustering de usuarios
        print(f"\n🔍 Realizando clustering de usuarios...")
        self.kmeans_model = KMeans(n_clusters=4, random_state=42)
        clusters = self.kmeans_model.fit_predict(X_train_scaled)
        results['clusters'] = {
            'n_clusters': 4,
            'cluster_distribution': np.bincount(clusters)
        }
        
        print(f"   📊 Distribución de clusters: {np.bincount(clusters)}")
        
        return results
    
    def predict_new_user(self, user_data):
        """
        Predice para un nuevo usuario usando el formato original
        """
        if not self.models:
            raise ValueError("❌ No hay modelos entrenados. Ejecuta train_models() primero.")
        
        # Convertir datos del usuario al formato con dummies
        user_features = self.convert_user_to_dummies(user_data)
        
        # Crear DataFrame con las mismas columnas que el entrenamiento
        user_df = pd.DataFrame([user_features])
        
        # Asegurar que tiene todas las columnas necesarias
        for col in self.feature_columns:
            if col not in user_df.columns:
                user_df[col] = 0
        
        # Reordenar columnas
        user_df = user_df[self.feature_columns]
        
        # Escalar
        user_scaled = self.scaler.transform(user_df)
        
        # Predicciones
        predictions = {}
        for target, model in self.models.items():
            pred = model.predict(user_scaled)[0]
            predictions[target] = round(float(pred), 2)
        
        # Cluster
        if self.kmeans_model:
            cluster = self.kmeans_model.predict(user_scaled)[0]
            predictions['user_cluster'] = int(cluster)
        
        return predictions
    
    def convert_user_to_dummies(self, user_data):
        """
        Convierte datos del usuario al formato dummy
        """
        dummy_data = {}
        
        # Copiar valores numéricos directamente
        numeric_fields = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']
        for field in numeric_fields:
            if field in user_data:
                dummy_data[field] = user_data[field]
        
        # Convertir género
        if 'Gender' in user_data:
            for col in self.feature_columns:
                if col.startswith('Gender_'):
                    gender_value = col.replace('Gender_', '')
                    dummy_data[col] = 1 if user_data['Gender'] == gender_value else 0
        
        # Convertir país
        if 'Country' in user_data:
            for col in self.country_columns:
                country_value = col.replace('Country_', '')
                dummy_data[col] = 1 if user_data['Country'] == country_value else 0
        
        # Convertir plataforma
        if 'Most_Used_Platform' in user_data:
            for col in self.platform_columns:
                platform_value = col.replace('Most_Used_Platform_', '').replace('Platform_', '')
                dummy_data[col] = 1 if user_data['Most_Used_Platform'] == platform_value else 0
        
        # Convertir relación
        if 'Relationship_Status' in user_data:
            for col in self.relationship_columns:
                relation_value = col.replace('Relationship_Status_', '').replace('Relationship_', '')
                dummy_data[col] = 1 if user_data['Relationship_Status'] == relation_value else 0
        
        # Convertir nivel académico
        if 'Academic_Level' in user_data:
            for col in self.feature_columns:
                if col.startswith('Academic_'):
                    academic_value = col.replace('Academic_Level_', '').replace('Academic_', '')
                    dummy_data[col] = 1 if user_data['Academic_Level'] == academic_value else 0
        
        # Convertir afecta rendimiento académico
        if 'Affects_Academic_Performance' in user_data:
            for col in self.feature_columns:
                if 'Affects_Academic_Performance' in col:
                    if 'Yes' in col:
                        dummy_data[col] = 1 if user_data['Affects_Academic_Performance'] == 'Yes' else 0
                    elif 'No' in col:
                        dummy_data[col] = 1 if user_data['Affects_Academic_Performance'] == 'No' else 0
        
        return dummy_data
    
    def get_feature_importance_report(self):
        """
        Genera reporte de importancia de características
        """
        if not self.models:
            print("❌ No hay modelos entrenados.")
            return
        
        print("\n📊 IMPORTANCIA DE CARACTERÍSTICAS")
        print("="*60)
        
        for target, model in self.models.items():
            print(f"\n🎯 {target}:")
            
            # Obtener importancias
            importances = model.feature_importances_
            feature_importance = list(zip(self.feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Top 10 características más importantes
            print("   Top 10 características más importantes:")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                print(f"   {i+1:2d}. {feature:<30} {importance:.4f}")
    
    def save_trained_model(self, filepath_prefix='trained_social_media_model'):
        """
        Guarda el modelo entrenado
        """
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'country_columns': self.country_columns,
            'platform_columns': self.platform_columns,
            'relationship_columns': self.relationship_columns,
            'kmeans_model': self.kmeans_model
        }
        
        joblib.dump(model_data, f'{filepath_prefix}.pkl')
        print(f"✅ Modelo guardado como: {filepath_prefix}.pkl")
    
    def load_trained_model(self, filepath):
        """
        Carga un modelo previamente entrenado
        """
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_columns = model_data['target_columns']
            self.country_columns = model_data['country_columns']
            self.platform_columns = model_data['platform_columns']
            self.relationship_columns = model_data['relationship_columns']
            self.kmeans_model = model_data['kmeans_model']
            
            print(f"✅ Modelo cargado desde: {filepath}")
            
        except Exception as e:
            print(f"❌ Error al cargar modelo: {str(e)}")

# Función principal para usar con tu archivo
def train_with_your_excel(excel_path, sheet_name=None):
    """
    Función principal para entrenar con tu archivo Excel
    """
    print("🎯 ENTRENADOR DE MODELO CON EXCEL PERSONALIZADO")
    print("="*60)
    
    # Crear predictor
    predictor = ExcelSocialMediaPredictor()
    
    # Cargar datos
    df = predictor.load_excel_data(excel_path, sheet_name)
    if df is None:
        return None
    
    # Analizar estructura
    available_targets = predictor.analyze_data_structure(df)
    
    if not available_targets:
        print("❌ No se encontraron variables objetivo válidas")
        return None
    
    # Entrenar modelos
    results = predictor.train_models(df)
    
    # Mostrar reporte de importancia
    predictor.get_feature_importance_report()
    
    # Guardar modelo
    predictor.save_trained_model('mi_modelo_redes_sociales')
    
    print("\n🎉 ENTRENAMIENTO COMPLETADO")
    print("="*40)
    print("✅ Modelos entrenados y guardados")
    print("✅ Listo para hacer predicciones")
    
    return predictor

# Ejemplo de uso
if __name__ == "__main__":
    # REEMPLAZA ESTA RUTA CON LA RUTA DE TU ARCHIVO EXCEL
    excel_file_path = "tu_archivo_redes_sociales.xlsx"
    
    # Entrenar con tu archivo
    predictor = train_with_your_excel(excel_file_path)
    
    if predictor:
        # Ejemplo de predicción
        nuevo_usuario = {
            'Age': 20,
            'Gender': 'Female',
            'Academic_Level': 'Undergraduate',
            'Country': 'Mexico',
            'Avg_Daily_Usage_Hours': 5.3,
            'Most_Used_Platform': 'Instagram',
            'Affects_Academic_Performance': 'Yes',
            'Sleep_Hours_Per_Night': 5.5,
            'Relationship_Status': 'Single'
        }
        
        try:
            predicciones = predictor.predict_new_user(nuevo_usuario)
            print(f"\n🔮 PREDICCIONES PARA NUEVO USUARIO:")
            for key, value in predicciones.items():
                print(f"   {key}: {value}")
        except Exception as e:
            print(f"❌ Error en predicción: {str(e)}")