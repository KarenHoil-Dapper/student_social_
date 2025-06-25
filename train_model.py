import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CARGA Y PREPROCESAMIENTO AVANZADO DE DATOS
# ==========================================

def load_and_preprocess_data(file_path):
    """Carga y preprocesa los datos con ingeniería de características avanzada"""
    try:
        df = pd.read_excel(file_path)
        print(f"📊 Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"📋 Columnas disponibles: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error cargando el archivo: {e}")
        raise
    
    # Columnas principales requeridas
    required_cols = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media']
    target_cols = ['Mental_Health_Score', 'Affects_Academic_Performance']
    
    # Verificar que las columnas requeridas existan
    missing_cols = [col for col in required_cols + target_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Columnas faltantes: {missing_cols}")
        print("Utilizando columnas disponibles...")
    
    # Identificar características numéricas disponibles
    potential_features = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64', 'float32', 'int32'] and col not in target_cols:
            potential_features.append(col)
    
    print(f"🔍 Características numéricas encontradas: {potential_features}")
    
    # Usar características disponibles (combinando requeridas y encontradas)
    available_required = [col for col in required_cols if col in df.columns]
    extended_features = list(set(available_required + potential_features))
    
    print(f"✅ Características a utilizar: {extended_features}")
    
    # Limpiar datos - solo incluir columnas que existen
    columns_to_check = []
    for col in extended_features + target_cols:
        if col in df.columns:
            columns_to_check.append(col)
    
    df_clean = df.dropna(subset=columns_to_check)
    print(f"🧹 Datos después de limpiar: {df_clean.shape[0]} filas")
    
    # INGENIERÍA DE CARACTERÍSTICAS
    # 1. Crear características derivadas (solo si las columnas base existen)
    if 'Avg_Daily_Usage_Hours' in df_clean.columns and 'Sleep_Hours_Per_Night' in df_clean.columns:
        print("🔧 Creando características derivadas...")
        df_clean['usage_sleep_ratio'] = df_clean['Avg_Daily_Usage_Hours'] / (df_clean['Sleep_Hours_Per_Night'] + 0.1)
        df_clean['total_daily_activity'] = df_clean['Avg_Daily_Usage_Hours'] + df_clean['Sleep_Hours_Per_Night']
        df_clean['sleep_deficit'] = np.where(df_clean['Sleep_Hours_Per_Night'] < 8, 
                                           8 - df_clean['Sleep_Hours_Per_Night'], 0)
        
        # Agregar nuevas características a la lista
        extended_features.extend(['usage_sleep_ratio', 'total_daily_activity', 'sleep_deficit'])
    
    # 2. Crear categorías de uso (solo si la columna existe)
    if 'Avg_Daily_Usage_Hours' in df_clean.columns:
        print("📊 Creando categorías de uso...")
        df_clean['usage_category'] = pd.cut(df_clean['Avg_Daily_Usage_Hours'], 
                                          bins=[0, 2, 4, 6, float('inf')], 
                                          labels=[0, 1, 2, 3])  # Usar números en lugar de strings
        
        # Convertir a numérico
        df_clean['usage_category_encoded'] = df_clean['usage_category'].astype(float)
        extended_features.append('usage_category_encoded')
    
    # 3. Obtener todas las columnas numéricas finales
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in target_cols]
    
    # Actualizar lista de características finales
    final_features = [col for col in numeric_cols if col in df_clean.columns]
    
    print(f"🎯 Características finales: {len(final_features)} características")
    print(f"   {final_features}")
    
    return df_clean, final_features

# ==========================================
# SELECCIÓN INTELIGENTE DE CARACTERÍSTICAS
# ==========================================

def select_best_features(X, y, task_type='regression', k=10):
    """Selecciona las mejores características basándose en scores estadísticos"""
    if task_type == 'regression':
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    else:
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"🎯 Características seleccionadas para {task_type}: {selected_features}")
    return X_selected, selected_features, selector

# ==========================================
# CLUSTERING MEJORADO
# ==========================================

def improved_clustering(X, feature_names):
    """Clustering mejorado con optimización de hiperparámetros"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encontrar número óptimo de clusters usando método del codo
    inertias = []
    K_range = range(2, min(8, len(X)//2))
    
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_scaled)
        inertias.append(kmeans_temp.inertia_)
    
    # Seleccionar k óptimo (simplificado)
    optimal_k = 3  # Puedes implementar método del codo más sofisticado
    
    # Modelo final
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Analizar clusters
    cluster_analysis = {}
    for i in range(optimal_k):
        cluster_mask = clusters == i
        cluster_data = X[cluster_mask]
        cluster_analysis[f'Cluster_{i}'] = {
            'size': np.sum(cluster_mask),
            'characteristics': {
                feature: {
                    'mean': cluster_data[feature].mean(),
                    'std': cluster_data[feature].std()
                } for feature in feature_names
            }
        }
    
    print("🔍 Análisis de clusters:")
    for cluster_name, analysis in cluster_analysis.items():
        print(f"{cluster_name}: {analysis['size']} muestras")
    
    return kmeans, scaler, clusters, cluster_analysis

# ==========================================
# MODELOS DE REGRESIÓN AVANZADOS
# ==========================================

def build_advanced_regression_models(X, y, feature_names):
    """Construye múltiples modelos de regresión con optimización"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelos a probar
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Polynomial': None  # Se configurará después
    }
    
    # Crear características polinomiales
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    models['Polynomial'] = LinearRegression()
    
    best_model = None
    best_score = float('-inf')
    results = {}
    
    for name, model in models.items():
        if name == 'Polynomial':
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)
            score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
        
        results[name] = {'r2_score': score, 'mse': mse}
        
        if score > best_score:
            best_score = score
            best_model = (name, model, poly_features if name == 'Polynomial' else None)
    
    print("📈 Resultados de regresión:")
    for name, metrics in results.items():
        print(f"{name}: R² = {metrics['r2_score']:.4f}, MSE = {metrics['mse']:.4f}")
    
    return best_model, results

# ==========================================
# MODELOS DE CLASIFICACIÓN AVANZADOS
# ==========================================

def build_advanced_classification_models(X, y):
    """Construye modelos de clasificación con optimización de hiperparámetros"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelos base mejorados
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Optimización de hiperparámetros para RandomForest
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    models['RandomForest_Optimized'] = rf_grid.best_estimator_
    
    # Ensemble final
    ensemble = VotingClassifier(
        estimators=[
            ('lr', models['LogisticRegression']),
            ('rf', models['RandomForest_Optimized']),
            ('gb', models['GradientBoosting'])
        ],
        voting='soft'
    )
    
    # Evaluar modelos
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = (name, model)
    
    # Evaluar ensemble
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    results['Ensemble'] = ensemble_accuracy
    
    if ensemble_accuracy > best_score:
        best_model = ('Ensemble', ensemble)
    
    print("🎯 Resultados de clasificación:")
    for name, accuracy in results.items():
        print(f"{name}: Precisión = {accuracy:.4f}")
    
    return best_model, results

# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def main():
    print("🚀 Iniciando entrenamiento de modelos mejorados...")
    
    try:
        # Cargar y preprocesar datos
        df, extended_features = load_and_preprocess_data("Social_Bueno.xlsx")
        
        if len(extended_features) == 0:
            print("❌ No se encontraron características válidas para entrenar")
            return
        
        # Verificar datos objetivo
        target_cols = ['Mental_Health_Score', 'Affects_Academic_Performance']
        available_targets = [col for col in target_cols if col in df.columns]
        
        if len(available_targets) == 0:
            print("❌ No se encontraron columnas objetivo válidas")
            return
        
        print(f"🎯 Columnas objetivo disponibles: {available_targets}")
        
        # Preparar datos
        X = df[extended_features]
        print(f"📊 Matriz de características: {X.shape}")
        
        # Crear directorio de modelos
        os.makedirs("model_advanced", exist_ok=True)
        
        # 1. CLUSTERING MEJORADO
        print("\n1️⃣ Entrenando modelo de clustering...")
        kmeans, scaler, clusters, cluster_analysis = improved_clustering(X, extended_features)
        df['cluster_advanced'] = clusters
        
        joblib.dump(kmeans, "model_advanced/kmeans_advanced.pkl")
        joblib.dump(scaler, "model_advanced/scaler_advanced.pkl")
        joblib.dump(cluster_analysis, "model_advanced/cluster_analysis.pkl")
        print("✅ Clustering completado")
        
        # 2. REGRESIÓN AVANZADA (si existe Mental_Health_Score)
        reg_trained = False
        if 'Mental_Health_Score' in df.columns:
            print("\n2️⃣ Entrenando modelos de regresión...")
            y_reg = df['Mental_Health_Score']
            
            # Verificar que no hay valores nulos en y_reg
            if y_reg.isnull().any():
                print("⚠️ Limpiando valores nulos en Mental_Health_Score...")
                valid_indices = ~y_reg.isnull()
                X_reg = X[valid_indices]
                y_reg = y_reg[valid_indices]
            else:
                X_reg = X
            
            if len(y_reg) > 0:
                X_reg_selected, reg_features, reg_selector = select_best_features(X_reg, y_reg, 'regression')
                best_reg_model, reg_results = build_advanced_regression_models(
                    pd.DataFrame(X_reg_selected, columns=reg_features), y_reg, reg_features
                )
                
                joblib.dump(best_reg_model, "model_advanced/regression_best.pkl")
                joblib.dump(reg_selector, "model_advanced/regression_selector.pkl")
                joblib.dump(reg_features, "model_advanced/regression_features.pkl")
                reg_trained = True
                print("✅ Regresión completada")
        else:
            print("⚠️ Saltando regresión - No se encontró 'Mental_Health_Score'")
            reg_features = []
            reg_results = {}
            best_reg_model = ("None", None)
        
        # 3. CLASIFICACIÓN AVANZADA (si existe Affects_Academic_Performance)
        clf_trained = False
        if 'Affects_Academic_Performance' in df.columns:
            print("\n3️⃣ Entrenando modelos de clasificación...")
            y_clf = df['Affects_Academic_Performance']
            
            # Verificar que no hay valores nulos en y_clf
            if y_clf.isnull().any():
                print("⚠️ Limpiando valores nulos en Affects_Academic_Performance...")
                valid_indices = ~y_clf.isnull()
                X_clf = X[valid_indices]
                y_clf = y_clf[valid_indices]
            else:
                X_clf = X
            
            if len(y_clf) > 0:
                X_clf_selected, clf_features, clf_selector = select_best_features(X_clf, y_clf, 'classification')
                best_clf_model, clf_results = build_advanced_classification_models(
                    pd.DataFrame(X_clf_selected, columns=clf_features), y_clf
                )
                
                joblib.dump(best_clf_model, "model_advanced/classification_best.pkl")
                joblib.dump(clf_selector, "model_advanced/classification_selector.pkl")
                joblib.dump(clf_features, "model_advanced/classification_features.pkl")
                clf_trained = True
                print("✅ Clasificación completada")
        else:
            print("⚠️ Saltando clasificación - No se encontró 'Affects_Academic_Performance'")
            clf_features = []
            clf_results = {}
            best_clf_model = ("None", None)
        
        # Guardar información del modelo
        model_info = {
            'extended_features': extended_features,
            'regression_features': reg_features if reg_trained else [],
            'classification_features': clf_features if clf_trained else [],
            'best_regression_model': best_reg_model[0] if reg_trained else "None",
            'best_classification_model': best_clf_model[0] if clf_trained else "None",
            'regression_results': reg_results if reg_trained else {},
            'classification_results': clf_results if clf_trained else {},
            'cluster_analysis': cluster_analysis,
            'models_trained': {
                'clustering': True,
                'regression': reg_trained,
                'classification': clf_trained
            }
        }
        
        joblib.dump(model_info, "model_advanced/model_info.pkl")
        
        print("\n" + "="*50)
        print("✅ ENTRENAMIENTO COMPLETADO!")
        print("="*50)
        print(f"📁 Archivos guardados en: ./model_advanced/")
        print(f"🔄 Clustering: ✅ Entrenado")
        print(f"📈 Regresión: {'✅ Entrenado' if reg_trained else '❌ No entrenado'}")
        if reg_trained:
            print(f"   🏆 Mejor modelo: {best_reg_model[0]}")
        print(f"🎯 Clasificación: {'✅ Entrenado' if clf_trained else '❌ No entrenado'}")
        if clf_trained:
            print(f"   🏆 Mejor modelo: {best_clf_model[0]}")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()