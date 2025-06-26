
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
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CARGA Y PREPROCESAMIENTO AVANZADO DE DATOS
# ==========================================

def load_and_preprocess_data(file_path):
    """Carga y preprocesa los datos con ingeniería de características avanzada"""
    try:
        df = pd.read_excel(file_path)
        print(f"[INFO] Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"[INFO] Primeras 5 columnas: {list(df.columns[:5])}")
        
    except Exception as e:
        print(f"[ERROR] Error cargando el archivo: {e}")
        raise
    
    # Columnas principales requeridas (actualizadas según tu dataset)
    required_cols = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media']
    target_cols = ['Mental_Health_Score', 'Affects_Academic_Performance']
    
    # Verificar que las columnas requeridas existan
    missing_cols = [col for col in required_cols + target_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARNING] Columnas faltantes: {missing_cols}")
        print("Utilizando columnas disponibles...")
    
    # Identificar características numéricas disponibles (excluyendo IDs y metadatos)
    exclude_cols = ['Student_ID', 'survey_date', 'survey_id'] + target_cols
    potential_features = []
    
    for col in df.columns:
        if col not in exclude_cols:
            # Incluir tanto numéricas como categóricas (one-hot encoded)
            if df[col].dtype in ['int64', 'float64', 'float32', 'int32', 'bool']:
                potential_features.append(col)
    
    print(f"[INFO] Características encontradas: {len(potential_features)}")
    
    # Usar todas las características disponibles
    extended_features = potential_features
    
    print(f"[SUCCESS] Características a utilizar: {len(extended_features)} características")
    
    # Limpiar datos - remover filas con valores nulos en columnas importantes
    important_cols = []
    for col in extended_features + target_cols:
        if col in df.columns:
            important_cols.append(col)
    
    # Contar valores nulos antes
    null_counts_before = df[important_cols].isnull().sum().sum()
    print(f"[INFO] Valores nulos encontrados: {null_counts_before}")
    
    # Limpiar datos
    df_clean = df.dropna(subset=target_cols)  # Solo eliminar si faltan targets
    
    # Rellenar valores nulos en características con la mediana/moda
    for col in extended_features:
        if col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['int64', 'float64', 'float32', 'int32']:
                    # Para numéricas, usar mediana
                    fill_value = df_clean[col].median()
                else:
                    # Para categóricas, usar moda
                    fill_value = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 0
                
                df_clean[col] = df_clean[col].fillna(fill_value)
                print(f"[FIX] Rellenados valores nulos en {col} con {fill_value}")
    
    print(f"[SUCCESS] Datos después de limpiar: {df_clean.shape[0]} filas")
    
    # INGENIERÍA DE CARACTERÍSTICAS MEJORADA
    print("[PROCESS] Creando características derivadas...")
    
    # 1. Características derivadas básicas
    if 'Avg_Daily_Usage_Hours' in df_clean.columns and 'Sleep_Hours_Per_Night' in df_clean.columns:
        df_clean['usage_sleep_ratio'] = df_clean['Avg_Daily_Usage_Hours'] / (df_clean['Sleep_Hours_Per_Night'] + 0.1)
        df_clean['total_daily_activity'] = df_clean['Avg_Daily_Usage_Hours'] + df_clean['Sleep_Hours_Per_Night']
        df_clean['sleep_deficit'] = np.where(df_clean['Sleep_Hours_Per_Night'] < 8, 
                                           8 - df_clean['Sleep_Hours_Per_Night'], 0)
        extended_features.extend(['usage_sleep_ratio', 'total_daily_activity', 'sleep_deficit'])
    
    # 2. Categorías de uso
    if 'Avg_Daily_Usage_Hours' in df_clean.columns:
        df_clean['usage_category_low'] = (df_clean['Avg_Daily_Usage_Hours'] <= 2).astype(int)
        df_clean['usage_category_medium'] = ((df_clean['Avg_Daily_Usage_Hours'] > 2) & 
                                           (df_clean['Avg_Daily_Usage_Hours'] <= 6)).astype(int)
        df_clean['usage_category_high'] = (df_clean['Avg_Daily_Usage_Hours'] > 6).astype(int)
        extended_features.extend(['usage_category_low', 'usage_category_medium', 'usage_category_high'])
    
    # 3. Características de bienestar mental (si existen)
    mental_indicators = ['Anxiety_Level', 'Mood_Changes', 'Social_Comparison', 'FOMO_Level']
    available_mental = [col for col in mental_indicators if col in df_clean.columns]
    
    if len(available_mental) >= 2:
        # Crear índice de bienestar mental combinado
        mental_cols = df_clean[available_mental]
        df_clean['mental_wellness_index'] = mental_cols.mean(axis=1)
        df_clean['mental_risk_high'] = (df_clean['mental_wellness_index'] > mental_cols.mean().mean()).astype(int)
        extended_features.extend(['mental_wellness_index', 'mental_risk_high'])
    
    # 4. Características de productividad (si existen)
    productivity_indicators = ['Concentration_Issues', 'Procrastination', 'Productivity_Impact']
    available_productivity = [col for col in productivity_indicators if col in df_clean.columns]
    
    if len(available_productivity) >= 2:
        # Crear índice de productividad
        prod_cols = df_clean[available_productivity]
        df_clean['productivity_impact_index'] = prod_cols.mean(axis=1)
        extended_features.append('productivity_impact_index')
    
    # 5. Interacciones importantes
    if 'Addicted_Score' in df_clean.columns and 'Avg_Daily_Usage_Hours' in df_clean.columns:
        df_clean['addiction_usage_interaction'] = df_clean['Addicted_Score'] * df_clean['Avg_Daily_Usage_Hours']
        extended_features.append('addiction_usage_interaction')
    
    # Filtrar características finales que realmente existen
    final_features = [col for col in extended_features if col in df_clean.columns]
    
    # Verificar varianza (eliminar características constantes)
    variance_check = df_clean[final_features].var()
    valid_features = variance_check[variance_check > 0.0001].index.tolist()
    
    print(f"[SUCCESS] Características finales válidas: {len(valid_features)}")
    print(f"   Ejemplos: {valid_features[:10]}")
    
    return df_clean, valid_features

# ==========================================
# SELECCIÓN INTELIGENTE DE CARACTERÍSTICAS
# ==========================================

def select_best_features(X, y, task_type='regression', k=15):
    """Selecciona las mejores características basándose en scores estadísticos"""
    # Ajustar k al número disponible de características
    k = min(k, X.shape[1], max(5, X.shape[1] // 3))
    
    if task_type == 'regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    else:
        selector = SelectKBest(score_func=f_classif, k=k)
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    scores = selector.scores_[selector.get_support()]
    
    print(f"[SUCCESS] Características seleccionadas para {task_type} (top {k}):")
    for feature, score in zip(selected_features, scores):
        print(f"   {feature}: {score:.3f}")
    
    return X_selected, selected_features, selector

# ==========================================
# CLUSTERING MEJORADO
# ==========================================

def improved_clustering(X, feature_names):
    """Clustering mejorado con optimización de hiperparámetros"""
    # Asegurar que tenemos suficientes datos
    if len(X) < 6:
        print("[WARNING] Insuficientes datos para clustering, usando k=2")
        optimal_k = 2
    else:
        # Determinar k óptimo usando método del codo
        inertias = []
        K_range = range(2, min(6, len(X)//3 + 1))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(X_scaled)
            inertias.append(kmeans_temp.inertia_)
        
        # Método del codo simplificado
        if len(inertias) >= 2:
            deltas = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
            optimal_k = deltas.index(max(deltas)) + 2
        else:
            optimal_k = 3
    
    print(f"[INFO] Número óptimo de clusters: {optimal_k}")
    
    # Modelo final
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Analizar clusters
    cluster_analysis = {}
    for i in range(optimal_k):
        cluster_mask = clusters == i
        cluster_data = X[cluster_mask]
        
        if len(cluster_data) > 0:
            cluster_analysis[f'Cluster_{i}'] = {
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(X) * 100),
                'characteristics': {}
            }
            
            # Solo analizar características más importantes
            important_features = feature_names[:min(10, len(feature_names))]
            for feature in important_features:
                if feature in cluster_data.columns:
                    cluster_analysis[f'Cluster_{i}']['characteristics'][feature] = {
                        'mean': float(cluster_data[feature].mean()),
                        'std': float(cluster_data[feature].std())
                    }
    
    print("[INFO] Análisis de clusters:")
    for cluster_name, analysis in cluster_analysis.items():
        print(f"{cluster_name}: {analysis['size']} muestras ({analysis['percentage']:.1f}%)")
    
    return kmeans, scaler, clusters, cluster_analysis

# ==========================================
# MODELOS DE REGRESIÓN AVANZADOS
# ==========================================

def build_advanced_regression_models(X, y, feature_names):
    """Construye múltiples modelos de regresión con optimización"""
    # Verificar tamaño del dataset
    if len(X) < 10:
        print("[WARNING] Dataset muy pequeño para división train/test, usando todo el dataset")
        X_train, X_test = X, X
        y_train, y_test = y, y
        test_size = 0
    else:
        test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"[INFO] Entrenamiento: {len(X_train)} muestras, Prueba: {len(X_test)} muestras")
    
    # Modelos a probar (ajustados para datasets pequeños)
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0, max_iter=2000)
    }
    
    # Solo usar polinomiales si tenemos suficientes datos
    poly_features = None
    if len(X_train) > 50 and X_train.shape[1] <= 10:
        try:
            poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test)
            
            # Verificar que no creamos demasiadas características
            if X_train_poly.shape[1] <= X_train.shape[0] // 2:
                models['Polynomial'] = LinearRegression()
            else:
                print("[WARNING] Demasiadas características polinomiales, saltando modelo polinomial")
                poly_features = None
        except Exception as e:
            print(f"[WARNING] Error creando características polinomiales: {e}")
            poly_features = None
    
    best_model = None
    best_score = float('-inf')
    results = {}
    
    for name, model in models.items():
        try:
            if name == 'Polynomial' and poly_features is not None:
                model.fit(X_train_poly, y_train)
                if test_size > 0:
                    y_pred = model.predict(X_test_poly)
                    score = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                else:
                    # Sin datos de prueba, usar score de entrenamiento
                    y_pred = model.predict(X_train_poly)
                    score = r2_score(y_train, y_pred)
                    mse = mean_squared_error(y_train, y_pred)
            else:
                model.fit(X_train, y_train)
                if test_size > 0:
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                else:
                    # Sin datos de prueba, usar score de entrenamiento
                    y_pred = model.predict(X_train)
                    score = r2_score(y_train, y_pred)
                    mse = mean_squared_error(y_train, y_pred)
            
            results[name] = {'r2_score': score, 'mse': mse}
            
            if score > best_score:
                best_score = score
                best_model = (name, model, poly_features if name == 'Polynomial' else None)
                
        except Exception as e:
            print(f"[WARNING] Error entrenando modelo {name}: {e}")
            results[name] = {'r2_score': -999, 'mse': 999}
    
    print("[SUCCESS] Resultados de regresión:")
    for name, metrics in results.items():
        if metrics['r2_score'] > -999:
            print(f"{name}: R² = {metrics['r2_score']:.4f}, MSE = {metrics['mse']:.4f}")
    
    return best_model, results

# ==========================================
# MODELOS DE CLASIFICACIÓN AVANZADOS
# ==========================================

def build_advanced_classification_models(X, y):
    """Construye modelos de clasificación con optimización de hiperparámetros"""
    # Verificar distribución de clases
    class_counts = pd.Series(y).value_counts()
    print(f"[INFO] Distribución de clases: {dict(class_counts)}")
    
    # Verificar tamaño del dataset
    if len(X) < 10:
        print("[WARNING] Dataset muy pequeño para división train/test")
        X_train, X_test = X, X
        y_train, y_test = y, y
        test_size = 0
    else:
        test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                          random_state=42, stratify=y if len(class_counts) > 1 else None)
    
    print(f"[INFO] Entrenamiento: {len(X_train)} muestras, Prueba: {len(X_test)} muestras")
    
    # Modelos base (ajustados para datasets pequeños)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42, C=1.0),
        'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    }
    
    # Solo usar GridSearch si tenemos suficientes datos
    if len(X_train) > 50:
        try:
            rf_params = {
                'n_estimators': [30, 50, 100],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            }
            
            rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, 
                                 cv=min(3, len(X_train)//10), scoring='accuracy')
            rf_grid.fit(X_train, y_train)
            models['RandomForest_Optimized'] = rf_grid.best_estimator_
            print(f"[INFO] Mejores parámetros RF: {rf_grid.best_params_}")
        except Exception as e:
            print(f"[WARNING] Error en GridSearch: {e}")
    
    # Evaluar modelos
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            if test_size > 0:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
            else:
                # Sin datos de prueba, usar accuracy de entrenamiento
                y_pred = model.predict(X_train)
                accuracy = accuracy_score(y_train, y_pred)
            
            results[name] = accuracy
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = (name, model)
                
        except Exception as e:
            print(f"[WARNING] Error entrenando modelo {name}: {e}")
            results[name] = 0
    
    # Ensemble solo si tenemos múltiples modelos buenos
    good_models = [(name, model) for name, model in models.items() 
                   if name in results and results[name] > 0.5]
    
    if len(good_models) >= 2 and len(X_train) > 20:
        try:
            ensemble = VotingClassifier(
                estimators=good_models[:3],  # Máximo 3 modelos
                voting='soft'
            )
            
            ensemble.fit(X_train, y_train)
            if test_size > 0:
                y_pred_ensemble = ensemble.predict(X_test)
                ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            else:
                y_pred_ensemble = ensemble.predict(X_train)
                ensemble_accuracy = accuracy_score(y_train, y_pred_ensemble)
            
            results['Ensemble'] = ensemble_accuracy
            
            if ensemble_accuracy > best_score:
                best_model = ('Ensemble', ensemble)
                
        except Exception as e:
            print(f"[WARNING] Error creando ensemble: {e}")
    
    print("[SUCCESS] Resultados de clasificación:")
    for name, accuracy in results.items():
        if accuracy > 0:
            print(f"{name}: Precisión = {accuracy:.4f}")
    
    return best_model, results

# ==========================================
# FUNCIÓN PRINCIPAL MEJORADA
# ==========================================

def main():
    print(" Iniciando entrenamiento de modelos mejorados...")
    print("="*60)
    
    try:
        # Cargar y preprocesar datos
        df, extended_features = load_and_preprocess_data("Social_Bueno.xlsx")
        
        if len(extended_features) == 0:
            print(" No se encontraron características válidas para entrenar")
            return
        
        if len(df) < 5:
            print(" Insuficientes datos para entrenar (mínimo 5 muestras)")
            return
        
        # Verificar datos objetivo
        target_cols = ['Mental_Health_Score', 'Affects_Academic_Performance']
        available_targets = [col for col in target_cols if col in df.columns]
        
        if len(available_targets) == 0:
            print(" No se encontraron columnas objetivo válidas")
            return
        
        print(f" Columnas objetivo disponibles: {available_targets}")
        
        # Preparar datos
        X = df[extended_features].copy()
        print(f" Matriz de características: {X.shape}")
        
        # Crear directorio de modelos
        os.makedirs("model_advanced", exist_ok=True)
        
        # 1. CLUSTERING MEJORADO
        print("\n" + "="*50)
        print("1 ENTRENANDO MODELO DE CLUSTERING")
        print("="*50)
        
        kmeans, scaler, clusters, cluster_analysis = improved_clustering(X, extended_features)
        df['cluster_advanced'] = clusters
        
        # Guardar modelos de clustering
        joblib.dump(kmeans, "model_advanced/kmeans_advanced.pkl")
        joblib.dump(scaler, "model_advanced/scaler_advanced.pkl")
        joblib.dump(cluster_analysis, "model_advanced/cluster_analysis.pkl")
        print(" Clustering completado y guardado")
        
        # 2. REGRESIÓN AVANZADA
        reg_trained = False
        reg_features = []
        reg_results = {}
        best_reg_model = ("None", None, None)
        
        if 'Mental_Health_Score' in df.columns:
            print("\n" + "="*50)
            print("2 ENTRENANDO MODELOS DE REGRESIÓN")
            print("="*50)
            
            y_reg = df['Mental_Health_Score'].copy()
            
            # Limpiar datos para regresión
            valid_indices = ~y_reg.isnull()
            X_reg = X[valid_indices].copy()
            y_reg = y_reg[valid_indices]
            
            print(f" Datos válidos para regresión: {len(y_reg)} muestras")
            
            if len(y_reg) >= 5:
                # Seleccionar mejores características
                X_reg_selected, reg_features, reg_selector = select_best_features(
                    X_reg, y_reg, 'regression', k=min(15, len(extended_features))
                )
                
                # Entrenar modelos
                best_reg_model, reg_results = build_advanced_regression_models(
                    pd.DataFrame(X_reg_selected, columns=reg_features), y_reg, reg_features
                )
                
                # Guardar modelos de regresión
                joblib.dump(best_reg_model, "model_advanced/regression_best.pkl")
                joblib.dump(reg_selector, "model_advanced/regression_selector.pkl")
                joblib.dump(reg_features, "model_advanced/regression_features.pkl")
                reg_trained = True
                print(" Regresión completada y guardada")
            else:
                print(" Insuficientes datos válidos para regresión")
        else:
            print(" Saltando regresión - No se encontró 'Mental_Health_Score'")
        
        # 3. CLASIFICACIÓN AVANZADA
        clf_trained = False
        clf_features = []
        clf_results = {}
        best_clf_model = ("None", None)
        
        if 'Affects_Academic_Performance' in df.columns:
            print("\n" + "="*50)
            print("3 ENTRENANDO MODELOS DE CLASIFICACIÓN")
            print("="*50)
            
            y_clf = df['Affects_Academic_Performance'].copy()
            
            # Limpiar datos para clasificación
            valid_indices = ~y_clf.isnull()
            X_clf = X[valid_indices].copy()
            y_clf = y_clf[valid_indices]
            
            print(f" Datos válidos para clasificación: {len(y_clf)} muestras")
            
            if len(y_clf) >= 5:
                # Seleccionar mejores características
                X_clf_selected, clf_features, clf_selector = select_best_features(
                    X_clf, y_clf, 'classification', k=min(15, len(extended_features))
                )
                
                # Entrenar modelos
                best_clf_model, clf_results = build_advanced_classification_models(
                    pd.DataFrame(X_clf_selected, columns=clf_features), y_clf
                )
                
                # Guardar modelos de clasificación
                joblib.dump(best_clf_model, "model_advanced/classification_best.pkl")
                joblib.dump(clf_selector, "model_advanced/classification_selector.pkl")
                joblib.dump(clf_features, "model_advanced/classification_features.pkl")
                clf_trained = True
                print(" Clasificación completada y guardada")
            else:
                print(" Insuficientes datos válidos para clasificación")
        else:
            print(" Saltando clasificación - No se encontró 'Affects_Academic_Performance'")
        
        # 4. GUARDAR INFORMACIÓN GENERAL DEL MODELO
        model_info = {
            'dataset_info': {
                'total_samples': len(df),
                'total_features': len(extended_features),
                'training_date': pd.Timestamp.now().isoformat()
            },
            'extended_features': extended_features,
            'regression_features': reg_features,
            'classification_features': clf_features,
            'best_regression_model': best_reg_model[0],
            'best_classification_model': best_clf_model[0],
            'regression_results': reg_results,
            'classification_results': clf_results,
            'cluster_analysis': cluster_analysis,
            'models_trained': {
                'clustering': True,
                'regression': reg_trained,
                'classification': clf_trained
            }
        }
        
        joblib.dump(model_info, "model_advanced/model_info.pkl")
        
        # 5. RESUMEN FINAL
        print("\n" + "="*60)
        print(" ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print("="*60)
        print(f" Archivos guardados en: ./model_advanced/")
        print(f" Dataset: {len(df)} muestras, {len(extended_features)} características")
        print(f" Clustering:  Entrenado ({len(set(clusters))} grupos)")
        print(f" Regresión: {' Entrenado' if reg_trained else '❌ No entrenado'}")
        if reg_trained:
            print(f"Mejor modelo: {best_reg_model[0]}")
            if reg_results:
                best_score = max(reg_results.values(), key=lambda x: x.get('r2_score', -999))
                print(f"Mejor R²: {best_score.get('r2_score', 0):.4f}")
        
        print(f"Clasificación: {' Entrenado' if clf_trained else '❌ No entrenado'}")
        if clf_trained:
            print(f"    Mejor modelo: {best_clf_model[0]}")
            if clf_results:
                best_acc = max(clf_results.values())
                print(f"   Mejor precisión: {best_acc:.4f}")
        
        print("\n El sistema está listo para hacer predicciones!")
        
    except Exception as e:
        print(f"\n ERROR DURANTE EL ENTRENAMIENTO:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()