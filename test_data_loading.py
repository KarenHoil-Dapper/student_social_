import pandas as pd
import numpy as np

def test_data_loading(file_path):
    """Script de prueba para verificar la carga de datos"""
    
    print("🔍 VERIFICANDO DATOS")
    print("="*40)
    
    try:
        # Cargar archivo
        df = pd.read_excel(file_path)
        print(f"✅ Archivo cargado exitosamente")
        print(f"📊 Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        
        # Mostrar información básica
        print(f"\n📋 COLUMNAS DISPONIBLES:")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            print(f"  {i:2}. {col:<30} | Tipo: {dtype:<10} | Nulls: {null_count}")
        
        # Verificar columnas objetivo
        target_cols = ['Mental_Health_Score', 'Affects_Academic_Performance']
        print(f"\n🎯 COLUMNAS OBJETIVO:")
        for col in target_cols:
            if col in df.columns:
                print(f"  ✅ {col} - Presente")
                if df[col].dtype in ['int64', 'float64']:
                    print(f"     Rango: {df[col].min():.2f} - {df[col].max():.2f}")
                else:
                    print(f"     Valores únicos: {df[col].unique()}")
            else:
                print(f"  ❌ {col} - No encontrada")
        
        # Identificar columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\n🔢 COLUMNAS NUMÉRICAS ({len(numeric_cols)}):")
        for col in numeric_cols:
            print(f"  • {col}")
        
        # Verificar datos faltantes
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            print(f"\n⚠️ DATOS FALTANTES:")
            for col, count in missing_data.items():
                percentage = (count / len(df)) * 100
                print(f"  • {col}: {count} ({percentage:.1f}%)")
        else:
            print(f"\n✅ No hay datos faltantes")
        
        # Estadísticas básicas de columnas numéricas importantes
        important_cols = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media']
        available_important = [col for col in important_cols if col in df.columns]
        
        if available_important:
            print(f"\n📈 ESTADÍSTICAS BÁSICAS:")
            stats = df[available_important].describe()
            print(stats.round(2))
        
        return True, df
        
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {file_path}")
        print("   Verifica que el archivo existe en la ubicación correcta")
        return False, None
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return False, None

def create_sample_data(file_path):
    """Crea datos de muestra si el archivo no existe"""
    print("\n🆕 CREANDO DATOS DE MUESTRA")
    print("="*30)
    
    # Generar datos sintéticos
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'User_ID': range(1, n_samples + 1),
        'Age': np.random.randint(13, 25, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'Avg_Daily_Usage_Hours': np.random.exponential(3, n_samples).round(1),
        'Sleep_Hours_Per_Night': np.random.normal(7, 1.5, n_samples).round(1),
        'Conflicts_Over_Social_Media': np.random.randint(0, 11, n_samples),
        'Social_Media_Platforms_Used': np.random.randint(1, 8, n_samples),
        'Time_Spent_Per_Platform': np.random.exponential(1, n_samples).round(1),
        'Mental_Health_Score': np.random.normal(6, 2, n_samples).round(1),
        'Affects_Academic_Performance': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    # Ajustar rangos
    data['Avg_Daily_Usage_Hours'] = np.clip(data['Avg_Daily_Usage_Hours'], 0.5, 12)
    data['Sleep_Hours_Per_Night'] = np.clip(data['Sleep_Hours_Per_Night'], 4, 12)
    data['Mental_Health_Score'] = np.clip(data['Mental_Health_Score'], 1, 10)
    
    # Crear correlaciones realistas
    for i in range(n_samples):
        # Más uso de redes sociales puede afectar el sueño
        if data['Avg_Daily_Usage_Hours'][i] > 6:
            data['Sleep_Hours_Per_Night'][i] = max(4, data['Sleep_Hours_Per_Night'][i] - np.random.uniform(0, 2))
        
        # Más conflictos pueden afectar la salud mental
        if data['Conflicts_Over_Social_Media'][i] > 7:
            data['Mental_Health_Score'][i] = max(1, data['Mental_Health_Score'][i] - np.random.uniform(0, 3))
    
    df = pd.DataFrame(data)
    
    try:
        df.to_excel(file_path, index=False)
        print(f"✅ Datos de muestra creados: {file_path}")
        print(f"📊 {len(df)} registros generados")
        return True, df
    except Exception as e:
        print(f"❌ Error creando archivo: {e}")
        return False, None

def main():
    file_path = "Social_Bueno.xlsx"
    
    # Intentar cargar datos existentes
    success, df = test_data_loading(file_path)
    
    if not success:
        # Si no existe, preguntar si crear datos de muestra
        response = input("\n¿Quieres crear datos de muestra? (s/n): ").lower()
        if response in ['s', 'si', 'yes', 'y']:
            success, df = create_sample_data(file_path)
            if success:
                print("\nVerificando datos creados...")
                test_data_loading(file_path)
        else:
            print("❌ No se puede continuar sin datos")
            return
    
    if success:
        print(f"\n✅ LISTO PARA ENTRENAR MODELOS")
        print("   Ejecuta el script principal para entrenar los modelos")

if __name__ == "__main__":
    main()