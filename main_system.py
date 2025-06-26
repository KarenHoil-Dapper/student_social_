#!/usr/bin/env python3
"""
Sistema Principal de Evaluación de Salud Mental y Redes Sociales
================================================================

Este sistema integra:
1. Entrenamiento de modelos de IA
2. Encuestas interactivas completas
3. Predicciones y recomendaciones personalizadas
4. Re-entrenamiento automático
5. Interfaz web moderna
6. Gestión avanzada de datos
7. Cálculo automático de adicción

Autor: Sistema de IA para Salud Mental
Fecha: 2024
Versión: 2.0 (Actualizada)
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
import json

class SocialMediaHealthSystem:
    """Sistema principal que gestiona todos los componentes"""
    
    def __init__(self):
        self.components = {
            'train_model.py': 'Script de entrenamiento de modelos',
            'updated_survey_system.py': 'Sistema de encuestas y predicciones (actualizado)',
            'auto_retrain_system.py': 'Sistema de re-entrenamiento automático',
            'web_interface.py': 'Interfaz web con Flask',
        }
        
        self.data_file = "Social_Bueno.xlsx"
        self.model_path = "model_advanced"
        
    def check_dependencies(self):
        """Verifica que todas las dependencias estén instaladas"""
        # Mapeo de nombres de paquetes pip a nombres de módulos Python
        required_modules = {
            'pandas': 'pandas',
            'numpy': 'numpy', 
            'sklearn': 'scikit-learn',  # sklearn es el módulo, scikit-learn es el paquete pip
            'joblib': 'joblib',
            'openpyxl': 'openpyxl',
            'flask': 'flask'
        }
        
        missing_packages = []
        
        for module_name, pip_name in required_modules.items():
            try:
                __import__(module_name)
            except ImportError:
                missing_packages.append(pip_name)
        
        # Verificar warnings (viene con Python estándar)
        try:
            import warnings
        except ImportError:
            print("⚠️ Problema con módulo warnings (debería estar incluido en Python)")
        
        if missing_packages:
            print("❌ Paquetes faltantes:")
            for package in missing_packages:
                print(f"   - {package}")
            print("\n💡 Instala con: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ Todas las dependencias están instaladas")
        return True
    
    def check_data_file(self):
        """Verifica que el archivo de datos exista"""
        if not os.path.exists(self.data_file):
            print(f"❌ Archivo de datos no encontrado: {self.data_file}")
            print("💡 Asegúrate de tener el archivo Excel con los datos de entrenamiento")
            return False
        
        print(f"✅ Archivo de datos encontrado: {self.data_file}")
        
        # Verificar contenido del archivo
        try:
            import pandas as pd
            df = pd.read_excel(self.data_file)
            print(f"📊 Archivo contiene: {len(df)} registros y {len(df.columns)} columnas")
            
            # Verificar columnas importantes
            required_cols = ['Mental_Health_Score', 'Affects_Academic_Performance']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ Columnas importantes faltantes: {missing_cols}")
            else:
                print("✅ Columnas objetivo encontradas")
                
        except Exception as e:
            print(f"⚠️ Error verificando archivo: {e}")
        
        return True
    
    def setup_environment(self):
        """Configura el entorno inicial"""
        print("🔧 Configurando entorno...")
        
        # Crear directorios necesarios
        directories = [self.model_path, 'logs', 'backups', 'templates']
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Directorio creado: {directory}")
        
        print("✅ Entorno configurado")
    
    def train_initial_models(self):
        """Entrena los modelos iniciales"""
        print("\n🤖 ENTRENAMIENTO INICIAL DE MODELOS")
        print("="*50)
        
        if os.path.exists(f"{self.model_path}/model_info.pkl"):
            response = input("Los modelos ya existen. ¿Re-entrenar? (s/n): ").strip().lower()
            if response not in ['s', 'si', 'sí', 'y', 'yes']:
                print("⏭️ Saltando entrenamiento")
                return True
        
        try:
            # Verificar que existe el script de entrenamiento
            if not os.path.exists("train_model.py"):
                print("❌ No se encontró train_model.py")
                print("💡 Asegúrate de tener el script de entrenamiento en el directorio")
                return False
                
            # Ejecutar script de entrenamiento
            print("🔄 Ejecutando entrenamiento de modelos...")
            result = subprocess.run([sys.executable, "train_model.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Modelos entrenados exitosamente")
                print("\n📋 Salida del entrenamiento:")
                print(result.stdout)
                return True
            else:
                print(f"❌ Error en entrenamiento:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando entrenamiento: {e}")
            return False
    
    def run_console_survey(self):
        """Ejecuta encuesta por consola"""
        print("\n📝 ENCUESTA POR CONSOLA")
        print("="*30)
        
        try:
            # Verificar que existe el sistema de encuestas actualizado
            if not os.path.exists("updated_survey_system.py"):
                print("❌ No se encontró updated_survey_system.py")
                print("💡 Asegúrate de tener el sistema de encuestas actualizado")
                return
                
            from updated_survey_system import SocialMediaHealthPredictor
            predictor = SocialMediaHealthPredictor()
            predictor.run_survey_session()
        except ImportError as e:
            print(f"❌ Error importando sistema de encuestas: {e}")
            print("💡 Verifica que updated_survey_system.py esté en el directorio")
        except Exception as e:
            print(f"❌ Error en encuesta: {e}")
    
    def run_web_interface(self):
        """Inicia la interfaz web"""
        print("\n🌐 INTERFAZ WEB")
        print("="*20)
        
        try:
            # Verificar que existe la interfaz web
            if not os.path.exists("web_interface.py"):
                print("❌ No se encontró web_interface.py")
                print("💡 Asegúrate de tener el archivo de interfaz web")
                return
                
            from web_interface import run_server
            run_server()
        except ImportError as e:
            print(f"❌ Error importando interfaz web: {e}")
            print("💡 Verifica que web_interface.py esté en el directorio")
        except Exception as e:
            print(f"❌ Error iniciando interfaz web: {e}")
    
    def run_retrain_check(self):
        """Ejecuta verificación de re-entrenamiento"""
        print("\n🔄 VERIFICACIÓN DE RE-ENTRENAMIENTO")
        print("="*40)
        
        try:
            if not os.path.exists("auto_retrain_system.py"):
                print("❌ No se encontró auto_retrain_system.py")
                print("💡 Asegúrate de tener el sistema de re-entrenamiento")
                return
                
            from auto_retrain_system import AutoRetrainSystem
            retrain_system = AutoRetrainSystem()
            retrain_system.run_auto_retrain_check()
        except ImportError as e:
            print(f"❌ Error importando sistema de re-entrenamiento: {e}")
        except Exception as e:
            print(f"❌ Error en verificación: {e}")
    
    def show_system_status(self):
        """Muestra el estado del sistema"""
        print("\n📊 ESTADO DEL SISTEMA")
        print("="*25)
        
        # Verificar archivos clave
        key_files = {
            self.data_file: "Archivo de datos",
            "train_model.py": "Script de entrenamiento",
            "updated_survey_system.py": "Sistema de encuestas (actualizado)",
            "web_interface.py": "Interfaz web",
            "auto_retrain_system.py": "Sistema de re-entrenamiento",
            f"{self.model_path}/model_info.pkl": "Información de modelos",
            f"{self.model_path}/kmeans_advanced.pkl": "Modelo de clustering",
            f"{self.model_path}/regression_best.pkl": "Modelo de regresión",
            f"{self.model_path}/classification_best.pkl": "Modelo de clasificación"
        }
        
        print("\n🔍 ARCHIVOS DEL SISTEMA:")
        for file_path, description in key_files.items():
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✅ {description}: {size} bytes")
            else:
                print(f"❌ {description}: No encontrado")
        
        # Estadísticas de datos
        try:
            import pandas as pd
            if os.path.exists(self.data_file):
                df = pd.read_excel(self.data_file)
                print(f"\n📈 ESTADÍSTICAS DE DATOS:")
                print(f"   📊 Total de muestras: {len(df)}")
                print(f"   📋 Columnas: {len(df.columns)}")
                
                # Verificar columnas importantes
                important_cols = [
                    'Mental_Health_Score', 'Affects_Academic_Performance', 
                    'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                    'Addicted_Score', 'Anxiety_Level'
                ]
                
                existing_cols = [col for col in important_cols if col in df.columns]
                print(f"   ✅ Columnas importantes: {len(existing_cols)}/{len(important_cols)}")
                
                if 'survey_date' in df.columns:
                    df['survey_date'] = pd.to_datetime(df['survey_date'], errors='coerce')
                    recent_count = len(df[df['survey_date'] >= pd.Timestamp.now() - pd.Timedelta(days=7)])
                    print(f"   📅 Muestras última semana: {recent_count}")
                
                # Estadísticas básicas
                if 'Mental_Health_Score' in df.columns:
                    avg_mental = df['Mental_Health_Score'].mean()
                    print(f"   🧠 Salud mental promedio: {avg_mental:.2f}/10")
                
                if 'Affects_Academic_Performance' in df.columns:
                    academic_impact = (df['Affects_Academic_Performance'].sum() / len(df)) * 100
                    print(f"   📚 Impacto académico: {academic_impact:.1f}%")
        
        except Exception as e:
            print(f"⚠️ Error obteniendo estadísticas: {e}")
        
        # Estado de modelos
        print(f"\n🤖 ESTADO DE MODELOS:")
        models_trained = {
            'clustering': os.path.exists(f"{self.model_path}/kmeans_advanced.pkl"),
            'regression': os.path.exists(f"{self.model_path}/regression_best.pkl"),
            'classification': os.path.exists(f"{self.model_path}/classification_best.pkl")
        }
        
        for model_type, exists in models_trained.items():
            status = "✅ Entrenado" if exists else "❌ No entrenado"
            print(f"   {model_type.title()}: {status}")
    
    def create_sample_data(self):
        """Crea datos de ejemplo si no existen"""
        if os.path.exists(self.data_file):
            print(f"✅ {self.data_file} ya existe")
            response = input("¿Quieres crear un archivo de ejemplo diferente? (s/n): ").strip().lower()
            if response not in ['s', 'si', 'sí', 'y', 'yes']:
                return
            self.data_file = f"Social_Ejemplo_{datetime.now().strftime('%Y%m%d')}.xlsx"
        
        print(f"📝 Creando archivo de datos de ejemplo: {self.data_file}")
        
        import pandas as pd
        import numpy as np
        
        # Generar datos de ejemplo más realistas
        np.random.seed(42)
        n_samples = 200
        
        print(f"🔄 Generando {n_samples} muestras de ejemplo...")
        
        data = {
            'Student_ID': range(1, n_samples + 1),
            'Age': np.random.randint(16, 35, n_samples),
            'Gender': np.random.choice([0, 1], n_samples),
            'Avg_Daily_Usage_Hours': np.random.normal(4.5, 2, n_samples).clip(0.5, 12),
            'Sleep_Hours_Per_Night': np.random.normal(7, 1.2, n_samples).clip(4, 10),
            'Physical_Activity_Hours': np.random.normal(3, 2, n_samples).clip(0, 15),
            'Conflicts_Over_Social_Media': np.random.randint(0, 5, n_samples),
            
            # Nuevas características de bienestar mental
            'Anxiety_Level': np.random.randint(1, 11, n_samples),
            'Mood_Changes': np.random.randint(1, 6, n_samples),
            'Social_Comparison': np.random.randint(1, 6, n_samples),
            'FOMO_Level': np.random.randint(1, 6, n_samples),
            
            # Características de productividad
            'Concentration_Issues': np.random.randint(1, 6, n_samples),
            'Procrastination': np.random.randint(1, 6, n_samples),
            'Productivity_Impact': np.random.randint(1, 6, n_samples),
            
            # Características sociales
            'Face_to_Face_Preference': np.random.randint(1, 6, n_samples),
            'Online_vs_Offline_Friends': np.random.choice([0, 1], n_samples),
            
            # Patrones de uso
            'Platforms_Used': np.random.randint(1, 8, n_samples),
            'Posting_Frequency': np.random.randint(1, 6, n_samples),
            'Scrolling_Before_Bed': np.random.randint(1, 6, n_samples),
            'Notification_Frequency': np.random.randint(1, 6, n_samples)
        }
        
        # Nivel académico (one-hot)
        academic_levels = np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.6, 0.2])
        data['Academic_Level_High School'] = (academic_levels == 1).astype(int)
        data['Academic_Level_Undergraduate'] = (academic_levels == 2).astype(int)
        data['Academic_Level_Graduate'] = (academic_levels == 3).astype(int)
        
        # Plataformas (one-hot) - simplificado
        platforms = ['Facebook', 'Instagram', 'TikTok', 'YouTube', 'WhatsApp', 'Twitter']
        for platform in platforms:
            data[f'Most_Used_Platform_{platform}'] = 0
        
        # Asignar plataforma principal aleatoria
        main_platforms = np.random.choice(platforms, n_samples)
        for i, platform in enumerate(main_platforms):
            data[f'Most_Used_Platform_{platform}'][i] = 1
        
        # Estado de relación (one-hot)
        relationship_status = np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.3, 0.2])
        data['Relationship_Status_Single'] = (relationship_status == 1).astype(int)
        data['Relationship_Status_In Relationship'] = (relationship_status == 2).astype(int)
        data['Relationship_Status_Complicated'] = (relationship_status == 3).astype(int)
        
        # Países (simplificado - solo algunos países principales)
        countries = ['USA', 'Mexico', 'Canada', 'Spain', 'Argentina']
        for country in countries:
            data[f'Country_{country}'] = 0
        
        main_countries = np.random.choice(countries, n_samples)
        for i, country in enumerate(main_countries):
            data[f'Country_{country}'][i] = 1
        
        # Calcular Addicted_Score basado en otras variables
        addiction_scores = []
        for i in range(n_samples):
            score = (
                data['Avg_Daily_Usage_Hours'][i] * 1.0 +
                data['Posting_Frequency'][i] * 0.8 +
                data['Notification_Frequency'][i] * 0.7 +
                data['FOMO_Level'][i] * 0.9 +
                data['Scrolling_Before_Bed'][i] * 0.8 +
                data['Concentration_Issues'][i] * 0.6
            ) / 6.0
            addiction_scores.append(min(10, max(1, round(score, 1))))
        
        data['Addicted_Score'] = addiction_scores
        
        # Crear variables objetivo basadas en otras variables
        mental_health_scores = []
        academic_impacts = []
        
        for i in range(n_samples):
            # Mental Health Score (1-10, inverso a factores negativos)
            mh_base = 8 - (
                data['Avg_Daily_Usage_Hours'][i] * 0.2 +
                data['Anxiety_Level'][i] * 0.3 +
                data['Addicted_Score'][i] * 0.2 +
                data['Social_Comparison'][i] * 0.1
            ) / 4
            
            mental_health = max(1, min(10, mh_base + np.random.normal(0, 0.8)))
            mental_health_scores.append(round(mental_health, 1))
            
            # Academic Impact (0 o 1)
            impact_prob = (
                data['Avg_Daily_Usage_Hours'][i] * 0.15 +
                data['Procrastination'][i] * 0.25 +
                data['Concentration_Issues'][i] * 0.2 +
                data['Addicted_Score'][i] * 0.1
            ) / 10
            
            academic_impacts.append(1 if np.random.random() < impact_prob else 0)
        
        data['Mental_Health_Score'] = mental_health_scores
        data['Affects_Academic_Performance'] = academic_impacts
        
        # Crear DataFrame y guardar
        df = pd.DataFrame(data)
        df.to_excel(self.data_file, index=False)
        
        print(f"✅ Archivo creado con {n_samples} muestras de ejemplo")
        print(f"📊 Características incluidas: {len(df.columns)}")
        print(f"🧠 Salud mental promedio: {np.mean(mental_health_scores):.2f}/10")
        print(f"📚 Impacto académico: {np.mean(academic_impacts)*100:.1f}%")
    
    def run_quick_test(self):
        """Ejecuta una prueba rápida del sistema"""
        print("\n🧪 PRUEBA RÁPIDA DEL SISTEMA")
        print("="*35)
        
        # 1. Verificar archivos
        print("1. Verificando archivos...")
        if not self.check_data_file():
            print("❌ Prueba fallida: archivo de datos faltante")
            return
        
        # 2. Verificar modelos
        print("\n2. Verificando modelos...")
        if not os.path.exists(f"{self.model_path}/model_info.pkl"):
            print("⚠️ Modelos no encontrados, entrenando...")
            if not self.train_initial_models():
                print("❌ Prueba fallida: error en entrenamiento")
                return
        
        # 3. Prueba de importación
        print("\n3. Probando importaciones...")
        try:
            from updated_survey_system import SocialMediaHealthPredictor
            predictor = SocialMediaHealthPredictor()
            print("✅ Sistema de encuestas importado correctamente")
        except Exception as e:
            print(f"❌ Error importando sistema: {e}")
            return
        
        # 4. Prueba básica de predicción
        print("\n4. Probando predicción básica...")
        try:
            test_data = {
                'Age': 22,
                'Gender': 1,
                'Avg_Daily_Usage_Hours': 4,
                'Sleep_Hours_Per_Night': 7,
                'anxiety_level': 5,
                'academic_level': 2,
                'main_platform': 2,
                'relationship_status': 1
            }
            
            prepared = predictor.prepare_features_for_model(test_data)
            predictions = predictor.make_predictions(prepared)
            
            print("✅ Predicción básica exitosa")
            if 'mental_health_score' in predictions:
                print(f"   🧠 Salud mental predicha: {predictions['mental_health_score']:.2f}/10")
            if 'affects_academic_performance' in predictions:
                print(f"   📚 Impacto académico: {'Sí' if predictions['affects_academic_performance'] else 'No'}")
                
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return
        
        print("\n🎉 ¡Prueba rápida completada exitosamente!")
        print("💡 El sistema está listo para usar")
    
    def show_menu(self):
        """Muestra el menú principal"""
        print("\n" + "="*70)
        print("🧠 SISTEMA DE EVALUACIÓN DE SALUD MENTAL Y REDES SOCIALES v2.0")
        print("="*70)
        print("1. 🔧 Configurar sistema (primera vez)")
        print("2. 🤖 Entrenar/Re-entrenar modelos")
        print("3. 📝 Realizar encuesta (consola)")
        print("4. 🌐 Iniciar interfaz web")
        print("5. 🔄 Verificar re-entrenamiento")
        print("6. 📊 Ver estado del sistema")
        print("7. 📋 Crear datos de ejemplo")
        print("8. 🧪 Prueba rápida del sistema")
        print("9. ❓ Ayuda")
        print("10. 🚪 Salir")
        print("="*70)
    
    def show_help(self):
        """Muestra información de ayuda"""
        help_text = """
🆘 AYUDA DEL SISTEMA v2.0
========================

NUEVAS CARACTERÍSTICAS:
- ✨ Cálculo automático de adicción (no más preguntas manuales)
- 📊 22+ preguntas para análisis completo
- 🧠 Indicadores de bienestar mental avanzados
- 📱 Patrones de uso detallados
- 💡 Recomendaciones personalizadas mejoradas

CONFIGURACIÓN INICIAL:
1. Ejecuta "Configurar sistema" para setup inicial
2. Asegúrate de tener el archivo Excel con datos (Social_Bueno.xlsx)
3. Entrena los modelos antes de usar predicciones

COMPONENTES:
- Entrenamiento: Crea modelos de IA con tus datos
- Encuestas: Recolecta datos y hace predicciones en tiempo real
- Re-entrenamiento: Mejora automática cuando llegan nuevos datos
- Interfaz Web: Acceso moderno vía navegador
- Cálculo de Adicción: Automático basado en patrones de uso

ARCHIVOS IMPORTANTES:
- Social_Bueno.xlsx: Datos de entrenamiento
- model_advanced/: Modelos entrenados
- updated_survey_system.py: Sistema de encuestas actualizado
- logs/: Registros del sistema

NUEVAS PREGUNTAS INCLUYEN:
📋 Bienestar Mental: Ansiedad, FOMO, comparación social, cambios humor
📋 Productividad: Concentración, procrastinación, impacto productividad
📋 Actividad Social: Ejercicio, preferencias sociales, amistades
📋 Patrones Uso: Notificaciones, publicaciones, uso nocturno

FLUJO RECOMENDADO:
1. Configurar sistema
2. Entrenar modelos
3. Usar encuestas (consola o web)
4. Re-entrenar cuando acumules datos
5. Monitorear con estadísticas

PUNTUACIÓN DE ADICCIÓN:
Se calcula automáticamente considerando:
- Horas de uso diario
- Frecuencia de publicación  
- Nivel de notificaciones
- FOMO y problemas de concentración
- Uso antes de dormir

Para más información, revisa la documentación
en los archivos de código.
"""
        print(help_text)
    
    def run(self):
        """Ejecuta el sistema principal"""
        print("🚀 Iniciando Sistema de Evaluación de Salud Mental v2.0...")
        
        # Verificar dependencias básicas
        if not self.check_dependencies():
            return
        
        while True:
            self.show_menu()
            
            try:
                choice = input("\nSelecciona una opción (1-10): ").strip()
                
                if choice == "1":
                    print("\n🔧 CONFIGURACIÓN INICIAL")
                    print("="*25)
                    self.setup_environment()
                    self.check_data_file()
                    print("✅ Configuración completada")
                
                elif choice == "2":
                    if not self.check_data_file():
                        continue
                    self.train_initial_models()
                
                elif choice == "3":
                    self.run_console_survey()
                
                elif choice == "4":
                    self.run_web_interface()
                
                elif choice == "5":
                    self.run_retrain_check()
                
                elif choice == "6":
                    self.show_system_status()
                
                elif choice == "7":
                    self.create_sample_data()
                
                elif choice == "8":
                    self.run_quick_test()
                
                elif choice == "9":
                    self.show_help()
                
                elif choice == "10":
                    print("\n👋 ¡Gracias por usar el sistema!")
                    break
                
                else:
                    print("❌ Opción inválida. Selecciona 1-10.")
                
                input("\nPresiona Enter para continuar...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Sistema interrumpido por el usuario")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("\nPresiona Enter para continuar...")

def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Sistema de Evaluación de Salud Mental y Redes Sociales v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main_system.py                    # Menú interactivo
  python main_system.py --train           # Solo entrenar modelos
  python main_system.py --survey          # Solo hacer encuesta
  python main_system.py --web             # Solo interfaz web
  python main_system.py --status          # Solo mostrar estado
  python main_system.py --setup           # Solo configurar
  python main_system.py --test            # Prueba rápida
        """
    )
    
    parser.add_argument('--train', action='store_true', 
                       help='Entrenar modelos y salir')
    parser.add_argument('--survey', action='store_true', 
                       help='Ejecutar encuesta por consola y salir')
    parser.add_argument('--web', action='store_true', 
                       help='Iniciar interfaz web y salir')
    parser.add_argument('--retrain', action='store_true', 
                       help='Verificar re-entrenamiento y salir')
    parser.add_argument('--status', action='store_true', 
                       help='Mostrar estado del sistema y salir')
    parser.add_argument('--setup', action='store_true', 
                       help='Configurar sistema y salir')
    parser.add_argument('--create-sample', action='store_true', 
                       help='Crear datos de ejemplo y salir')
    parser.add_argument('--test', action='store_true', 
                       help='Ejecutar prueba rápida y salir')
    
    args = parser.parse_args()
    
    system = SocialMediaHealthSystem()
    
    # Ejecutar acción específica si se proporciona
    if args.setup:
        system.setup_environment()
        system.check_data_file()
    elif args.train:
        if system.check_data_file():
            system.train_initial_models()
    elif args.survey:
        system.run_console_survey()
    elif args.web:
        system.run_web_interface()
    elif args.retrain:
        system.run_retrain_check()
    elif args.status:
        system.show_system_status()
    elif args.create_sample:
        system.create_sample_data()
    elif args.test:
        system.run_quick_test()
    else:
        # Ejecutar menú interactivo
        system.run()

if __name__ == "__main__":
    main()