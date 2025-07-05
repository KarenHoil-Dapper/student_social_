import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import json
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AutoRetrainSystem:
    """Sistema de re-entrenamiento automático de modelos"""
    
    def __init__(self, data_file="Social_Bueno.xlsx", model_path="model_advanced"):
        self.data_file = data_file
        self.model_path = model_path
        self.backup_path = f"{model_path}_backup"
        self.retrain_config = {
            'min_new_samples': 10,  # Mínimo de nuevas muestras para re-entrenar
            'retrain_frequency_days': 7,  # Re-entrenar cada 7 días máximo
            'performance_threshold': 0.95,  # Si el rendimiento baja del 95%, re-entrenar
            'backup_old_models': True
        }
    
    def check_retrain_needed(self):
        """Verifica si es necesario re-entrenar el modelo"""
        reasons = []
        
        try:
            # 1. Verificar si existe archivo de configuración de último entrenamiento
            last_train_file = f"{self.model_path}/last_training_info.json"
            
            if os.path.exists(last_train_file):
                with open(last_train_file, 'r') as f:
                    last_train_info = json.load(f)
                
                # Verificar fecha del último entrenamiento
                last_train_date = datetime.fromisoformat(last_train_info['timestamp'])
                days_since_training = (datetime.now() - last_train_date).days
                
                if days_since_training >= self.retrain_config['retrain_frequency_days']:
                    reasons.append(f"Han pasado {days_since_training} días desde el último entrenamiento")
                
                # Verificar nuevas muestras
                if os.path.exists(self.data_file):
                    current_df = pd.read_excel(self.data_file)
                    new_samples = len(current_df) - last_train_info.get('total_samples', 0)
                    
                    if new_samples >= self.retrain_config['min_new_samples']:
                        reasons.append(f"Se han agregado {new_samples} nuevas muestras")
            
            else:
                reasons.append("No hay información de entrenamiento previo")
            
            # 2. Verificar rendimiento del modelo actual (si es posible)
            if self.can_evaluate_current_model():
                current_performance = self.evaluate_current_model()
                if current_performance < self.retrain_config['performance_threshold']:
                    reasons.append(f"El rendimiento del modelo ha bajado a {current_performance:.3f}")
            
        except Exception as e:
            reasons.append(f"Error verificando necesidad de re-entrenamiento: {e}")
        
        return len(reasons) > 0, reasons
    
    def can_evaluate_current_model(self):
        """Verifica si se puede evaluar el modelo actual"""
        try:
            # Verificar que existan suficientes datos y modelos
            if not os.path.exists(self.data_file):
                return False
            
            df = pd.read_excel(self.data_file)
            if len(df) < 20:  # Necesitamos al menos 20 muestras para evaluación
                return False
            
            # Verificar que existan los modelos
            model_files = ['model_info.pkl', 'regression_best.pkl', 'classification_best.pkl']
            for file in model_files:
                if not os.path.exists(f"{self.model_path}/{file}"):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def evaluate_current_model(self):
        """Evalúa el rendimiento del modelo actual con datos recientes"""
        try:
            # Cargar datos
            df = pd.read_excel(self.data_file)
            
            # Usar solo datos recientes para evaluación (últimos 30 días si hay fecha)
            if 'survey_date' in df.columns:
                df['survey_date'] = pd.to_datetime(df['survey_date'], errors='coerce')
                recent_date = datetime.now() - timedelta(days=30)
                df_recent = df[df['survey_date'] >= recent_date]
                if len(df_recent) < 10:
                    df_recent = df.tail(20)  # Usar últimas 20 muestras si no hay suficientes recientes
            else:
                df_recent = df.tail(20)  # Usar últimas 20 muestras
            
            # Cargar modelo info
            model_info = joblib.load(f"{self.model_path}/model_info.pkl")
            
            # Preparar características
            extended_features = model_info.get('extended_features', [])
            available_features = [col for col in extended_features if col in df_recent.columns]
            
            if len(available_features) == 0:
                return 0.5  # Score neutral si no hay características disponibles
            
            X = df_recent[available_features].fillna(0)
            
            scores = []
            
            # Evaluar regresión si existe
            if 'mental_health_score' in df_recent.columns and len(df_recent['mental_health_score'].dropna()) > 5:
                y_reg = df_recent['mental_health_score'].fillna(df_recent['mental_health_score'].mean())
                
                # Cargar modelo de regresión
                try:
                    reg_model = joblib.load(f"{self.model_path}/regression_best.pkl")
                    reg_selector = joblib.load(f"{self.model_path}/regression_selector.pkl")
                    
                    X_reg_selected = reg_selector.transform(X)
                    model_name, model, poly_features = reg_model
                    
                    if poly_features is not None:
                        X_reg_selected = poly_features.transform(X_reg_selected)
                    
                    y_pred = model.predict(X_reg_selected)
                    r2 = r2_score(y_reg, y_pred)
                    scores.append(max(0, r2))  # Asegurar que no sea negativo
                    
                except Exception as e:
                    print(f"Error evaluando regresión: {e}")
            
            # Evaluar clasificación si existe
            if 'Affects_Academic_Performance' in df_recent.columns and len(df_recent['Affects_Academic_Performance'].dropna()) > 5:
                y_clf = df_recent['Affects_Academic_Performance'].fillna(0)
                
                try:
                    clf_model = joblib.load(f"{self.model_path}/classification_best.pkl")
                    clf_selector = joblib.load(f"{self.model_path}/classification_selector.pkl")
                    
                    X_clf_selected = clf_selector.transform(X)
                    model_name, model = clf_model
                    
                    y_pred = model.predict(X_clf_selected)
                    accuracy = accuracy_score(y_clf, y_pred)
                    scores.append(accuracy)
                    
                except Exception as e:
                    print(f"Error evaluando clasificación: {e}")
            
            # Retornar promedio de scores o score neutral
            return np.mean(scores) if scores else 0.5
            
        except Exception as e:
            print(f"Error en evaluación del modelo: {e}")
            return 0.5
    
    def backup_current_models(self):
        """Hace backup de los modelos actuales"""
        try:
            if os.path.exists(self.model_path):
                # Crear directorio de backup con timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = f"{self.backup_path}_{timestamp}"
                
                shutil.copytree(self.model_path, backup_dir)
                print(f"✅ Backup creado en: {backup_dir}")
                
                # Mantener solo los últimos 5 backups
                self.cleanup_old_backups()
                
                return backup_dir
        except Exception as e:
            print(f"❌ Error creando backup: {e}")
            return None
    
    def cleanup_old_backups(self):
        """Limpia backups antiguos, manteniendo solo los 5 más recientes"""
        try:
            backup_dirs = []
            for item in os.listdir('.'):
                if item.startswith(f"{os.path.basename(self.backup_path)}_"):
                    backup_dirs.append(item)
            
            # Ordenar por fecha (asumiendo formato timestamp)
            backup_dirs.sort(reverse=True)
            
            # Eliminar backups antiguos
            for old_backup in backup_dirs[5:]:
                shutil.rmtree(old_backup)
                print(f"🗑️ Backup antiguo eliminado: {old_backup}")
                
        except Exception as e:
            print(f"⚠️ Error limpiando backups: {e}")
    
    def retrain_models(self):
        """Re-entrena todos los modelos con datos actualizados"""
        try:
            print("🔄 Iniciando re-entrenamiento de modelos...")
            
            # Backup de modelos actuales
            backup_dir = self.backup_current_models()
            
            # Importar y ejecutar el entrenamiento
            import subprocess
            import sys
            
            # Ejecutar el script de entrenamiento original
            result = subprocess.run([sys.executable, "train_model.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Re-entrenamiento completado exitosamente")
                
                # Guardar información del entrenamiento
                self.save_training_info()
                
                # Evaluar nuevo modelo
                new_performance = self.evaluate_current_model()
                print(f"📊 Rendimiento del nuevo modelo: {new_performance:.3f}")
                
                return True
            else:
                print(f"❌ Error en re-entrenamiento: {result.stderr}")
                # Restaurar backup si falló
                if backup_dir:
                    self.restore_backup(backup_dir)
                return False
                
        except Exception as e:
            print(f"❌ Error durante re-entrenamiento: {e}")
            return False
    
    def save_training_info(self):
        """Guarda información del entrenamiento actual"""
        try:
            df = pd.read_excel(self.data_file)
            
            training_info = {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(df),
                'training_method': 'automatic_retrain',
                'data_file': self.data_file,
                'model_path': self.model_path
            }
            
            with open(f"{self.model_path}/last_training_info.json", 'w') as f:
                json.dump(training_info, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error guardando info de entrenamiento: {e}")
    
    def restore_backup(self, backup_dir):
        """Restaura modelos desde backup"""
        try:
            if os.path.exists(backup_dir):
                # Eliminar directorio actual
                if os.path.exists(self.model_path):
                    shutil.rmtree(self.model_path)
                
                # Restaurar desde backup
                shutil.copytree(backup_dir, self.model_path)
                print(f"🔄 Modelos restaurados desde backup: {backup_dir}")
                return True
        except Exception as e:
            print(f"❌ Error restaurando backup: {e}")
            return False
    
    def run_auto_retrain_check(self):
        """Ejecuta verificación automática y re-entrenamiento si es necesario"""
        print("🤖 SISTEMA DE RE-ENTRENAMIENTO AUTOMÁTICO")
        print("="*50)
        
        # Verificar si es necesario re-entrenar
        needs_retrain, reasons = self.check_retrain_needed()
        
        if needs_retrain:
            print("🔍 Se detectó la necesidad de re-entrenamiento:")
            for reason in reasons:
                print(f"   • {reason}")
            
            # Confirmar re-entrenamiento
            print(f"\n¿Proceder con el re-entrenamiento? (s/n): ", end="")
            response = input().strip().lower()
            
            if response in ['s', 'si', 'sí', 'y', 'yes']:
                success = self.retrain_models()
                if success:
                    print("\n🎉 Re-entrenamiento completado exitosamente!")
                    return True
                else:
                    print("\n❌ Re-entrenamiento falló")
                    return False
            else:
                print("⏭️ Re-entrenamiento cancelado por el usuario")
                return False
        else:
            print("✅ No es necesario re-entrenar en este momento")
            print("   Los modelos están actualizados y funcionando correctamente")
            return True
    
    def setup_automatic_schedule(self):
        """Configura re-entrenamiento automático programado"""
        schedule_script = '''#!/usr/bin/env python3
import os
import sys
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_retrain_system import AutoRetrainSystem

def main():
    print(f"[{datetime.now()}] Ejecutando verificación automática...")
    
    retrain_system = AutoRetrainSystem()
    
    # Ejecutar verificación silenciosa
    needs_retrain, reasons = retrain_system.check_retrain_needed()
    
    if needs_retrain:
        print(f"[{datetime.now()}] Re-entrenamiento necesario. Ejecutando...")
        success = retrain_system.retrain_models()
        
        if success:
            print(f"[{datetime.now()}] Re-entrenamiento completado exitosamente")
        else:
            print(f"[{datetime.now()}] Error en re-entrenamiento")
    else:
        print(f"[{datetime.now()}] No es necesario re-entrenar")

if __name__ == "__main__":
    main()
'''
        
        # Guardar script de programación
        with open("scheduled_retrain.py", "w") as f:
            f.write(schedule_script)
        
        print("📅 Script de re-entrenamiento programado creado: scheduled_retrain.py")
        print("\n💡 Para programar ejecución automática:")
        print("   Linux/Mac: Agregar a crontab:")
        print("   0 2 * * * /usr/bin/python3 /path/to/scheduled_retrain.py")
        print("\n   Windows: Usar Programador de Tareas:")
        print("   python scheduled_retrain.py")

def main():
    """Función principal del sistema de re-entrenamiento"""
    retrain_system = AutoRetrainSystem()
    
    print("Selecciona una opción:")
    print("1. Verificar necesidad de re-entrenamiento")
    print("2. Forzar re-entrenamiento")
    print("3. Configurar re-entrenamiento automático")
    print("4. Evaluar modelo actual")
    print("5. Salir")
    
    while True:
        choice = input("\nOpción (1-5): ").strip()
        
        if choice == "1":
            retrain_system.run_auto_retrain_check()
            break
        elif choice == "2":
            print("🔄 Forzando re-entrenamiento...")
            success = retrain_system.retrain_models()
            if success:
                print("✅ Re-entrenamiento forzado completado")
            break
        elif choice == "3":
            retrain_system.setup_automatic_schedule()
            break
        elif choice == "4":
            if retrain_system.can_evaluate_current_model():
                performance = retrain_system.evaluate_current_model()
                print(f"📊 Rendimiento actual del modelo: {performance:.3f}")
            else:
                print("❌ No se puede evaluar el modelo actual")
            break
        elif choice == "5":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida. Por favor selecciona 1-5.")

if __name__ == "__main__":
    main()