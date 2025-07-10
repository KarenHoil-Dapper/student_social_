import pandas as pd
import numpy as np
from datetime import datetime
import os
import shutil

class DataSaver:
    def __init__(self, excel_file='Social_Bueno.xlsx'):
        self.excel_file = excel_file
        
    def get_next_student_id(self):
        """Obtiene el próximo Student_ID disponible"""
        try:
            df = pd.read_excel(self.excel_file)
            return int(df['Student_ID'].max()) + 1
        except:
            return 1
    
    def convert_form_data_to_row(self, user_data, predictions):
        """Convierte los datos del formulario a una fila del Excel"""
        
        # Obtener próximo ID
        student_id = self.get_next_student_id()
        
        # Crear fila base
        row_data = {
            'Student_ID': student_id,
            'Age': user_data['edad'],
            'Gender': 1 if user_data['genero'] == 'Hombre' else 0,
            'Avg_Daily_Usage_Hours': user_data['uso_diario'],
            'Affects_Academic_Performance': 1 if user_data['afecta_academico'] == 'Si' else 0,
            'Sleep_Hours_Per_Night': user_data['horas_sueno'],
            'Conflicts_Over_Social_Media': user_data['conflictos'],
            
            # Scores calculados por ML
            'Mental_Health_Score': np.mean(list(predictions['mental_health'].values())),
            'Addicted_Score': np.mean(list(predictions['addiction'].values()))
        }
        
        # Nivel académico (one-hot encoding)
        row_data['Academic_Level_Graduate'] = 1 if user_data['nivel_academico'] == 'Posgrado' else 0
        row_data['Academic_Level_High School'] = 1 if user_data['nivel_academico'] == 'Bachillerato' else 0
        row_data['Academic_Level_Undergraduate'] = 1 if user_data['nivel_academico'] == 'Licenciatura' else 0
        
        # Países (por defecto México)
        countries = [
            'Afghanistan', 'Albania', 'Andorra', 'Argentina', 'Armenia', 'Australia', 'Austria', 
            'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Bhutan', 
            'Bolivia', 'Bosnia', 'Brazil', 'Bulgaria', 'Canada', 'Chile', 'China', 'Colombia', 
            'Costa Rica', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Ecuador', 'Egypt', 
            'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Hong Kong', 
            'Hungary', 'Iceland', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Israel', 'Italy', 
            'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 
            'Latvia', 'Lebanon', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malaysia', 'Maldives', 
            'Malta', 'Mexico', 'Moldova', 'Monaco', 'Montenegro', 'Morocco', 'Nepal', 'Netherlands', 
            'New Zealand', 'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama', 
            'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 
            'San Marino', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 
            'Spain', 'Sri Lanka', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 
            'Trinidad', 'Turkey', 'UAE', 'UK', 'USA', 'Ukraine', 'Uruguay', 'Uzbekistan', 
            'Vatican City', 'Venezuela', 'Vietnam', 'Yemen'
        ]
        
        for country in countries:
            row_data[f'Country_{country}'] = 1 if country == 'Mexico' else 0
        
        # Plataformas más usadas (one-hot encoding)
        platforms = ['Facebook', 'Instagram', 'KakaoTalk', 'LINE', 'LinkedIn', 'Snapchat', 
                    'TikTok', 'Twitter', 'VKontakte', 'WeChat', 'WhatsApp', 'YouTube']
        
        for platform in platforms:
            row_data[f'Most_Used_Platform_{platform}'] = 1 if platform == user_data['plataforma_mas_usada'] else 0
        
        # Estado de relación (one-hot encoding)
        relationship_map = {
            'Soltero': 'Single',
            'En relación': 'In Relationship', 
            'Complicado': 'Complicated'
        }
        
        relationships = ['Complicated', 'In Relationship', 'Single']
        target_relationship = relationship_map.get(user_data['estado_relacion'], 'Single')
        
        for relationship in relationships:
            row_data[f'Relationship_Status_{relationship}'] = 1 if relationship == target_relationship else 0
        
        return row_data
    
    def save_user_data(self, user_data, predictions):
        """Guarda los datos del usuario en el Excel"""
        try:
            print("💾 Guardando datos del usuario...")
            
            # Cargar dataset actual
            df = pd.read_excel(self.excel_file)
            print(f"📊 Registros actuales: {len(df)}")
            
            # Convertir datos del formulario a fila
            new_row = self.convert_form_data_to_row(user_data, predictions)
            print(f"🆔 Nuevo Student_ID: {new_row['Student_ID']}")
            
            # Agregar nueva fila
            new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Guardar archivo actualizado
            new_df.to_excel(self.excel_file, index=False)
            
            print(f"✅ Datos guardados exitosamente!")
            print(f"📈 Total registros: {len(new_df)}")
            
            return {
                'student_id': new_row['Student_ID'],
                'mental_health_score': round(new_row['Mental_Health_Score'], 2),
                'addiction_score': round(new_row['Addicted_Score'], 2),
                'total_records': len(new_df)
            }
            
        except Exception as e:
            print(f"❌ Error guardando datos: {str(e)}")
            return None

# Función principal para usar desde Flask
def save_user_prediction(user_data, predictions):
    """Función principal para guardar datos desde Flask"""
    saver = DataSaver()
    
    try:
        result = saver.save_user_data(user_data, predictions)
        
        if result:
            return {
                'success': True,
                'user_info': result,
                'message': f"Datos guardados como Student_ID {result['student_id']}"
            }
        else:
            return {
                'success': False,
                'message': "Error guardando los datos"
            }
            
    except Exception as e:
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }