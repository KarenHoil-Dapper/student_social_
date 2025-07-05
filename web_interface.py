from flask import Flask, render_template, request, session
import subprocess
import os
import sys
from updated_survey_system import SocialMediaHealthPredictor
from main_system import SocialMediaHealthSystem
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'secret_karen_2025'  # Necesario para sesiones
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

system = SocialMediaHealthSystem()

@app.route('/')
def entrenamiento():
    return render_template('entrenamiento.html', result=None, file_loaded=session.get('excel_file'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'excel_file' not in request.files:
        return render_template('entrenamiento.html', result="❌ No se envió ningún archivo.")
    
    file = request.files['excel_file']
    
    if file.filename == '':
        return render_template('entrenamiento.html', result="❌ Nombre de archivo vacío.")
    
    if file and file.filename.endswith(('.xlsx', '.xls')):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Guardar ruta en sesión para usarla en todo el sistema
        session['excel_file'] = file_path
        system.data_file = file_path  # También actualizar el sistema principal
        
        return render_template('entrenamiento.html', result=f"✅ Archivo cargado correctamente: {filename}", file_loaded=file_path)
    
    return render_template('entrenamiento.html', result="❌ Solo se permiten archivos Excel (.xlsx, .xls)")

@app.route('/setup')
def setup():
    system.setup_environment()
    return render_template('entrenamiento.html', result="✅ Entorno configurado.", file_loaded=session.get('excel_file'))

@app.route('/train')
def train():
    if not system.check_data_file():
        return render_template('entrenamiento.html', result="❌ Archivo Excel no cargado.", file_loaded=None)
    
    try:
        result = subprocess.run([sys.executable, "train_model.py", session['excel_file']], capture_output=True, text=True)
        if result.returncode == 0:
            return render_template('entrenamiento.html', result="✅ Modelos entrenados exitosamente:<br><pre>" + result.stdout + "</pre>", file_loaded=session['excel_file'])
        else:
            return render_template('entrenamiento.html', result="❌ Error al entrenar modelos:<br><pre>" + result.stderr + "</pre>", file_loaded=session['excel_file'])
    except Exception as e:
        return render_template('entrenamiento.html', result=f"❌ Error ejecutando entrenamiento: {e}", file_loaded=session.get('excel_file'))

@app.route('/analyze')
def analyze():
    if not session.get('excel_file'):
        return render_template('entrenamiento.html', result="❌ No se ha cargado un archivo Excel.", file_loaded=None)

    predictor = SocialMediaHealthPredictor()
    result = predictor.analyze_excel_data(session['excel_file'])

    if result.get("success"):
        message = (
            f"✅ Análisis completado.<br>"
            f"📊 Registros analizados: {result['analyzed_records']}<br>"
            f"📄 Archivo generado: {result['output_file']}"
        )
    else:
        message = f"❌ Error: {result.get('error', 'Desconocido')}"
    
    return render_template('entrenamiento.html', result=message, file_loaded=session['excel_file'])

@app.route('/status')
def status():
    if not session.get('excel_file'):
        return render_template('entrenamiento.html', result="❌ No se ha cargado un archivo Excel.", file_loaded=None)

    system.data_file = session['excel_file']

    from io import StringIO
    import sys
    import pandas as pd

    buffer = StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer

    try:
        system.show_system_status()
    finally:
        sys.stdout = sys_stdout

    output = buffer.getvalue()

    # Información adicional
    extra_info = ""
    try:
        df = pd.read_excel(system.data_file)

        # Normalizar columnas: quitar espacios, pasar a minúsculas
        df.columns = df.columns.str.strip().str.lower()

        # Mostrar columnas detectadas (útil para debug)
        print("🧪 Columnas detectadas:", list(df.columns))

        # Columnas esperadas en minúsculas
        columnas_relevantes = [
            'mental_health_score', 'addicted_score', 'screen_time',
            'sleep_quality', 'support_level', 'academic_impact'
        ]
        columnas_presentes = [col for col in columnas_relevantes if col in df.columns]

        extra_info += f"<br>✅ Columnas importantes: {len(columnas_presentes)}/{len(columnas_relevantes)}<br>"

        # Promedio de salud mental
        if 'mental_health_score' in df.columns:
            mental_vals = pd.to_numeric(df['mental_health_score'], errors='coerce')
            mental_avg = mental_vals.dropna().mean()
            extra_info += f"🧠 Salud mental promedio: {mental_avg:.2f}/10<br>"
        else:
            extra_info += "⚠️ Columna 'mental_health_score' no encontrada.<br>"

        # Promedio de adicción
        if 'addicted_score' in df.columns:
            addicted_vals = pd.to_numeric(df['addicted_score'], errors='coerce')
            addicted_avg = addicted_vals.dropna().mean()
            extra_info += f"📱 Adicción promedio: {addicted_avg:.2f}/10<br>"
        else:
            extra_info += "⚠️ Columna 'addicted_score' no encontrada.<br>"

    except Exception as e:
        extra_info += f"<br>⚠️ Error leyendo estadísticas: {e}"

    return render_template('entrenamiento.html', result=f"<pre>{output}</pre>{extra_info}", file_loaded=session['excel_file'])



if __name__ == '__main__':
    app.run(debug=True)
