from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
import joblib 
import mysql.connector 
import os 
from dotenv import load_dotenv 
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO



# Cargar variables del archivo .env
load_dotenv()
app = Flask(__name__, template_folder='templates')

app.config["DEBUG"] = True
app.config["ENV"] = "development"


# Configuración de la base de datos
db_config = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "dataset_diabetes"),
    "port": int(os.getenv("DB_PORT", 3306))
}

def create_risk_chart(patient_data, probabilidad):
    """Crea una gráfica de riesgo mejorada considerando múltiples factores"""
    try:
        plt.switch_backend('Agg')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Datos del paciente
        factores = ['Embarazos', 'Glucosa', 'Presión', 'IMC', 'Edad']
        valores = [
            patient_data['embarazos'],
            patient_data['glucosa'],
            patient_data['presion'],
            patient_data['imc'],
            patient_data['edad']
        ]
        
        # Rangos normales de referencia
        rangos_normales = {
            'Glucosa': (70, 140),  # mg/dL
            'Presión': (60, 120),   # mmHg (presión diastólica)
            'IMC': (18.5, 24.9),    # Índice de masa corporal
            'Edad': (20, 35)        # Edad ideal para embarazo
        }
        
        # Gráfico 1: Factores de riesgo comparados
        colors = ['#ff9999' if (factor in rangos_normales and 
                               (val > rangos_normales[factor][1] or 
                                val < rangos_normales[factor][0])) 
                 else '#66b3ff' for factor, val in zip(factores[1:], valores[1:])]
        
        ax1.bar(factores[1:], valores[1:], color=colors)
        ax1.set_title('Factores de Riesgo Clave')
        ax1.set_ylabel('Valores')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Añadir líneas de rango normal
        for factor in rangos_normales:
            idx = factores.index(factor)
            ax1.plot([idx-1.5, idx-0.5], [rangos_normales[factor][0]]*2, 'g--')
            ax1.plot([idx-1.5, idx-0.5], [rangos_normales[factor][1]]*2, 'g--')
        
        # Gráfico 2: Probabilidad de diabetes
        ax2.pie([probabilidad, 1-probabilidad], 
               labels=['Riesgo', 'No riesgo'], 
               colors=['#ff9999','#66b3ff'],
               autopct='%1.1f%%',
               startangle=90)
        ax2.set_title(f'Probabilidad de Diabetes: {probabilidad*100:.1f}%')
        
        plt.suptitle('Evaluación de Riesgo de Diabetes Gestacional', fontsize=16)
        plt.tight_layout()
        
        # Guardar la gráfica
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return image_base64
    except Exception as e:
        print(f"Error al generar gráfico: {str(e)}")
        return None


# Rutas para todas las etapas
@app.route('/')
def inicio():
    return render_template('etapa1.html')

@app.route('/etapa1')
def etapa1():
    return render_template('etapa1.html')

@app.route('/etapa2')
def etapa2():
    return render_template('etapa2.html')

@app.route('/etapa3')
def etapa3():
    return render_template('etapa3.html')

@app.route('/etapa4')
def etapa4():
    return render_template('etapa4.html')

@app.route('/prediccion')
def prediccion():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_path = os.path.join('model', 'modelo_diabetes.pkl')
        if not os.path.exists(model_path):
            return render_template('base.html', 
                                resultado="Error: Modelo no encontrado",
                                probabilidad=0,
                                chart_image=None)

        model = joblib.load(model_path)

        # Obtener datos del formulario
        data = {
            'embarazos': float(request.form.get("embarazos", 0)),
            'glucosa': float(request.form.get("glucosa", 0)),
            'presion': float(request.form.get("presion", 0)),
            'pliegue': float(request.form.get("pliegue", 0)),
            'insulina': float(request.form.get("insulina", 0)),
            'imc': float(request.form.get("imc", 0)),
            'dpf': float(request.form.get("dpf", 0)),
            'edad': float(request.form.get("edad", 0))
        }

        input_data = [[
            data['embarazos'], data['glucosa'], data['presion'],
            data['pliegue'], data['insulina'], data['imc'],
            data['dpf'], data['edad']
        ]]

        # Realizar predicción
        prediction = model.predict(input_data)[0]
        resultado = 'Riesgo Alto (Diabético)' if prediction == 1 else 'Riesgo Bajo (No Diabético)'
        probabilidad = float(model.predict_proba(input_data)[0][1])
        
        # Generar gráfica usando la función que ya tienes
        chart_image = create_risk_chart(data, probabilidad)

        return render_template('base.html',
                            resultado=resultado,
                            probabilidad=probabilidad,
                            chart_image=chart_image)

    except Exception as e:
        print(f"Error en /predict: {str(e)}")
        return render_template('base.html',
                            resultado=f"Error: {str(e)}",
                            probabilidad=0,
                            chart_image=None)

if __name__ == '__main__':
    app.run(debug=True)