from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
import joblib 
import mysql.connector 
import os 
from dotenv import load_dotenv 

# Cargar variables del archivo .env
load_dotenv()
app = Flask(__name__, template_folder='templates')

# Configuración de la base de datos
db_config = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "dataset_diabetes"),
    "port": int(os.getenv("DB_PORT", 3306))
}

# ... (mantén todas las funciones anteriores igual hasta las rutas)

# Rutas corregidas
@app.route('/')
def inicio():
    return render_template('etapa1.html')

@app.route('/etapa1')
def etapa1():
    return render_template('etapa1.html')

@app.route('/etapa2')
def etapa2():
    return render_template('etapa2.html')

@app.route('/prediccion')
def prediccion():
    return render_template('base.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_path = os.path.join('model', 'modelo_diabetes.pkl')
        if not os.path.exists(model_path):
            return render_template('etapa1.html', resultado="Error: Modelo no encontrado. Entrene el modelo primero.")
            
        model = joblib.load(model_path)

        input_data = [[
            float(request.form.get("embarazos", 0)),
            float(request.form.get("glucosa", 0)),
            float(request.form.get("presion", 0)),
            float(request.form.get("grosor_piel", 0)),
            float(request.form.get("insulina", 0)),
            float(request.form.get("imc", 0)),
            float(request.form.get("dpf", 0)),
            float(request.form.get("edad", 0))
        ]]

        prediction = model.predict(input_data)[0]
        resultado = 'Diabético' if prediction == 1 else 'No Diabético'

        return render_template('etapa1.html', 
                            resultado=resultado,
                            probabilidad=float(model.predict_proba(input_data)[0][1]))

    except Exception as e:
        return render_template('etapa1.html', 
                            resultado=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)