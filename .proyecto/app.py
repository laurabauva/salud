from flask import Flask, request, jsonify 
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
app = Flask(__name__)

# Configuración de la conexión a la base de datos
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT")
}

# Ruta para entrenar el modelo
@app.route('/train', methods=['POST'])
def entreno_modelo():
    
    try:
        # Conexión a la base de datos
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Cargar los datos de la tabla
        query = "SELECT * FROM diabetes"
        cursor.execute(query)
        data = cursor.fetchall()

        # Obtener nombres de las columnas
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        print("Columnas del DataFrame:", df.columns)
        print(df.head())



        # Separar características (X) y etiqueta (y)
        X = df.drop(columns=['Resultado'])
        y = df['Resultado']

        # División de entrenamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar modelo
        model = LogisticRegression(max_iter=500)
        model.fit(x_train, y_train)

        # Asegurarse de que la carpeta 'model' exista
        os.makedirs('model', exist_ok=True)

        # Guardar el modelo entrenado
        model_path = os.path.join('model', 'modelo_diabetes.pkl')
        joblib.dump(model, model_path)

        # Evaluar el modelo
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        return jsonify({'status': 'Modelo entrenado', 'accuracy': accuracy})
    
    except Exception as e:
        return jsonify({'status': 'Error', 'message': str(e)})

# Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cargar modelo entrenado
        model_path = os.path.join('model', 'modelo_diabetes.pkl')
        model = joblib.load(model_path)

        # Obtener los datos en el orden correcto
        columnas = [
            "Embarazos", "Glucosa", "PresionSanguinea", "PliegueCutaneo",
            "Insulina", "IMC", "AntecedentesDiabetes", "Edad"
        ]

        data = [float(request.form[col]) for col in columnas]

        # Hacer predicción
        prediction = model.predict([data])
        result = 'Diabética' if prediction[0] == 1 else 'No Diabética'

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'status': 'Error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
