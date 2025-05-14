from flask import Flask, request, jsonify 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
import joblib 
import mysql.connector 
import os 
from dotenv import load_dotenv 

#cargar las varibales del .env
load_dotenv()
app = Flask(__name__)


#se hace la configuracion con el dataset
db_config ={
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT")
}

#ruta de entrenamiento del modelo
@app.route ('/train', methods = ['POST'])
def entreno_modelo():
    try:

        #se hace la conexion con el dataset
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        #se carga los datos
        query = "SELECT * FROM diabetes"
        cursor.execute(query)
        data = cursor.fetchall()

        #se traen los nombres de las columnas
        columns =[desc [0] for desc in cursor.description]
        
        df = pd.DataFrame(data, columns=columns)

        #separacion de etiquetas
        print(df.columns)

        X = df.drop(columns=['Resultado'])
        y = df['Resulltado']

        #división de entrenamiento y la prueba
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

        #se entrena el modelo
        model = LogisticRegression(max_iter= 500)
        model.fit(x_train, y_train)

        #se guarda el modelo ya entrenado
        model_path = os.path.join('model', 'modelo_diabetes.pkl')
        joblib.dump(model, model_path)

        #se evalua el modelo
        y_pred = model .predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        return jsonify ({'status': 'modelo entrenado', 'accuracy': accuracy})
    
    except Exception as e:
        return jsonify ({'status': 'Error', 'message': str(e)})
