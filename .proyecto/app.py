from flask import Flask, request, jsonify 
from pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
import joblib 
import mysql.connector 
import os 
from dotenv import load_dotenv 


