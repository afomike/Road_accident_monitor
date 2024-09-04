from flask import Flask, request, render_template, redirect, url_for, session, flash
import os
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# Set SECRET_KEY
app.config['SECRET_KEY'] = os.urandom(24).hex()

# Set up database
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created directory: {DATA_DIR}")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(DATA_DIR, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False



db = SQLAlchemy(app)

# User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load the trained model and encoders once
model = joblib.load('model/Accident_severity.joblib')
one_hot_encoder = joblib.load('model/one_hot_encoder.joblib')
label_encoders = joblib.load('model/label_encoders.joblib')
scaler = joblib.load('model/scaler.joblib')

# Define preprocessing function
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    numeric_columns = ['Number_of_vehicles_involved', 'Number_of_casualties']

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df[numeric_columns] = scaler.transform(df[numeric_columns])

    label_encode_features = [
        'Age_band_of_driver',
        'Driving_experience',
        'Service_year_of_vehicle',
        'Accident_severity'
    ]
    for column in label_encode_features:
        if column in df.columns:
            le = label_encoders[column]
            known_classes = set(le.classes_)
            df[column] = df[column].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)

    one_hot_features = [
        'Day_of_week', 'Sex_of_driver', 'Educational_level', 'Type_of_vehicle',
        'Owner_of_vehicle', 'Defect_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians',
        'Road_allignment', 'Types_of_Junction', 'Road_surface_type', 'Road_surface_conditions',
        'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement',
        'Cause_of_accident'
    ]

    for feature in one_hot_features:
        if feature not in df.columns or df[feature].isin(one_hot_encoder.categories_[one_hot_features.index(feature)]).sum() == 0:
            df[feature] = 'Unknown'
    
    if all(feature in df.columns for feature in one_hot_features):
        one_hot_encoded = one_hot_encoder.transform(df[one_hot_features])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_features))
        df = df.drop(columns=one_hot_features)
        df = pd.concat([df.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], axis=1)

    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='sha256')

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('form'))

        flash('Login failed! Please check your username and password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/form', methods=['POST', 'GET'])
def form():
    if 'user_id' not in session:
        flash('Please log in to access the prediction page.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            data = request.form.to_dict()
            required_fields = [
                'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
                'Driving_experience', 'Type_of_vehicle', 'Owner_of_vehicle',
                'Service_year_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians',
                'Road_allignment', 'Types_of_Junction', 'Road_surface_type',
                'Road_surface_conditions', 'Light_conditions', 'Weather_conditions',
                'Type_of_collision', 'Number_of_vehicles_involved', 'Number_of_casualties',
                'Vehicle_movement', 'Cause_of_accident'
            ]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return render_template('form.html', prediction=f"Missing required fields: {', '.join(missing_fields)}")

            preprocessed_data = preprocess_input(data)
            prediction = model.predict(preprocessed_data)
            prediction_result = prediction[0]

            if prediction_result == 2:
                prediction_result = "Slight Injury"
            elif prediction_result == 1:
                prediction_result = "Serious Injury"
            elif prediction_result == 0:
                prediction_result = "Fatal Injury"
            else:
                prediction_result = "Unknown injury type"

            return render_template('predict.html', prediction=f'From Observation The Accident can cause a: {prediction_result}')
        except Exception as e:
            return render_template('form.html', prediction=f"Error: {str(e)}")
    
    return render_template('form.html')
    
app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'user_id' not in session:
        flash('Please log in to access the prediction page.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
            return render_template('predict.html')

    return render_template('predict.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
