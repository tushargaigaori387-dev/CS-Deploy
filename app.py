from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ================= LOAD MODELS =================
try:
    kmeans = joblib.load('model/kmeans_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    encoder = joblib.load('model/encoder.pkl')
    pca = joblib.load('model/pca.pkl') if os.path.exists('model/pca.pkl') else None
    MODEL_LOADED = True
    print("✓ All models loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠ Error loading models: {e}")

# ================= CLUSTER INFORMATION =================
cluster_info = {
    0: {
        "name": "Budget Shoppers",
        "description": "Low income, minimal spending, occasional buyers",
        "strategy": "Discount campaigns, value products, loyalty rewards",
        "color": "#f44336"
    },
    1: {
        "name": "Regular Customers",
        "description": "Moderate income, consistent purchases, balanced engagement",
        "strategy": "Upsell opportunities, seasonal offers, email campaigns",
        "color": "#2196F3"
    },
    2: {
        "name": "Premium Buyers",
        "description": "High income, premium spenders, strong purchasing power",
        "strategy": "VIP treatment, exclusive products, personalized service",
        "color": "#4CAF50"
    },
    3: {
        "name": "Family Focused",
        "description": "Value-conscious, larger households, practical buyers",
        "strategy": "Family bundles, bulk discounts, kids products",
        "color": "#FF9800"
    }
}

# ================= TEAM MEMBERS =================
team_members = ["Prasanna", "Kusuma", "Gowtham", "Lavanya", "Tushar"]

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html', 
                         clusters=cluster_info, 
                         team=team_members,
                         model_loaded=MODEL_LOADED)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check model files in /model folder.'
        })
    
    try:
        # Collect input data
        data = {
            'Income': float(request.form['income']),
            'Recency': int(request.form['recency']),
            'Age': int(request.form['age']),
            'Total_Kids': int(request.form['kids']),
            'Total_Spend': float(request.form['spend']),
            'Total_Purchases': int(request.form['web_visits']),
            'NumWebVisitsMonth': int(request.form['web_visits']),
            'Education': request.form['education'],
            'Marital_Status': request.form['marital']
        }
        
        # Validate inputs
        if data['Income'] < 0 or data['Age'] < 18 or data['Age'] > 100:
            return jsonify({
                'success': False,
                'error': 'Invalid input values. Please check your data.'
            })
        
        # Create DataFrame
        df_input = pd.DataFrame([data])
        
        # Transform data
        encoded = encoder.transform(df_input)
        scaled = scaler.transform(encoded)
        final_input = pca.transform(scaled) if pca else scaled
        
        # Predict cluster
        cluster = int(kmeans.predict(final_input)[0])
        
        return jsonify({
            'success': True,
            'cluster': cluster,
            'name': cluster_info[cluster]['name'],
            'description': cluster_info[cluster]['description'],
            'strategy': cluster_info[cluster]['strategy'],
            'color': cluster_info[cluster]['color']
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid input format. Please enter valid numbers.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("CUSTOMER SEGMENTATION SYSTEM")
    print("="*60)
    print(f"Model Status: {'✓ Loaded' if MODEL_LOADED else '✗ Not Loaded'}")
    print("Team: " + ", ".join(team_members))
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)