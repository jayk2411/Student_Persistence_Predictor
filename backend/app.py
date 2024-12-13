from flask import Flask, request, jsonify
from flask_cors import CORS
from neural_network import NeuralNetwork
from database import Database

app = Flask(__name__)
CORS(app)

nn = NeuralNetwork()
db = Database()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    try:
        prediction = nn.predict(data)
        db.save_prediction(data, prediction)
        
        return jsonify({
            'probability': prediction,
            'message': 'Prediction saved successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/history', methods=['GET'])
def get_history():
    predictions = db.get_predictions()
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
