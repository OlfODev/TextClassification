from flask import Flask, request, jsonify, render_template
import ai  # Import your ai.py module

app = Flask(__name__)

# Create a route for text classification
@app.route('/api/classify', methods=['GET'])
def classify_text():
    text = request.args.get('text')

    if text is None:
        return jsonify({"error": "Text parameter is missing"}), 400

    labels = ai.classify_text(text)
    
    return jsonify({"predicted_labels": labels})

# Create a route for the index page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)