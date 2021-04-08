from flask import Flask
from utils import classify_age
import json
import logging

# configure logging
logging.basicConfig(filename='main.log',
                    level=logging.ERROR)

app = Flask(__name__)
app.config['debug'] = True


@app.route("/detect_age", methods=['GET'])
def detect_age():
    logging.info('detect_age method requested')

    age, conf = classify_age('Pawan.jpg')

    logging.info(f'age ------ {age}')
    logging.info(f'confidence ------ {conf}')

    value = {'estimated age': age,
             'prediction percentage': conf
             }

    return json.dumps(value)


if __name__ == "__main__":
    # to run locally
    host = '0.0.0.0'
    port = 5000
    app.run(debug=True)
