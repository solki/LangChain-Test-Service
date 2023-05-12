import os
from flask import Flask
from flask_restful import Api
from dotenv import load_dotenv, find_dotenv
from router import api_routes
import openai

_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
api = Api(app)

api_routes(api)

if __name__ == '__main__':
    app.run(debug=True)
