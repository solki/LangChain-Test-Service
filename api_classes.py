from flask import request
from flask_restful import Resource
from functools import wraps
import faiss
import json
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import FAISS

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


def load_api_keys():
    with open('keys.json') as f:
        api_keys = json.load(f)
    return api_keys


def check_api_key(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        api_keys = load_api_keys()
        api_key = request.headers.get('api_key')
        if api_key in api_keys.values():
            return func(*args, **kwargs)
        else:
            return dict(error='Invalid API key'), 401

    return decorated_function


def get_prompt_by_task(task_name):
    # read the json file
    with open('prompts.json', 'r') as f:
        prompts_data = json.load(f)
    for task in prompts_data["tasks"]:
        if task["var"] == task_name:
            return task["sys_msg"], task["message"]
    return None, None


class GetSummaryOnShortText(Resource):
    @check_api_key
    def post(self):
        data = request.get_json()
        if not data or "message" not in data:
            return dict(error="Please provide a piece of message in the JSON data using the 'message' key"), 400
        user_input = data['message']
        chat_model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
        system_template = "You are a helpful assistant."
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = "Please summary the given text in detail in Chinese The text is delimited by triple " \
                         "backticks ```{text}``` "
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

        # get a chat completion from the formatted messages
        resp = chat_model(chat_prompt.format_prompt(text=user_input).to_messages())
        return dict(message=resp.content)


"""
request 
parameters: your_task
data: message
"""
class GetChatResponse(Resource):
    @check_api_key
    def post(self):
        data = request.get_json()
        if not data or 'message' not in data:
            return dict(error="Please provide a piece of message in the JSON data using the 'message' key"), 400
        user_input = data['message']
        chat_model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
        task_name = request.args.get('your_task')
        if not task_name:
            print("no task is found")
            return dict(error="Please provide a task using the 'your_task' parameter"), 400
        system_template, human_template = get_prompt_by_task(task_name)
        messages = []
        if system_template != "empty" and system_template:
            messages.append(SystemMessagePromptTemplate.from_template(system_template))
        if human_template:
            messages.append(HumanMessagePromptTemplate.from_template(human_template))
        else:
            return dict(error="Prompt is not found"), 400
        chat_prompt = ChatPromptTemplate.from_messages(messages)

        # get a chat completion from the formatted messages
        resp = chat_model(chat_prompt.format_prompt(text=user_input).to_messages())
        # return dict(results=json.loads(resp.content))
        print(resp.content)
        # return jsonify({'results': resp.content})
        return json.loads(resp.content)


class SummarizeLongDoc(Resource):
    @check_api_key
    def post(self):
        file = request.files.get('file')
        if not file:
            return dict(error='No file provided')
        content = file.read().decode('utf-8')
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=150)
        docs = text_splitter.create_documents([content])
        llm = OpenAI(model_name='text-davinci-003', temperature=0)
        # Get your chain ready to use
        chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)
        output = chain.run(docs)
        return dict(summary=output)


class ResponseFromDoc(Resource):
    @check_api_key
    def post(self):
        # Check if file and query are present in the request
        if 'file' not in request.files or 'query' not in request.form:
            return dict(message='File and query are required.'), 400
        query = request.form['query']
        file = request.files.get('file')
        if not file:
            return dict(error='No file provided')
        content = file.read().decode('utf-8')
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=3000, chunk_overlap=350)
        docs = text_splitter.split_documents([content])
        embeddings = OpenAIEmbeddings()
        llm = OpenAI(model_name='text-davinci-003', temperature=0)
        doc_search = FAISS.from_documents(docs, embeddings)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=doc_search.as_retriever())
        return qa.run(query)


class GetGPTResponse(Resource):
    @check_api_key
    def post(self):
        chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
        data = request.get_json()
        if not data or 'message' not in data:
            return dict(error="Please provide a piece of message in the JSON data using the 'message' key"), 400
        user_input = data['message']
        messages = [
            SystemMessage(content="You are a helpful customer service assistant."),
            HumanMessage(content=user_input)
        ]
        resp = chat(messages)
        return dict(message=resp.content)


class GetDocResponse(Resource):
    @check_api_key
    def post(self):
        with open("./data/faiss_store_openai.pkl", "rb") as f:
            vector_store = pickle.load(f)
        if not vector_store:
            return dict(error='No vector file provided')
        llm = OpenAI(temperature=0)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
        data = request.get_json()
        if not data or 'message' not in data:
            return dict(error="Please provide a piece of message in the JSON data using the 'message' key"), 400
        query = data['message']
        print(query)
        return dict(message=chain({"question": query}))
