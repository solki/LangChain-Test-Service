from flask import request
from flask_restful import Resource
from functools import wraps
import json


def load_api_keys():
    with open('keys.json') as f:
        api_keys = json.load(f)
    return api_keys


def check_api_key(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        api_keys = load_api_keys()
        api_key = request.headers.get('api_key')
        print(api_key)
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
            return task["message"]
    return None


def get_sys_by_task(task_name):
    # read the json file
    with open('prompts.json', 'r') as f:
        prompts_data = json.load(f)
    for task in prompts_data["tasks"]:
        if task["var"] == task_name:
            return task["sys_msg"]
    return None


def update_prompt_by_task(task_name, new_prompt):
    # read the json file
    with open('prompts.json', 'r') as f:
        prompts_data = json.load(f)
    for task in prompts_data["tasks"]:
        if task["var"] == task_name:
            task['message'] = new_prompt
            with open('prompts.json', 'w') as file:
                json.dump(prompts_data, file, indent=2)
            return task['message']
    return None


class Prompt(Resource):
    @check_api_key
    def get(self):
        task_name = request.args.get('your_task')
        if not task_name:
            print("no task is found")
            return dict(error="Please provide a task using 'your_task' parameter", message="Please choose a task"), 400
        prompt = get_prompt_by_task(task_name)
        if prompt is None:
            return dict(error="Prompt get failed", message="Failed")
        return dict(prompt=prompt, message="Successful")

    @check_api_key
    def post(self):
        task_name = request.args.get('your_task')
        if not task_name:
            print("no task is found")
            return dict(error="Please provide a task using the 'your_task' parameter"), 400
        data = request.get_json()
        if not data or 'message' not in data:
            return dict(error="Please provide a piece of message in the JSON data using the 'message' key"), 400
        new_prompt = data['message']
        if update_prompt_by_task(task_name, new_prompt) is None:
            return dict(error="update is not successful"), 400
        return dict(message="The prompt has been updated"), 200


class Task(Resource):
    @check_api_key
    def get(self):
        task_name = request.args.get('your_task')
        if not task_name:
            print("no task is found")
            return dict(error="Please provide a task using the 'your_task' parameter"), 400


class UpdatePromptByTask(Resource):
    @check_api_key
    def get(self):
        task_name = request.args.get('your_task')
        if not task_name:
            print("no task is found")
            return dict(error="Please provide a task using the 'your_task' parameter"), 400