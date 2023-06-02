from api_classes import GetGPTResponse, GetDocResponse, SummarizeLongDoc, ResponseFromDoc, GetSummaryOnShortText, GetChatResponse
from prompt_handler import Prompt, Task


def api_routes(api):
    api.add_resource(GetGPTResponse, '/responses')
    api.add_resource(GetDocResponse, '/doc_responses')
    api.add_resource(GetSummaryOnShortText, '/summary-on-short-text')
    api.add_resource(GetChatResponse, '/chat')
    api.add_resource(ResponseFromDoc, '/document/answer')
    api.add_resource(SummarizeLongDoc, '/document/summary')
    api.add_resource(Prompt, '/prompt')
    api.add_resource(Task, '/task')
