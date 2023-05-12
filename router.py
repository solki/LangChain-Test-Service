from api_classes import GetGPTResponse, SummarizeLongDoc, GetSummaryOnShortText, GetChatResponse, HelloWorld


def api_routes(api):
    api.add_resource(GetGPTResponse, '/responses')
    api.add_resource(GetSummaryOnShortText, '/summary-on-short-text')
    api.add_resource(GetChatResponse, '/chat')
    api.add_resource(SummarizeLongDoc, '/document/summary')
    api.add_resource(HelloWorld, '/hello')
