import requests

class Huggingface:
    def __init__(self,model):
        self.apiUrl = "https://api-inference.huggingface.co/models/" + model
        self.headers = {"Authorization": "Bearer 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"}

    
    def getResult(self,payload):
        try:
            response = requests.post(self.apiUrl, headers=self.headers, json=payload)
            return response.json()
        except Exception as e:
            return e