import requests
import json
from chatbot.milvus import Milvus


class Huggingface:
    def __init__(self, model):
        self.apiUrl = "https://api-inference.huggingface.co/models/" + model
        self.headers = {
            "Authorization": "Bearer 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"}
        self.milvusObject = Milvus()

    def getResultOfLlm(self, text):
        try:
            contentString = ''
            userTextEmbedding = self.__createEmbedding(text)
            similarContent = self.milvusObject.getSimilarityMetric(
                userTextEmbedding)

            dataList = json.loads(similarContent)

            for data in dataList:
                contentString = contentString + \
                    ', ' + data['entity']['content']

            prompt = f"<s>[INST]ROLE: You are a helpful assistant CONTEXT: ```{contentString}``` I want to asnwer my user's question \n\n QUESTION:{text} [/INST]"

            payload = {"inputs": prompt}
            response = requests.post(
                self.apiUrl, headers=self.headers, json=payload)
            return response.json()
        except Exception as e:
            return e

    def __createEmbedding(self, text):
        try:
            embedding = self.milvusObject.generateEmbedding(text)
            return embedding
        except Exception as e:
            return e
