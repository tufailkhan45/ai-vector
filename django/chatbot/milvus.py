import json
from pymilvus import MilvusClient, DataType, utility, connections, Collection
from InstructorEmbedding import INSTRUCTOR
import pandas as pd
from django.conf import settings


class Milvus:
    def __init__(self):
        # settings.MILVUS_URI
        # tcp://milvus-standalone:19530
        # http://localhost:19530
        self.client = MilvusClient(uri='tcp://milvus-standalone:19530')
        self.collectionName = 'chatbot_collection'

    def __setupCollection(self):
        try:
            if self.collectionName not in self.client.list_collections():
                schema = self.client.create_schema(
                    auto_id=True,
                    enable_dynamic_field=True
                )

                schema.add_field(field_name="id",
                                 datatype=DataType.INT64,
                                 is_primary=True)

                schema.add_field(field_name="content",
                                 datatype=DataType.STRING,
                                 max_length=1500)

                schema.add_field(field_name="vector",
                                 datatype=DataType.FLOAT_VECTOR,
                                 dim=768)

                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    index_type="IVF_FLAT",
                    metric_type="L2",
                    params={"nlist": 1536}
                )

                self.client.create_collection(
                    collection_name=self.collectionName,
                    schema=schema,
                    index_params=index_params
                )
                print(
                    f"Collection '{self.collectionName}' created successfully")
            else:
                print(f"Collection '{self.collectionName}' already exists")
        except Exception as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def __getExternalData():
        client = MilvusClient(
            uri="http://13.234.239.110:19530"
        )

        response = client.query(
            collection_name="overalls_pk",
            filter='',
            limit=100,
            output_fields=["content", "embeddings"],
        )

        return response

    def __generateEmbeddings(self):
        try:
            insertArray = []
            rawData = self.__getExternalData()
            embeddingModel = INSTRUCTOR('hkunlp/instructor-large')
            for index, item in enumerate(rawData):
                embeddings = embeddingModel.encode([item['content']])
                insertArray.append(
                    {"content": item['content'], "vector": embeddings[0]}
                )

            res = self.client.insert(
                collection_name=self.collectionName,
                data=insertArray
            )
            print(res)
        except Exception as e:
            print(f"An error occurred: {e}")

    def __getTotalRecordsCount(self):
        try:
            response = self.client.query(
                collection_name=self.collectionName,
                filter='',
                limit=200,
                output_fields=["content", "vector"],
            )
            print(f"Collection has {len(response)} records")
        except Exception as e:
            print(f"An error occurred: {e}")

    def generateEmbedding(self, text):
        try:
            embeddingModel = INSTRUCTOR('hkunlp/instructor-large')
            embeddings = embeddingModel.encode([text])
            return embeddings
        except Exception as e:
            return e

    def getSimilarityMetric(self, userMetric):
        try:
            # return userMetric
            res = self.client.search(
                collection_name=self.collectionName,
                data=userMetric,
                limit=10,
                search_params={"metric_type": "L2",
                               "params": {}},
                output_fields=["content"],
            )

            # result = json.dumps(res[0], indent=4)
            result = json.dumps(res[0])
            return result
        except Exception as e:
            return e

    def runMilvusSetup(self):
        try:
            self.__setupCollection()
            self.__generateEmbeddings()
            self.__getTotalRecordsCount()
        except Exception as e:
            print(f"An error occurred: {e}")


# obj = Milvus()
# # obj.runMilvusSetup()
# embedding = obj.generateEmbedding("What is savyour")
# print(obj.getSimilarityMetric(embedding))
