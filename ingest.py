#!/usr/bin/env python

import os
import pandas as pd
import weaviate
from weaviate.util import generate_uuid5

weaviate_api_key = os.environ['WEAVIATE_API_KEY']
weaviate_url = os.environ['WEAVIATE_URL']
openai_api_key = os.environ['AZURE_OPENAI_API_KEY']

auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)

# set up the client
CLIENT = weaviate.Client(
    url=weaviate_url,
    auth_client_secret=auth_config,
    additional_headers={
        "X-OpenAI-Api-Key": openai_api_key
    }
)

CLIENT.schema.delete_class("Work")

resource_name = "rdc-openai"
llm_deployment_id = "dc-gpt-35"

print(f"Resource name: {resource_name}")
print(f"Deployment ID: {llm_deployment_id}")

# create the schema
schema = {
    "classes": [
        {
            "class": "Work",
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": True,
                },
                "generative-openai": {
                    "resourceName": resource_name,
                    "deploymentId": llm_deployment_id,
                    "temperature": 0.0
                },
                "qna-openai": {
                    "resourceName": resource_name,
                    "deploymentId": llm_deployment_id,
                    "temperature": 0.0
                 }
            },
            "properties": [
                {"dataType": ["text"], "name": "title"},
                {"dataType": ["text"], "name": "accession_number", "tokenization": "field"},
                {"dataType": ["text[]"], "name": "alternate_title"},
                {"dataType": ["text"], "name": "api_model", "tokenization": "field"},
                {"dataType": ["text"], "name": "catalog_key", "tokenization": "field"},
                {"dataType": ["text"], "name": "collection", "tokenization": "field"},
                {"dataType": ["text[]"], "name": "contributor"},
                {"dataType": ["text"], "name": "create_date"},
                {"dataType": ["text[]"], "name": "creator"},
                {"dataType": ["text[]"], "name": "date_created"},
                {"dataType": ["text[]"], "name": "description"},
                {"dataType": ["text[]"], "name": "genre"},
                {"dataType": ["text"], "name": "identifier", "tokenization": "field"},
                {"dataType": ["text"], "name": "identifier_human_readable"},
                {"dataType": ["text"], "name": "keywords"},
                {"dataType": ["text[]"], "name": "language"},
                {"dataType": ["text"], "name": "library_unit"},
                {"dataType": ["text[]"], "name": "location"},
                {"dataType": ["text[]"], "name": "physical_description_material"},
                {"dataType": ["text[]"], "name": "physical_description_size"},
                {"dataType": ["text"], "name": "preservation_level"},
                {"dataType": ["boolean"], "name": "published"},
                {"dataType": ["text[]"], "name": "related_material"},
                {"dataType": ["text[]"], "name": "related_url"},
                {"dataType": ["text"], "name": "rights_holder"},
                {"dataType": ["text"], "name": "rights_statement"},
                {"dataType": ["text"], "name": "scope_and_contents"},
                {"dataType": ["text[]"], "name": "series"},
                {"dataType": ["text"], "name": "source", "tokenization": "field"},
                {"dataType": ["text"], "name": "status", "tokenization": "field"},
                {"dataType": ["text[]"], "name": "style_period"},
                {"dataType": ["text[]"], "name": "subject"},
                {"dataType": ["text"], "name": "table_of_contents"},
                {"dataType": ["text[]"], "name": "technique"},
                {"dataType": ["text"], "name": "visibility", "tokenization": "field"},
                {"dataType": ["text"], "name": "work_type", "tokenization": "field"},
            ]
        }
    ]
}

CLIENT.schema.create(schema)

data = pd.read_pickle("./data/nuldc_work_data.pkl")
print(f"Number of records: {len(data)}")


CLIENT.batch.configure(
    batch_size=20,
    callback=weaviate.util.check_batch_result,
    dynamic=True,
    timeout_retries=5,
    connection_error_retries=5,
)

with CLIENT.batch as batch:
    for i, d in enumerate(data.iloc):
        filtered = d.dropna().to_dict()
        uuid = filtered['identifier']
        print(f"{i} / 110160 ({round(i / 110160 * 100, 1)}%) -- {uuid}")
        batch.add_data_object(data_object=filtered, class_name="Work", uuid=uuid)