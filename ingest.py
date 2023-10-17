#!/usr/bin/env python

import os
import pandas as pd
import weaviate
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()

weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
weaviate_url = os.environ["WEAVIATE_URL"]

auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)

# set up the client
CLIENT = weaviate.Client(url=weaviate_url, auth_client_secret=auth_config)

# Set your class name here
# class_name = "DCWork"
# CLIENT.schema.delete_class(class_name)

resource_name = "rdc-openai"
llm_deployment_id = "dc-gpt-35"

properties = [
    {"dataType": ["text"], "name": "title"},
    {"dataType": ["text"], "name": "accession_number", "tokenization": "field"},
    {"dataType": ["text"], "name": "alternate_title"},
    {"dataType": ["text"], "name": "api_model", "tokenization": "field"},
    {"dataType": ["text"], "name": "collection", "tokenization": "field"},
    {"dataType": ["text"], "name": "contributor"},
    {"dataType": ["text"], "name": "create_date"},
    {"dataType": ["text"], "name": "creator"},
    {"dataType": ["text"], "name": "date_created"},
    {"dataType": ["text"], "name": "description"},
    {"dataType": ["text"], "name": "genre"},
    {"dataType": ["text"], "name": "identifier_descriptive"},
    {"dataType": ["text"], "name": "keywords"},
    {"dataType": ["text"], "name": "language"},
    {"dataType": ["text"], "name": "library_unit"},
    {"dataType": ["text"], "name": "location"},
    {"dataType": ["text"], "name": "physical_description_material"},
    {"dataType": ["text"], "name": "physical_description_size"},
    {"dataType": ["text"], "name": "preservation_level"},
    {"dataType": ["boolean"], "name": "published"},
    {"dataType": ["text"], "name": "related_material"},
    {"dataType": ["text"], "name": "related_url"},
    {"dataType": ["text"], "name": "rights_holder"},
    {"dataType": ["text"], "name": "rights_statement"},
    {"dataType": ["text"], "name": "scope_and_contents"},
    {"dataType": ["text"], "name": "series"},
    {"dataType": ["text"], "name": "source", "tokenization": "field"},
    {"dataType": ["text"], "name": "source_descriptive"},
    {"dataType": ["text"], "name": "status", "tokenization": "field"},
    {"dataType": ["text"], "name": "style_period"},
    {"dataType": ["text"], "name": "subject"},
    {"dataType": ["text"], "name": "table_of_contents"},
    {"dataType": ["text"], "name": "technique"},
    {"dataType": ["text"], "name": "visibility", "tokenization": "field"},
    {"dataType": ["text"], "name": "work_type", "tokenization": "field"},
]

# create the schema
schema = {
    "classes": [
        {
            "class": "DCWork",
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": True,
                },
                "generative-openai": {
                    "resourceName": resource_name,
                    "deploymentId": llm_deployment_id,
                    "temperature": 0.0,
                },
                "qna-openai": {
                    "resourceName": resource_name,
                    "deploymentId": llm_deployment_id,
                    "temperature": 0.0,
                },
            },
            "properties": properties,
        }
    ]
}

# property_names = [prop["name"] for prop in properties]
# print(property_names)

CLIENT.schema.create(schema)

rename_mapping = {
    "source": "source_descriptive",
    "id": "source",
    "date_created": "date_created_edtf",
    "identifier": "identifier_descriptive",
}
data = pd.read_pickle("./data/merged.pkl")
data.rename(columns=rename_mapping, inplace=True)
total = len(data)
print(f"Number of records: {total}")

CLIENT.batch.configure(
    batch_size=25,
    callback=weaviate.util.check_batch_result,
    dynamic=True,
    timeout_retries=5,
    connection_error_retries=5,
    num_workers=2,
)

valid_fields = [
    "title",
    "accession_number",
    "alternate_title",
    "api_model",
    "collection",
    "contributor",
    "create_date",
    "creator",
    "date_created",
    "description",
    "genre",
    "identifier_descriptive",
    "keywords",
    "language",
    "library_unit",
    "location",
    "physical_description_material",
    "physical_description_size",
    "preservation_level",
    "published",
    "related_material",
    "related_url",
    "rights_holder",
    "rights_statement",
    "scope_and_contents",
    "series",
    "source",
    "source_descriptive",
    "status",
    "style_period",
    "subject",
    "table_of_contents",
    "technique",
    "visibility",
    "work_type",
]

pbar = tqdm(total=total, initial=1, desc="Vectorizing data")

with CLIENT.batch as batch:
    for i, d in enumerate(data.iloc):
        filtered = d.dropna().to_dict()
        filtered = {k: v for k, v in filtered.items() if k in valid_fields}
        uuid = filtered["source"]
        batch.add_data_object(data_object=filtered, class_name=class_name, uuid=uuid)
        pbar.update(1)

pbar.close()
