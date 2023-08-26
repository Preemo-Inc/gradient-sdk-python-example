import time
import os
import gradientai
from gradientai.models.create_model_request_body import CreateModelRequestBody
from gradientai.models.create_model_success import CreateModelSuccess
from gradientai.rest import ApiException
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

configuration = gradientai.Configuration(
    access_token=os.getenv("GRADIENT_ACCESS_TOKEN")
)

with gradientai.ApiClient(configuration) as api_client:
    api_instance = gradientai.ModelsApi(api_client)
    x_gradient_workspace_id = os.getenv("GRADIENT_WORKSPACE_ID")
    model = gradientai.CreateModelRequestBodyModel(
        name="My sample model!",
        baseModelId="99148c6d-c2a0-4fbe-a4a7-e7c05bdb8a09_base_ml_model")
    create_model_request_body = gradientai.CreateModelRequestBody(model=model)

    try:
        api_response = api_instance.create_model(x_gradient_workspace_id, create_model_request_body)
        print("The response of ModelsApi->create_model:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ModelsApi->create_model: %s\n" % e)

    only_base = False # bool |  (optional) (default to False)
    try:
        # List available models
        api_response = api_instance.list_models(x_gradient_workspace_id, only_base=only_base)
        print("The response of ModelsApi->list_models:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ModelsApi->list_models: %s\n" % e)

