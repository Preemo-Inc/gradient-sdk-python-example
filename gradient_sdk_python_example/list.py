import gradientai
import os

from dotenv import load_dotenv
load_dotenv()

configuration = gradientai.Configuration(
    access_token=os.getenv("GRADIENT_ACCESS_TOKEN")
)

with gradientai.ApiClient(configuration) as api_client:
    models_api = gradientai.ModelsApi(api_client)
    model_list = models_api.list_models(
        x_gradient_workspace_id=os.getenv("GRADIENT_WORKSPACE_ID")
    )

print(model_list.models[0].actual_instance)
