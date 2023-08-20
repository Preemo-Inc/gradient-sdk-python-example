import gradientai
import os

from dotenv import load_dotenv
load_dotenv()

configuration = gradientai.Configuration(
    host=os.getenv("GRADIENT_API_URL"),
    access_token=os.getenv("GRADIENT_ACCESS_TOKEN")
)

models_api = gradientai.ModelsApi(gradientai.ApiClient(configuration))
model_list = models_api.list_models(
    only_base=False,
    x_preemo_workspace_id=os.getenv("GRADIENT_WORKSPACE_ID")
)

print(model_list.models[0].actual_instance)
