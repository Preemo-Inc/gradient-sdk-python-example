from gradientai.models.create_model_request_body import (
    CreateModelRequestBody,
)
from gradientai.models.create_model_request_body_model import (
    CreateModelRequestBodyModel,
)
from gradientai.models.create_model_success import CreateModelSuccess
from pprint import pprint
from gradientai.api.models_api import (
    ModelsApi,
)


def create_model_adapter(
    *,
    api_instance: ModelsApi,
    base_model_id: str,
    name: str,
    stdout=True,
    workspace_id: str,
) -> CreateModelSuccess:
    api_response = api_instance.create_model(
        create_model_request_body=CreateModelRequestBody(
            model=CreateModelRequestBodyModel(name=name, baseModelId=base_model_id)
        ),
        x_gradient_workspace_id=workspace_id,
    )

    if stdout:
        print("The response of ModelsApi->create_model:\n")
        pprint(api_response)

    return api_response
