from gradientai.models.complete_model_body_params import CompleteModelBodyParams
from gradientai.api.models_api import (
    ModelsApi,
    CompleteModelSuccess,
)


def complete_model_adapter(
    *,
    api_instance: ModelsApi,
    model_adapter_id: str,
    stdout=True,
    query: str,
    workspace_id: str,
) -> CompleteModelSuccess:
    api_response = api_instance.complete_model(
        id=model_adapter_id,
        x_gradient_workspace_id=workspace_id,
        complete_model_body_params=CompleteModelBodyParams(query=query),
    )
    if stdout:
        print("The response of ModelsApi->complete_model:\n")
        print(api_response)

    return api_response
