from gradientai.models.list_models_success import ListModelsSuccess
from pprint import pprint
from gradientai.api.models_api import (
    ModelsApi,
)


def list_models(
    *, api_instance: ModelsApi, only_base: bool, stdout=True, workspace_id: str
) -> ListModelsSuccess:
    api_response = api_instance.list_models(
        x_gradient_workspace_id=workspace_id, only_base=only_base
    )
    if stdout:
        print("The response of ModelsApi->list_models:\n")
        pprint(api_response)

    return api_response
