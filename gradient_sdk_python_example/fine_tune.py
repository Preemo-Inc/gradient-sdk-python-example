from typing import Literal
from gradientai.models.fine_tune_model_request_body import FineTuneModelRequestBody
from gradientai.models.fine_tune_model_success import FineTuneModelSuccess
from pprint import pprint
from gradientai.api.models_api import (
    ModelsApi,
)


def fine_tune_model_adapter(
    *,
    api_instance: ModelsApi,
    model_adapter_id: str,
    samples: list[dict[Literal["inputs"], str]],
    stdout=True,
    workspace_id: str
) -> FineTuneModelSuccess:
    api_response: FineTuneModelSuccess = api_instance.fine_tune_model(
        id=model_adapter_id,
        x_gradient_workspace_id=workspace_id,
        fine_tune_model_request_body=FineTuneModelRequestBody(samples=samples),
    )
    if stdout:
        print("The response of ModelsApi->fine_tune_model:\n")
        pprint(api_response)

    return api_response
