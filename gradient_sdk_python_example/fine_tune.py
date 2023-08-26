from typing import Literal
from gradientai.models.train_model_request_body import TrainModelRequestBody
from gradientai.models.train_model_success import TrainModelSuccess
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
) -> TrainModelSuccess:
    api_response: TrainModelSuccess = api_instance.train_model(
        id=model_adapter_id,
        x_gradient_workspace_id=workspace_id,
        train_model_request_body=TrainModelRequestBody(samples=samples),
    )
    if stdout:
        print("The response of ModelsApi->fine_tune_model:\n")
        pprint(api_response)

    return api_response
