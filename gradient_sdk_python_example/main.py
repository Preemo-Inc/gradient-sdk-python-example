import os

from gradient_sdk_python_example.complete import complete_model_adapter
from gradient_sdk_python_example.fine_tune import fine_tune_model_adapter
from gradientai.api_client import ApiClient
from gradientai.api.models_api import ModelsApi
from gradientai.configuration import Configuration
from gradientai.models.list_models_success_models_inner import (
    ListModelsSuccessModelsInner,
)
from dotenv import load_dotenv
from gradient_sdk_python_example.create import create_model_adapter

from gradient_sdk_python_example.list import list_models

load_dotenv()


with ApiClient(
    Configuration(access_token=os.getenv("GRADIENT_ACCESS_TOKEN"))
) as api_client:
    api_instance = ModelsApi(api_client)
    workspace_id = os.getenv("GRADIENT_WORKSPACE_ID")

    models = list_models(
        api_instance=api_instance, only_base=True, workspace_id=workspace_id
    ).models

    base_model: ListModelsSuccessModelsInner = models[0].actual_instance

    test_model_adapter = create_model_adapter(
        api_instance=api_instance,
        base_model_id=base_model.id,
        name="testModel123",
        workspace_id=workspace_id,
    )

    fine_tune_model_adapter(
        api_instance=api_instance,
        model_adapter_id=test_model_adapter.id,
        samples=[{"inputs": "some prompt"}],
        workspace_id=workspace_id,
    )

    complete_model_adapter(
        api_instance=api_instance,
        model_adapter_id=test_model_adapter.id,
        query="some prompt",
        workspace_id=workspace_id,
    )
