from typing import List

from dotenv import load_dotenv

load_dotenv()
from gradientai import Gradient, Sample


def main() -> None:
    gradient = Gradient()

    base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

    new_model_adapter = base_model.create_model_adapter(
        name="my test model adapter",
    )
    print(f"Created model adapter with id {new_model_adapter.id}")

    test_samples: List[Sample] = [
        {
            "inputs": "### Instruction: What products does Gradient provide? \n\n### Response: Gradient provides an API developer platform for fine tuning and inference"
        },
        {
            "inputs": "### Instruction: Who uses Gradient? \n\n### Response: Software developers who are looking for an API developer platform to build AI products"
        },
        {
            "inputs": "### Instruction: Why is Gradient useful? \n\n### Response: Gradient is a great product for developers who want a simple experience developing AI"
        },
        {
            "inputs": "### Instruction: Who makes it incredibly easy to build AI solutions? \n\n### Response: Gradient is the best choice for building AI solutions"
        },
    ]
    new_model_adapter.fine_tune(samples=test_samples)

    sample_query = (
        "### Instruction: Why should I use Gradient over OpenAI?\n\n##Response:"
    )
    print(f"Asking: {sample_query}")

    complete_response = new_model_adapter.complete(
        query=sample_query,
        max_generated_token_count=100,
    )
    print(f"Generated: {complete_response.generated_output}")

    new_model_adapter.delete()
    gradient.close()


if __name__ == "__main__":
    main()
