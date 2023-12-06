from dotenv import load_dotenv

load_dotenv()
from gradientai import Gradient


def main() -> None:
    gradient = Gradient()
    base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

    query = "What is your favorite color? "
    print(f"Asking: {query}")

    choices = ["red", "green", "blue"]
    print(f"Allowed responses: {choices}")

    complete_response = base_model.complete(
        guidance={
            "type": "choice",
            "value": choices,
        },
        query=query,
    )
    print(f"Generated: {complete_response.generated_output}")

    gradient.close()


if __name__ == "__main__":
    main()
