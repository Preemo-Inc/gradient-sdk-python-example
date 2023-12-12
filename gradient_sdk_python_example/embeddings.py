from dotenv import load_dotenv

load_dotenv()
from gradientai import Gradient
import asyncio

def main() -> None:
    print("running sync embeddings example")
    gradient = Gradient()

    embeddings_model = gradient.get_embeddings_model(slug="bge-large")

    generate_embeddings_response = embeddings_model.embed(
        inputs=[
            {
                "input": "Multimodal brain MRI is the preferred method to evaluate for acute ischemic infarct and ideally should be obtained within 24 hours of symptom onset, and in most centers will follow a NCCT"
            },
            {
                "input": "CTA has a higher sensitivity and positive predictive value than magnetic resonance angiography (MRA) for detection of intracranial stenosis and occlusion and is recommended over time-of-flight (without contrast) MRA"
            },
            {
                "input": "Echocardiographic strain imaging has the advantage of detecting early cardiac involvement, even before thickened walls or symptoms are apparent"
            },
        ],
    )

    for embedding in generate_embeddings_response.embeddings:
        print(f"generated embedding: {embedding.embedding}")

    gradient.close()

async def main_async() -> None:
    print("running async embeddings example")
    gradient = Gradient()

    embeddings_model = gradient.get_embeddings_model(slug="bge-large")

    generate_embeddings_response = await embeddings_model.aembed(
        inputs=[
            {
                "input": "Multimodal brain MRI is the preferred method to evaluate for acute ischemic infarct and ideally should be obtained within 24 hours of symptom onset, and in most centers will follow a NCCT"
            },
            {
                "input": "CTA has a higher sensitivity and positive predictive value than magnetic resonance angiography (MRA) for detection of intracranial stenosis and occlusion and is recommended over time-of-flight (without contrast) MRA"
            },
            {
                "input": "Echocardiographic strain imaging has the advantage of detecting early cardiac involvement, even before thickened walls or symptoms are apparent"
            },
        ],
    )

    for embedding in generate_embeddings_response.embeddings:
        print(f"generated embedding: {embedding.embedding}")

    gradient.close()

if __name__ == "__main__":
    main()
    asyncio.run(main_async())