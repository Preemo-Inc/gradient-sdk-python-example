import os
from dotenv import load_dotenv
from gradientai import Gradient

load_dotenv()


def main():
    gradient = Gradient()

    rag_collection = gradient.create_rag_collection(
        filepaths=[
            "resources/Lorem_Ipsum.pdf",
        ],
        name="My RAG collection",
        slug="bge-large",
    )
    print(f"Created RAG collection with id: {rag_collection.id_}")

    rag_collection.add_files(filepaths=["resources/Life_Kit.mp3"])
    print(f"Added file to RAG Collection with id: {rag_collection.id_}")

    gradient.close()


if __name__ == "__main__":
    main()
