from dotenv import load_dotenv
from gradientai import Gradient, SimpleNodeParser

load_dotenv()


def main():
    gradient = Gradient()

    rag_collection = gradient.create_rag_collection(
        filepaths=[
            "resources/Lorem_Ipsum.pdf",
        ],
        name="My RAG collection",
        parser=SimpleNodeParser(chunk_size=1024, chunk_overlap=20),
        slug="bge-large",
    )
    print(f"Created RAG collection with id: {rag_collection.id_}")
    print(f"RAG collection: {rag_collection}")

    rag_collection.add_files(filepaths=["resources/Life_Kit.mp3"])
    print(f"RAG collection files: {rag_collection.files}")

    rag_collection.delete()

    gradient.close()


if __name__ == "__main__":
    main()
