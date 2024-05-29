from dotenv import load_dotenv
from gradientai import Gradient, SentenceWindowNodeParser, SimpleNodeParser

load_dotenv()


def main() -> None:
    gradient = Gradient()

    sentence_window_node_parser_rag_collection = gradient.create_rag_collection(
        filepaths=[
            "resources/Lorem_Ipsum.pdf",
        ],
        name="RAG Collection with Sentence Window Node Parser",
        parser=SentenceWindowNodeParser(
            chunk_size=1024, chunk_overlap=20, window_size=3
        ),
        slug="bge-large",
    )
    print(
        f"Created RAG collection with id: {sentence_window_node_parser_rag_collection.id_}"
    )
    print(f"RAG collection: {sentence_window_node_parser_rag_collection}")

    sentence_window_node_parser_rag_collection.add_files(
        filepaths=["resources/Life_Kit.mp3"]
    )
    print(
        f"RAG collection files: {sentence_window_node_parser_rag_collection.files}"
    )

    simple_node_parser_rag_collection = gradient.create_rag_collection(
        filepaths=[
            "resources/Lorem_Ipsum.pdf",
        ],
        name="RAG Collection with Simple Node Parser",
        parser=SimpleNodeParser(chunk_size=1024, chunk_overlap=20),
        slug="bge-large",
    )
    print(
        f"Created RAG collection with id: {simple_node_parser_rag_collection.id_}"
    )
    print(f"RAG collection: {simple_node_parser_rag_collection}")

    simple_node_parser_rag_collection.add_files(
        filepaths=["resources/Life_Kit.mp3"]
    )
    print(f"RAG collection files: {simple_node_parser_rag_collection.files}")

    sentence_window_node_parser_rag_collection.delete()
    simple_node_parser_rag_collection.delete()

    gradient.close()


if __name__ == "__main__":
    main()
