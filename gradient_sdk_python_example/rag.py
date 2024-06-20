from dotenv import load_dotenv
from gradientai import (
    FileChunker,
    Gradient,
    MeaningBasedChunker,
    NormalChunker,
    RAGChunker,
    RAGCollection,
    SentenceWithContextChunker,
)

load_dotenv()

FILE_CHUNKER = FileChunker()

MEANING_BASED_CHUNKER_CONFIG = MeaningBasedChunker(
    overlap=20,
    sentence_group_length=1,
    similiarity_percent_threshold=95,
    size=1024,
)

NORMAL_CHUNKER_CONFIG = NormalChunker(
    overlap=20,
    size=1024,
)

SENTECE_WITH_CONTEXT_CHUNKER_CONFIG = SentenceWithContextChunker(
    context_sentences=3,
    overlap=20,
    size=1024,
)


def build_rag_collection(
    gradient: Gradient, chunker: RAGChunker
) -> RAGCollection:
    rag_collection = gradient.create_rag_collection(
        filepaths=[
            "resources/Lorem_Ipsum.pdf",
        ],
        name=f"RAG Collection with {chunker.chunker_type}",
        chunker=chunker,
        slug="bge-large",
    )
    print(f"Created RAG collection with id: {rag_collection.id_}")
    print(f"Chunker: {rag_collection.chunker}")

    rag_collection.add_files(filepaths=["resources/Life_Kit.mp3"])
    print(f"RAG collection files: {rag_collection.files}")

    return rag_collection


def main() -> None:
    gradient = Gradient()

    file_chunker_rag_collection = build_rag_collection(gradient, FILE_CHUNKER)

    meaning_based_chunker_rag_collection = build_rag_collection(
        gradient, MEANING_BASED_CHUNKER_CONFIG
    )

    normal_chunker_rag_collection = build_rag_collection(
        gradient, NORMAL_CHUNKER_CONFIG
    )

    sentence_with_context_chunker_rag_collection = build_rag_collection(
        gradient, SENTECE_WITH_CONTEXT_CHUNKER_CONFIG
    )

    file_chunker_rag_collection.delete()
    meaning_based_chunker_rag_collection.delete()
    normal_chunker_rag_collection.delete()
    sentence_with_context_chunker_rag_collection.delete()

    gradient.close()


if __name__ == "__main__":
    main()
