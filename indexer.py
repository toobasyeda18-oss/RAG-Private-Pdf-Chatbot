from qdrant_client.models import Distance, VectorParams, PointStruct

from pdf_utils import load_pages_and_chunk
from rag_core import embed_model, qdrant, COLLECTION


def index_pdf(pdf_path, file_name):
    chunks = load_pages_and_chunk(pdf_path)
    if len(chunks) == 0:
        return 0

    texts = []
    for chunk in chunks:
        texts.append(chunk["text"])

    vectors = embed_model.encode(texts)

    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    points = []
    for number in range(len(chunks)):
        points.append(
            PointStruct(
                id=number,
                vector=vectors[number].tolist(),
                payload={
                    "text": chunks[number]["text"],
                    "page": chunks[number]["page"],
                    "source": file_name,
                },
            )
        )

    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(chunks)