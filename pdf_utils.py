from pypdf import PdfReader


def load_and_chunk(pdf_path, chunk_size=800, overlap=150):
    reader = PdfReader(pdf_path)

    all_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            all_text = all_text + page_text + "\n"

    chunks = []
    start = 0
    while start < len(all_text):
        end = start + chunk_size
        chunks.append(all_text[start:end])
        if end >= len(all_text):
            break
        start = end - overlap

    return chunks

def load_pages_and_chunk(pdf_path, chunk_size=1000, overlap=200):
    reader = PdfReader(pdf_path)
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            continue

        text = " ".join(text.split())

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))

            if end < len(text):
                cut = text.rfind(". ", start + chunk_size // 2, end)
                if cut != -1:
                    end = cut + 1

            piece = text[start:end].strip()
            if len(piece) > 50:
                chunks.append({"text": piece, "page": page_number})

            if end >= len(text):
                break
            start = max(end - overlap, start + 1)

    return chunks