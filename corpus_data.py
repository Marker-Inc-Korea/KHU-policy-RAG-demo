import os
from datetime import datetime

import fitz  # PyMuPDF
import pandas as pd
from chunkipy import TextChunker
import uuid


def main():
    text_df = extract_text_from_pdf("../data/source_file")
    simple_chunker(text_df)


def extract_text_from_pdf(folder_path: str):
    texts = []
    for file_name in os.listdir(folder_path):
        pdf_path = os.path.join(folder_path, file_name)
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        texts.append(text)
    text_df = pd.DataFrame({'contents': texts})
    return text_df


def simple_chunker(corpus_data: pd.DataFrame):
    chunked_data = []
    text_chunker = TextChunker(512, tokens=True, overlap_percent=0.3)
    for _, data_row in corpus_data.iterrows():
        chunks = text_chunker.chunk(data_row['contents'])
        for chunk in chunks:
            chunked_data.append({
                'doc_id': str(uuid.uuid4()),
                'contents': chunk,
            })

    corpus_df = pd.DataFrame(chunked_data)
    metadata_dict = {'last_modified_datetime': datetime.now()}
    corpus_df['metadata'] = [metadata_dict for _ in range(len(corpus_df))]
    corpus_df.to_parquet("corpus.parquet", index=False)


if __name__ == "__main__":
    main()
