# Document ingestion

`POST /document` keeps the existing background worker, Cohere
`embed-multilingual-v3.0` embeddings, Supabase storage and callbacks. A processing
error deletes all stored fragments with the upload's `file_hash`, including any
batches already written.

## PDF extraction and OCR

Each page is extracted separately in reading order. Native text is used when it
passes heuristic checks: at least 100 non-whitespace characters, few invalid or
replacement characters, and a reasonable proportion of letters and numbers.
Fragmented single-character text and `(cid:...)` glyph mappings trigger OCR too.
A large page image with less than 500 characters of native text also triggers OCR,
to catch scanned bodies that expose only a native header.

Pages requiring OCR are rendered as RGB PNGs at up to 200 DPI, capped at 3000
pixels on their longest side. The worker sends only that page to the OpenAI
Responses API. The prompt requests a faithful transcription in the original
language, with paragraphs, lists and Markdown tables, and forbids summaries,
translation, invented text and following instructions printed in the document.
The transcription replaces the unreliable native text before chunking.

The API returns structured `status` and `text` fields. Blank pages and images with
no text can be skipped; unreadable pages, refusals, malformed responses and
incomplete output fail the upload. Transient transport errors, HTTP 408/409/429
and server errors are retried up to three attempts, with a 120-second timeout per
attempt. Requests use `store=false`. Password-protected PDFs are rejected.

Configuration is passed to the `data_distributor` service by Compose:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | required by Compose | Existing shared OpenAI key. Needed when a page requires OCR. |
| `DOCUMENT_OCR_MODEL` | `gpt-6-luna` | Vision model supporting Responses and structured outputs. |
| `DOCUMENT_OCR_MIN_TEXT_CHARS` | `100` | Minimum non-whitespace character count for native PDF text. |

These checks are heuristics. Syntactically plausible but incorrect PDF text can
still pass, and vision transcription can make mistakes. Validate OCR quality on
representative documents before relying on exact numbers or names.

## DOCX, text and chunking

DOCX extraction preserves the order of paragraphs and tables, including nested
tables, hyperlinks, headers and footers. Shared headers and footers are read once.
Explicit page breaks are hard chunk boundaries. Actual Word page numbers are not
inferred because they depend on a layout engine. Images inside DOCX files are not
sent through OCR.

TXT, Markdown, CSV and TSV are read as text. UTF-8/16/32 BOMs and MIME charset
declarations are supported; other encodings are detected with charset-normalizer.
Undeclared legacy encodings can be ambiguous. Form feeds create hard boundaries.
CSV and TSV retain their text representation; they are not parsed as spreadsheets.

Text is normalized without collapsing paragraph breaks. Chunks remain limited to
1000 characters with up to 150 characters of overlap, preferring paragraph, line
and sentence boundaries. A PDF chunk never crosses a page boundary. Stored
metadata retains existing fields and adds `document_id`, zero-based
`chunk_index`, `extraction_method` and the one-based PDF `page` number.

Embedding batches contain up to 90 chunks and are saved immediately after
checking vector count, consistent dimensions and finite values. Empty documents
are rejected. The website crawler and its processor are unchanged.

## Checks

From `data_distributor/`, run `venv/bin/python -m pytest -q`. Tests generate real
PDFs and DOCX files and mock HTTP/database services; they make no paid API calls.

API documentation checked on 2026-10-30:

- [Images and vision](https://developers.openai.com/api/docs/guides/images-vision)
- [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
