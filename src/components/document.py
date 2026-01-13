from typing import Dict, Any, Optional

class Document:
    """Class representing a single text chunk with its associated metadata."""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"Document(chunk_id={self.metadata.get('chunk_id')}, url={self.metadata.get('context_url')})"