"""
Metadata Document Loader for Disk-Based Processing

This module provides classes for loading document metadata from cache
and accessing full document content on-demand to minimize memory usage.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from langchain_core.documents import Document as LCDocument


@dataclass
class ChunkMapping:
    """Maps summary chunks to offsets in the full document"""
    chunk_id: str
    start: int
    end: int
    summary: str


class MetadataDocument:
    """
    Lightweight document wrapper with cached metadata and on-demand full text loading.

    This class minimizes memory usage by:
    1. Keeping only metadata (headlines, keyphrases, summary) in RAM
    2. Loading full document chunks from disk only when needed
    3. Releasing chunks after use

    Attributes:
        filename: Source document filename
        metadata: Document metadata dict
        headlines: Cached headlines extracted from document
        keyphrases: Cached keyphrases for relationship building
        summary: Brief document summary for knowledge graph
        char_count: Total character count of original document
        chunk_mappings: Maps chunks to original document offsets
    """

    def __init__(
        self,
        filename: str,
        metadata: Dict[str, Any],
        headlines: List[str],
        keyphrases: List[str],
        summary: str,
        char_count: int,
        chunk_mappings: List[Dict[str, Any]],
        source_path: Optional[Path] = None
    ):
        self.filename = filename
        self.metadata = metadata
        self.headlines = headlines
        self.keyphrases = keyphrases
        self.summary = summary
        self.char_count = char_count
        self.chunk_mappings = [
            ChunkMapping(
                chunk_id=cm['chunk_id'],
                start=cm['start'],
                end=cm['end'],
                summary=cm.get('summary', '')
            )
            for cm in chunk_mappings
        ]
        self.source_path = source_path or Path('knowledge-files') / filename

        # Cache for loaded chunks (cleared after use)
        self._chunk_cache: Dict[str, str] = {}

    @classmethod
    def from_cache(cls, cache_path: Path, source_dir: Path = None) -> 'MetadataDocument':
        """
        Load MetadataDocument from cached JSON file.

        Args:
            cache_path: Path to cached metadata JSON file
            source_dir: Directory containing original documents (default: knowledge-files)

        Returns:
            MetadataDocument instance
        """
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        source_dir = source_dir or Path('knowledge-files')
        source_path = source_dir / data['filename']

        return cls(
            filename=data['filename'],
            metadata=data.get('metadata', {}),
            headlines=data['headlines'],
            keyphrases=data['keyphrases'],
            summary=data['summary'],
            char_count=data['char_count'],
            chunk_mappings=data['chunk_mappings'],
            source_path=source_path
        )

    def to_langchain_document(self, use_summary: bool = True) -> LCDocument:
        """
        Convert to LangChain Document format.

        Args:
            use_summary: If True, use summary as page_content; if False, use full document

        Returns:
            LangChain Document with metadata
        """
        content = self.summary if use_summary else self.load_full_content()

        metadata = {
            **self.metadata,
            'filename': self.filename,
            'source': str(self.source_path),
            'char_count': self.char_count,
            'headlines': self.headlines,
            'keyphrases': self.keyphrases,
            'is_summary': use_summary
        }

        return LCDocument(page_content=content, metadata=metadata)

    def load_full_content(self) -> str:
        """
        Load full document content from disk.

        Returns:
            Full document text
        """
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source document not found: {self.source_path}")

        with open(self.source_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_chunk(self, chunk_id: str) -> str:
        """
        Load a specific chunk from the original document on-demand.

        Args:
            chunk_id: ID of chunk to load

        Returns:
            Chunk text content
        """
        # Check cache first
        if chunk_id in self._chunk_cache:
            return self._chunk_cache[chunk_id]

        # Find chunk mapping
        chunk_mapping = next((cm for cm in self.chunk_mappings if cm.chunk_id == chunk_id), None)
        if not chunk_mapping:
            raise ValueError(f"Chunk ID not found: {chunk_id}")

        # Load chunk from disk
        with open(self.source_path, 'r', encoding='utf-8') as f:
            f.seek(chunk_mapping.start)
            chunk_text = f.read(chunk_mapping.end - chunk_mapping.start)

        # Cache for potential reuse
        self._chunk_cache[chunk_id] = chunk_text

        return chunk_text

    def clear_cache(self):
        """Clear cached chunks to free memory"""
        self._chunk_cache.clear()

    def get_size_estimate(self) -> int:
        """
        Estimate memory footprint in bytes (excluding cached chunks).

        Returns:
            Estimated memory usage in bytes
        """
        size = 0
        size += len(self.summary.encode('utf-8'))
        size += sum(len(h.encode('utf-8')) for h in self.headlines)
        size += sum(len(k.encode('utf-8')) for k in self.keyphrases)
        size += sum(len(cm.summary.encode('utf-8')) for cm in self.chunk_mappings)
        return size

    def __repr__(self) -> str:
        return (
            f"MetadataDocument(filename='{self.filename}', "
            f"headlines={len(self.headlines)}, "
            f"keyphrases={len(self.keyphrases)}, "
            f"summary_len={len(self.summary)}, "
            f"chunks={len(self.chunk_mappings)}, "
            f"memory~{self.get_size_estimate()//1024}KB)"
        )


class MetadataCache:
    """
    Manager for cached document metadata.

    Handles loading multiple MetadataDocuments from cache directory.
    """

    def __init__(self, cache_dir: Path = None, source_dir: Path = None):
        """
        Initialize metadata cache manager.

        Args:
            cache_dir: Directory containing cached metadata (default: .cache/metadata)
            source_dir: Directory containing original documents (default: knowledge-files)
        """
        self.cache_dir = cache_dir or Path('.cache/metadata')
        self.source_dir = source_dir or Path('knowledge-files')

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> List[MetadataDocument]:
        """
        Load all cached metadata documents.

        Returns:
            List of MetadataDocument instances
        """
        metadata_docs = []

        for json_file in sorted(self.cache_dir.glob('*.json')):
            try:
                doc = MetadataDocument.from_cache(json_file, self.source_dir)
                metadata_docs.append(doc)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

        return metadata_docs

    def load_by_filename(self, filename: str) -> MetadataDocument:
        """
        Load specific document by filename.

        Args:
            filename: Name of the document file

        Returns:
            MetadataDocument instance
        """
        # Convert filename to cache JSON filename
        cache_filename = Path(filename).stem + '.json'
        cache_path = self.cache_dir / cache_filename

        if not cache_path.exists():
            raise FileNotFoundError(f"Cached metadata not found: {cache_path}")

        return MetadataDocument.from_cache(cache_path, self.source_dir)

    def is_cached(self, filename: str) -> bool:
        """
        Check if metadata for a document is cached.

        Args:
            filename: Name of the document file

        Returns:
            True if cached, False otherwise
        """
        cache_filename = Path(filename).stem + '.json'
        cache_path = self.cache_dir / cache_filename
        return cache_path.exists()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached metadata.

        Returns:
            Dict with cache statistics
        """
        cached_files = list(self.cache_dir.glob('*.json'))
        total_size = sum(f.stat().st_size for f in cached_files)

        return {
            'cached_files': len(cached_files),
            'total_cache_size': total_size,
            'cache_dir': str(self.cache_dir)
        }

    def __repr__(self) -> str:
        stats = self.get_cache_stats()
        return (
            f"MetadataCache(cache_dir='{self.cache_dir}', "
            f"cached_files={stats['cached_files']}, "
            f"total_size={stats['total_cache_size']//1024}KB)"
        )
