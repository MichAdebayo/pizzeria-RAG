from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging


class BaseExtractor(ABC):
    """Base class for all document extractors"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.supported_formats = []
    
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this extractor can handle the given file"""
        pass
    
    @abstractmethod
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from the file"""
        pass
    
    @abstractmethod
    def validate_extraction(self, result: Dict[str, Any]) -> float:
        """Return confidence score for extraction quality"""
        pass
    
    def _get_basic_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get basic file metadata"""
        return {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'extractor': self.__class__.__name__
        }
