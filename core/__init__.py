"""Ядро обработки изображений."""
from .processor import ImageProcessor
from .analyzer import analyze_frame

__all__ = ["ImageProcessor", "analyze_frame"]