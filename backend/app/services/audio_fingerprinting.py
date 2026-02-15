"""Audio Fingerprinting and Duplicate Detection.

Uses Chromaprint for acoustic fingerprinting.
"""

import hashlib
import io
from typing import List, Optional
import structlog

logger = structlog.get_logger(__name__)


class AudioFingerprinter:
    """Generate and compare audio fingerprints for duplicate detection."""
    
    def __init__(self):
        self.fingerprints_cache = {}  # track_id -> fingerprint
    
    def generate_fingerprint(self, audio_data: bytes) -> str:
        """Generate acoustic fingerprint from audio data.
        
        Args:
            audio_data: Audio file bytes
        
        Returns:
            Fingerprint string
        """
        try:
            import acoustid
            import audioread
            
            # Decode audio
            audio_file = io.BytesIO(audio_data)
            
            # Generate fingerprint using chromaprint
            duration, fingerprint = acoustid.fingerprint_file(audio_file)
            
            logger.info("fingerprint_generated", duration=duration)
            
            return fingerprint
            
        except ImportError:
            logger.warning("acoustid_not_available_using_hash_fallback")
            # Fallback: use hash of audio data
            return hashlib.sha256(audio_data).hexdigest()
        except Exception as e:
            logger.error("fingerprint_generation_failed", error=str(e))
            # Fallback
            return hashlib.sha256(audio_data).hexdigest()
    
    def compare_fingerprints(self, fp1: str, fp2: str) -> float:
        """Compare two fingerprints and return similarity score.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if fp1 == fp2:
            return 1.0
        
        try:
            import acoustid
            
            # Compare using acoustid's comparison
            # Note: This is a simplified approach
            # In production, use proper acoustid API or chromaprint compare
            
            # Calculate edit distance
            distance = self._levenshtein_distance(fp1, fp2)
            max_len = max(len(fp1), len(fp2))
            
            if max_len == 0:
                return 0.0
            
            similarity = 1.0 - (distance / max_len)
            return similarity
            
        except ImportError:
            # Fallback: exact match only
            return 1.0 if fp1 == fp2 else 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    async def find_duplicates(self, track_id: str, fingerprint: str, threshold: float = 0.9) -> List[str]:
        """Find duplicate tracks by comparing fingerprints.
        
        Args:
            track_id: ID of track to check
            fingerprint: Fingerprint of track
            threshold: Similarity threshold (0.9 = 90% similar)
        
        Returns:
            List of duplicate track IDs
        """
        duplicates = []
        
        # Compare with all cached fingerprints
        for other_id, other_fp in self.fingerprints_cache.items():
            if other_id == track_id:
                continue
            
            similarity = self.compare_fingerprints(fingerprint, other_fp)
            
            if similarity >= threshold:
                duplicates.append(other_id)
                logger.info(
                    "duplicate_found",
                    track_id=track_id,
                    duplicate_id=other_id,
                    similarity=similarity,
                )
        
        return duplicates
    
    def cache_fingerprint(self, track_id: str, fingerprint: str):
        """Cache fingerprint for future comparisons."""
        self.fingerprints_cache[track_id] = fingerprint
        
        # Limit cache size
        if len(self.fingerprints_cache) > 10000:
            # Remove oldest 20%
            keys_to_remove = list(self.fingerprints_cache.keys())[:2000]
            for key in keys_to_remove:
                del self.fingerprints_cache[key]
    
    async def analyze_track_uniqueness(self, audio_data: bytes) -> dict:
        """Analyze track uniqueness.
        
        Args:
            audio_data: Audio file bytes
        
        Returns:
            Analysis dict with fingerprint and duplicate check
        """
        fingerprint = self.generate_fingerprint(audio_data)
        
        # Find similar tracks
        duplicates = await self.find_duplicates("temp", fingerprint, threshold=0.85)
        
        return {
            "fingerprint": fingerprint[:32],  # Truncate for display
            "is_unique": len(duplicates) == 0,
            "similar_tracks_count": len(duplicates),
            "similarity_threshold": 0.85,
        }


# Global instance
fingerprinter = AudioFingerprinter()
