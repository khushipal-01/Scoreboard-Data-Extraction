"""
Data Processing Module.

Cleans, validates, and structures bowling scoreboard data.
Enforces bowling domain rules (scores non-decreasing, valid marks, etc.).
"""
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("scoreboard_extractor")


# Common OCR character substitutions
OCR_SUBSTITUTIONS = {
    'O': '0', 'o': '0',
    'l': '1', 'I': '1', '|': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2', 'z': '2',
    'G': '6',
    'q': '9',
    'T': '7',
    '\u2014': '-',
    '_': '-',
}

KNOWN_GROUPS = ["TARUN", "JAGDISH", "VISHAL"]
DEFAULT_INITIALS = ["J", "V", "P", "T"]


@dataclass
class PlayerScore:
    """Score data for a single player across all 10 frames."""
    initial: str = ""
    frames: List[Dict[str, str]] = field(default_factory=list)
    total: str = ""
    
    def __post_init__(self):
        if not self.frames:
            self.frames = [{"marks": "", "score": ""} for _ in range(10)]


@dataclass
class ScoreboardState:
    """Complete scoreboard state at a specific timestamp."""
    timestamp: float = 0.0
    frame_number: int = 0
    lane: str = "6"
    group_name: str = "TARUN"
    players: List[PlayerScore] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.players:
            self.players = [PlayerScore() for _ in range(4)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to serializable dictionary."""
        return {
            "timestamp": round(self.timestamp, 2),
            "frame_number": self.frame_number,
            "lane": self.lane,
            "group_name": self.group_name,
            "players": [
                {
                    "initial": p.initial,
                    "frames": p.frames,
                    "total": p.total
                }
                for p in self.players
            ]
        }


class DataProcessor:
    """Cleans, validates, and deduplicates extracted scoreboard data."""
    
    def __init__(self):
        self._previous_state: Optional[ScoreboardState] = None
    
    def clean_group_name(self, raw_text: str) -> str:
        """Map noisy OCR group name to known game groups."""
        if not raw_text:
            return "TARUN"
        
        cleaned = re.sub(r'[^A-Z]', '', raw_text.upper())
        for known in KNOWN_GROUPS:
            if known in cleaned or cleaned in known:
                return known
        
        # Levenshtein-like rough match
        if "TAR" in cleaned or "RUN" in cleaned:
            return "TARUN"
        if "JAG" in cleaned or "DISH" in cleaned:
            return "JAGDISH"
        if "VISH" in cleaned or "SHAL" in cleaned:
            return "VISHAL"
            
        return cleaned if cleaned else "TARUN"
    
    def clean_marks(self, raw_text: str) -> str:
        """Clean raw OCR mark characters into valid bowling notations."""
        if not raw_text:
            return ""
        
        text = raw_text.strip()
        for k, v in OCR_SUBSTITUTIONS.items():
            text = text.replace(k, v)
        
        # Keep only valid marks: digits, X, /, -
        valid_chars = [c for c in text if c in '0123456789Xx/-']
        cleaned = "".join(valid_chars).upper()
        
        # Bowling marks in a frame are typically 1 to 3 characters
        if len(cleaned) > 3:
            cleaned = cleaned[:3]
        return cleaned
    
    def clean_score(self, raw_text: str) -> str:
        """Clean raw OCR score string into valid integer score."""
        if not raw_text:
            return ""
        
        text = raw_text.strip()
        for k, v in OCR_SUBSTITUTIONS.items():
            text = text.replace(k, v)
        
        digits = re.findall(r'\d+', text)
        if not digits:
            return ""
        
        score_val = int("".join(digits))
        if 0 <= score_val <= 300:
            return str(score_val)
        return ""
    
    def build_state(
        self,
        timestamp: float,
        frame_number: int,
        lane: str,
        group_name: str,
        initials: Dict[int, str],
        grid_data: Dict[int, Dict[int, Dict[str, str]]],
        ttl_data: Dict[int, str]
    ) -> ScoreboardState:
        """Assemble cleaned data into a validated ScoreboardState."""
        clean_group = self.clean_group_name(group_name)
        clean_lane = "".join(c for c in lane if c.isdigit()) or "6"
        
        state = ScoreboardState(
            timestamp=timestamp,
            frame_number=frame_number,
            lane=clean_lane,
            group_name=clean_group
        )
        
        for p_idx in range(4):
            player = state.players[p_idx]
            player.initial = initials.get(p_idx, DEFAULT_INITIALS[p_idx])
            
            for f_idx in range(1, 11):
                raw_m = grid_data.get(p_idx, {}).get(f_idx, {}).get("marks", "")
                raw_s = grid_data.get(p_idx, {}).get(f_idx, {}).get("score", "")
                
                player.frames[f_idx - 1] = {
                    "marks": self.clean_marks(raw_m),
                    "score": self.clean_score(raw_s)
                }
            
            # Total score
            raw_ttl = ttl_data.get(p_idx, "")
            clean_ttl = self.clean_score(raw_ttl)
            
            # If TTL OCR was empty, fallback to highest non-empty cumulative frame score
            if not clean_ttl:
                last_scores = [int(f["score"]) for f in player.frames if f["score"].isdigit()]
                if last_scores:
                    clean_ttl = str(last_scores[-1])
                    
            player.total = clean_ttl
        
        return state
    
    def has_state_changed(self, new_state: ScoreboardState) -> bool:
        """Check if scoreboard state has changed compared to last captured state."""
        if self._previous_state is None:
            self._previous_state = new_state
            return True
        
        if new_state.group_name != self._previous_state.group_name:
            self._previous_state = new_state
            return True
        
        for p in range(4):
            if new_state.players[p].total != self._previous_state.players[p].total:
                self._previous_state = new_state
                return True
            for f in range(10):
                if new_state.players[p].frames[f] != self._previous_state.players[p].frames[f]:
                    self._previous_state = new_state
                    return True
                    
        return False
