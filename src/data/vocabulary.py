"""Vocabulary system for token-level prediction in Mini-Lisp programs."""

from collections import Counter
from pathlib import Path
from typing import Any

from src.data.asg_builder import ASTNode, NodeType


class Vocabulary:
    """Token vocabulary for value-level prediction.
    
    Builds a vocabulary from Mini-Lisp templates, mapping tokens to integer IDs.
    Supports special tokens (<MASK>, <PAD>, <UNK>) and handles operators,
    keywords, symbols, and numbers.
    """
    
    # Special tokens (fixed IDs)
    MASK_TOKEN = "<MASK>"
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self):
        """Initialize empty vocabulary."""
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._next_id = 0
        
        # Add special tokens first (guaranteed IDs 0, 1, 2)
        self._add_token(self.MASK_TOKEN)
        self._add_token(self.PAD_TOKEN)
        self._add_token(self.UNK_TOKEN)
        
        # Track token frequencies for analysis
        self.token_counts: Counter = Counter()
        
    def _add_token(self, token: str) -> int:
        """Add token to vocabulary if not present."""
        if token not in self.token_to_id:
            token_id = self._next_id
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            self._next_id += 1
            return token_id
        return self.token_to_id[token]
    
    def build_from_templates(self, templates: list[Any], num_samples: int = 1000) -> None:
        """Extract all unique tokens from template programs.
        
        Args:
            templates: List of ProgramTemplate instances
            num_samples: Number of samples to generate per template
        """
        import random
        
        # Collect tokens from all template samples
        all_tokens = []
        
        for template in templates:
            rng = random.Random(42)
            for _ in range(num_samples // len(templates)):
                ast, _ = template.generate(rng)
                tokens = self._extract_tokens_from_ast(ast)
                all_tokens.extend(tokens)
        
        # Count token frequencies
        self.token_counts = Counter(all_tokens)
        
        # Add all unique tokens to vocabulary (sorted for determinism)
        unique_tokens = sorted(set(all_tokens))
        for token in unique_tokens:
            self._add_token(token)
    
    def _extract_tokens_from_ast(self, node: ASTNode) -> list[str]:
        """Recursively extract all tokens from AST.
        
        Returns:
            List of token strings representing the AST nodes
        """
        tokens = []
        
        # Add node type keyword if structural
        if node.node_type == NodeType.DEFINE:
            tokens.append("define")
        elif node.node_type == NodeType.LAMBDA:
            tokens.append("lambda")
        elif node.node_type == NodeType.IF:
            tokens.append("if")
        elif node.node_type == NodeType.LET:
            tokens.append("let")
        
        # Add node value if present
        if node.value is not None:
            # Convert value to string token
            if isinstance(node.value, (int, float)):
                tokens.append(str(node.value))
            else:
                tokens.append(str(node.value))
        
        # Recursively process children
        for child in node.children or []:
            tokens.extend(self._extract_tokens_from_ast(child))
        
        return tokens
    
    def encode(self, token: str) -> int:
        """Convert token to ID. Returns UNK_TOKEN ID if not found."""
        return self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN])
    
    def decode(self, token_id: int) -> str:
        """Convert ID to token. Returns UNK_TOKEN if not found."""
        return self.id_to_token.get(token_id, self.UNK_TOKEN)
    
    def encode_batch(self, tokens: list[str]) -> list[int]:
        """Encode a batch of tokens."""
        return [self.encode(token) for token in tokens]
    
    def decode_batch(self, token_ids: list[int]) -> list[str]:
        """Decode a batch of token IDs."""
        return [self.decode(token_id) for token_id in token_ids]
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.token_to_id)
    
    @property
    def mask_token_id(self) -> int:
        """ID of the mask token."""
        return self.token_to_id[self.MASK_TOKEN]
    
    @property
    def pad_token_id(self) -> int:
        """ID of the pad token."""
        return self.token_to_id[self.PAD_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        """ID of the unknown token."""
        return self.token_to_id[self.UNK_TOKEN]
    
    def save(self, path: Path) -> None:
        """Save vocabulary to file."""
        import json
        
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
            "vocab_size": self.vocab_size,
            "token_counts": dict(self.token_counts),
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from file."""
        import json
        
        with open(path) as f:
            data = json.load(f)
        
        vocab = cls()
        vocab.token_to_id = data["token_to_id"]
        vocab.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        vocab._next_id = data["vocab_size"]
        vocab.token_counts = Counter(data.get("token_counts", {}))
        
        return vocab
    
    def get_stats(self) -> dict[str, Any]:
        """Get vocabulary statistics."""
        return {
            "vocab_size": self.vocab_size,
            "num_special_tokens": 3,
            "num_regular_tokens": self.vocab_size - 3,
            "most_common": self.token_counts.most_common(20),
            "rarest": self.token_counts.most_common()[-20:] if len(self.token_counts) > 20 else [],
        }
    
    def __repr__(self) -> str:
        return f"Vocabulary(size={self.vocab_size}, tokens={list(self.token_to_id.keys())[:10]}...)"
