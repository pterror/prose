"""Tests for vocabulary system."""

import pytest

from src.data.asg_builder import ASTNode, NodeType
from src.data.vocabulary import Vocabulary


class TestVocabulary:
    """Test vocabulary building and encoding/decoding."""

    def test_special_tokens(self):
        """Test special tokens have fixed IDs."""
        vocab = Vocabulary()

        assert vocab.MASK_TOKEN == "<MASK>"
        assert vocab.PAD_TOKEN == "<PAD>"
        assert vocab.UNK_TOKEN == "<UNK>"

        assert vocab.mask_token_id == 0
        assert vocab.pad_token_id == 1
        assert vocab.unk_token_id == 2

    def test_token_extraction(self):
        """Test extracting tokens from AST."""
        vocab = Vocabulary()

        # Simple arithmetic: (+ a b)
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="+"),
                ASTNode(NodeType.SYMBOL, value="a"),
                ASTNode(NodeType.SYMBOL, value="b"),
            ],
        )

        tokens = vocab._extract_tokens_from_ast(ast)
        assert "+" in tokens
        assert "a" in tokens
        assert "b" in tokens

    def test_token_encoding_decoding(self):
        """Test encoding and decoding tokens."""
        vocab = Vocabulary()

        # Manually add some tokens
        vocab._add_token("+")
        vocab._add_token("x")
        vocab._add_token("5")

        # Encode
        plus_id = vocab.encode("+")
        x_id = vocab.encode("x")
        num_id = vocab.encode("5")

        # Should get unique IDs
        assert plus_id >= 3  # After special tokens
        assert x_id >= 3
        assert num_id >= 3
        assert len({plus_id, x_id, num_id}) == 3

        # Decode
        assert vocab.decode(plus_id) == "+"
        assert vocab.decode(x_id) == "x"
        assert vocab.decode(num_id) == "5"

    def test_unknown_token(self):
        """Test handling of unknown tokens."""
        vocab = Vocabulary()

        # Unknown token should return UNK_TOKEN_ID
        unknown_id = vocab.encode("UNKNOWN_TOKEN")
        assert unknown_id == vocab.unk_token_id

        # Decoding unknown ID should return UNK_TOKEN
        assert vocab.decode(99999) == vocab.UNK_TOKEN

    def test_batch_encoding(self):
        """Test batch encoding/decoding."""
        vocab = Vocabulary()
        vocab._add_token("+")
        vocab._add_token("x")
        vocab._add_token("1")

        tokens = ["+", "x", "1"]
        ids = vocab.encode_batch(tokens)
        decoded = vocab.decode_batch(ids)

        assert decoded == tokens

    def test_vocabulary_size(self):
        """Test vocabulary size property."""
        vocab = Vocabulary()

        # Initially just special tokens
        assert vocab.vocab_size == 3

        # Add more tokens
        vocab._add_token("+")
        vocab._add_token("-")
        assert vocab.vocab_size == 5

    def test_save_load(self, tmp_path):
        """Test saving and loading vocabulary."""
        vocab = Vocabulary()
        vocab._add_token("+")
        vocab._add_token("x")
        vocab._add_token("1")

        # Save
        path = tmp_path / "vocab.json"
        vocab.save(path)

        # Load
        loaded_vocab = Vocabulary.load(path)

        # Check equality
        assert loaded_vocab.vocab_size == vocab.vocab_size
        assert loaded_vocab.token_to_id == vocab.token_to_id
        assert loaded_vocab.id_to_token == vocab.id_to_token

    def test_roundtrip_ast(self):
        """Test extracting tokens from AST and encoding them."""
        vocab = Vocabulary()

        # Define AST: (define (fact n) (if (= n 0) 1 (* n (fact (- n 1)))))
        ast = ASTNode(
            NodeType.DEFINE,
            value="fact",
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.SYMBOL, value="fact"),
                        ASTNode(NodeType.SYMBOL, value="n"),
                    ],
                ),
                ASTNode(
                    NodeType.IF,
                    children=[
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value="="),
                                ASTNode(NodeType.SYMBOL, value="n"),
                                ASTNode(NodeType.NUMBER, value=0),
                            ],
                        ),
                        ASTNode(NodeType.NUMBER, value=1),
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value="*"),
                                ASTNode(NodeType.SYMBOL, value="n"),
                                ASTNode(
                                    NodeType.LIST,
                                    children=[
                                        ASTNode(NodeType.SYMBOL, value="fact"),
                                        ASTNode(
                                            NodeType.LIST,
                                            children=[
                                                ASTNode(NodeType.OPERATOR, value="-"),
                                                ASTNode(NodeType.SYMBOL, value="n"),
                                                ASTNode(NodeType.NUMBER, value=1),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )

        # Extract and build vocabulary
        tokens = vocab._extract_tokens_from_ast(ast)
        for token in set(tokens):
            vocab._add_token(token)

        # Encode all tokens
        token_ids = vocab.encode_batch(tokens)

        # Decode back
        decoded_tokens = vocab.decode_batch(token_ids)

        # Should match original
        assert decoded_tokens == tokens

        # Check some expected tokens
        assert "define" in tokens
        assert "if" in tokens
        assert "fact" in tokens
        assert "=" in tokens
        assert "*" in tokens
        assert "-" in tokens
        assert "n" in tokens
        assert "0" in tokens
        assert "1" in tokens
