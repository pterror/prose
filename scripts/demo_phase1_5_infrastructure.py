#!/usr/bin/env python3
"""Demonstrate the Phase 1.5 infrastructure: vocabulary, interpreter, and test generation."""

import random

from src.data.asg_builder import ASTNode, NodeType
from src.data.synthetic_gen import RecursionTemplate
from src.data.test_generator import TestGenerator
from src.data.vocabulary import Vocabulary
from src.runtime.interpreter import MiniLispInterpreter


def demo_vocabulary():
    """Demonstrate vocabulary building."""
    print("=" * 60)
    print("Demo 1: Vocabulary System")
    print("=" * 60)

    vocab = Vocabulary()

    # Create a simple AST
    ast = ASTNode(
        NodeType.LIST,
        children=[
            ASTNode(NodeType.OPERATOR, value="+"),
            ASTNode(NodeType.SYMBOL, value="x"),
            ASTNode(NodeType.NUMBER, value=5),
        ],
    )

    # Extract tokens
    tokens = vocab._extract_tokens_from_ast(ast)
    print(f"\nExtracted tokens from '(+ x 5)': {tokens}")

    # Build vocabulary
    for token in set(tokens):
        vocab._add_token(token)

    # Encode/decode
    encoded = vocab.encode_batch(tokens)
    decoded = vocab.decode_batch(encoded)

    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Roundtrip successful: {tokens == decoded}")
    print(f"Vocabulary size: {vocab.vocab_size}")


def demo_interpreter():
    """Demonstrate interpreter evaluation."""
    print("\n" + "=" * 60)
    print("Demo 2: Mini-Lisp Interpreter")
    print("=" * 60)

    interp = MiniLispInterpreter()

    # Example 1: Simple arithmetic
    print("\n[Example 1] Simple arithmetic: (+ 3 (* 2 4))")
    ast1 = ASTNode(
        NodeType.LIST,
        children=[
            ASTNode(NodeType.OPERATOR, value="+"),
            ASTNode(NodeType.NUMBER, value=3),
            ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value="*"),
                    ASTNode(NodeType.NUMBER, value=2),
                    ASTNode(NodeType.NUMBER, value=4),
                ],
            ),
        ],
    )
    result1 = interp.eval(ast1)
    print(f"Result: {result1}")

    # Example 2: Lambda function
    print("\n[Example 2] Lambda: ((lambda (x) (* x 2)) 5)")
    ast2 = ASTNode(
        NodeType.LIST,
        children=[
            ASTNode(
                NodeType.LAMBDA,
                children=[
                    ASTNode(NodeType.LIST, children=[ASTNode(NodeType.SYMBOL, value="x")]),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value="*"),
                            ASTNode(NodeType.SYMBOL, value="x"),
                            ASTNode(NodeType.NUMBER, value=2),
                        ],
                    ),
                ],
            ),
            ASTNode(NodeType.NUMBER, value=5),
        ],
    )
    result2 = interp.eval(ast2)
    print(f"Result: {result2}")

    # Example 3: Define and call recursive function
    print("\n[Example 3] Recursive function: factorial")
    from src.runtime.interpreter import Environment

    env = Environment(parent=interp.global_env)

    # (define (fact n) (if (= n 0) 1 (* n (fact (- n 1)))))
    ast3 = ASTNode(
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
    interp.eval(ast3, env)

    # Test factorial
    for n in [0, 1, 3, 5]:
        func = env.lookup("fact")
        result = func(n)
        print(f"  fact({n}) = {result}")


def demo_test_generation():
    """Demonstrate test generation."""
    print("\n" + "=" * 60)
    print("Demo 3: Test Generation")
    print("=" * 60)

    # Generate a program using template
    rng = random.Random(42)
    template = RecursionTemplate("recursion", "Recursive factorial")
    program, metadata = template.generate(rng)

    print(f"\nGenerated program from template: {template.template_id}")
    print(f"  Depth: {metadata.depth}")
    print(f"  Nodes: {metadata.num_nodes}")
    print(f"  Has recursion: {metadata.has_recursion}")

    # Generate tests
    test_gen = TestGenerator()
    tests = test_gen.generate_tests_for_function(
        program=program,
        func_name="fact",  # From RecursionTemplate
        num_params=1,
        rng=rng,
        num_tests=5,
        param_range=(0, 6),  # Small range for factorial
    )

    print(f"\nGenerated {len(tests)} test cases:")
    for i, test in enumerate(tests, 1):
        print(f"  Test {i}: f({test.inputs}) = {test.expected_output}")

    # Run tests to verify
    interp = MiniLispInterpreter()
    results = interp.run_tests(program, tests)

    print(f"\nTest results: {sum(results)}/{len(results)} passing")
    for test in tests:
        print(f"  {test}")


def demo_execution_tracing():
    """Demonstrate execution tracing."""
    print("\n" + "=" * 60)
    print("Demo 4: Execution Tracing")
    print("=" * 60)

    interp = MiniLispInterpreter()
    from src.runtime.interpreter import Environment

    env = Environment(parent=interp.global_env)

    # Define simple function
    program = ASTNode(
        NodeType.DEFINE,
        value="add1",
        children=[
            ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.SYMBOL, value="add1"),
                    ASTNode(NodeType.SYMBOL, value="x"),
                ],
            ),
            ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value="+"),
                    ASTNode(NodeType.SYMBOL, value="x"),
                    ASTNode(NodeType.NUMBER, value=1),
                ],
            ),
        ],
    )

    print("\nProgram: (define (add1 x) (+ x 1))")

    # Trace execution
    traced = interp.trace_execution(program, test_inputs=[5])
    print(f"\nNodes visited during execution: {len(traced)}")
    print(f"  Traced node IDs: {list(traced)[:5]}... (showing first 5)")

    print("\nThis trace can be used to mark failing nodes for test feedback!")


def main():
    """Run all demonstrations."""
    print("\n" + "█" * 60)
    print("Phase 1.5 Infrastructure Demo")
    print("Vocabulary + Interpreter + Test Generation")
    print("█" * 60)

    demo_vocabulary()
    demo_interpreter()
    demo_test_generation()
    demo_execution_tracing()

    print("\n" + "█" * 60)
    print("✓ All infrastructure components working!")
    print("█" * 60)
    print("\nNext steps:")
    print("  1. Extend node features for Phase 1.5")
    print("  2. Implement IterativeGraphUNet")
    print("  3. Create trajectory generation")
    print("  4. Implement training with test feedback")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
