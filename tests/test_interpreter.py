"""Tests for Mini-Lisp interpreter."""

import pytest

from src.data.asg_builder import ASTNode, NodeType
from src.runtime.interpreter import (
    DivisionByZeroError,
    Environment,
    InterpreterError,
    MiniLispInterpreter,
    TestCase,
    TypeMismatchError,
    UndefinedVariableError,
)


class TestEnvironment:
    """Test environment and variable scoping."""

    def test_define_and_lookup(self):
        """Test variable definition and lookup."""
        env = Environment()
        env.define("x", 42)
        assert env.lookup("x") == 42

    def test_undefined_variable(self):
        """Test undefined variable raises error."""
        env = Environment()
        with pytest.raises(UndefinedVariableError):
            env.lookup("undefined")

    def test_parent_scope(self):
        """Test lexical scoping with parent environment."""
        parent = Environment()
        parent.define("x", 10)

        child = Environment(parent=parent)
        child.define("y", 20)

        # Child can see parent's variables
        assert child.lookup("x") == 10
        assert child.lookup("y") == 20

        # Parent cannot see child's variables
        assert parent.lookup("x") == 10
        with pytest.raises(UndefinedVariableError):
            parent.lookup("y")

    def test_shadowing(self):
        """Test variable shadowing."""
        parent = Environment()
        parent.define("x", 10)

        child = Environment(parent=parent)
        child.define("x", 20)  # Shadow parent's x

        assert child.lookup("x") == 20
        assert parent.lookup("x") == 10


class TestBasicEvaluation:
    """Test basic expression evaluation."""

    def test_number(self):
        """Test number evaluation."""
        interp = MiniLispInterpreter()
        ast = ASTNode(NodeType.NUMBER, value=42)
        assert interp.eval(ast) == 42

    def test_symbol(self):
        """Test symbol lookup."""
        interp = MiniLispInterpreter()
        env = Environment(parent=interp.global_env)
        env.define("x", 100)

        ast = ASTNode(NodeType.SYMBOL, value="x")
        assert interp.eval(ast, env) == 100

    def test_arithmetic_addition(self):
        """Test addition: (+ 1 2) = 3."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="+"),
                ASTNode(NodeType.NUMBER, value=1),
                ASTNode(NodeType.NUMBER, value=2),
            ],
        )
        assert interp.eval(ast) == 3

    def test_arithmetic_subtraction(self):
        """Test subtraction: (- 5 3) = 2."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="-"),
                ASTNode(NodeType.NUMBER, value=5),
                ASTNode(NodeType.NUMBER, value=3),
            ],
        )
        assert interp.eval(ast) == 2

    def test_arithmetic_multiplication(self):
        """Test multiplication: (* 3 4) = 12."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="*"),
                ASTNode(NodeType.NUMBER, value=3),
                ASTNode(NodeType.NUMBER, value=4),
            ],
        )
        assert interp.eval(ast) == 12

    def test_arithmetic_division(self):
        """Test division: (/ 10 2) = 5."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="/"),
                ASTNode(NodeType.NUMBER, value=10),
                ASTNode(NodeType.NUMBER, value=2),
            ],
        )
        assert interp.eval(ast) == 5

    def test_division_by_zero(self):
        """Test division by zero raises error."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="/"),
                ASTNode(NodeType.NUMBER, value=10),
                ASTNode(NodeType.NUMBER, value=0),
            ],
        )
        with pytest.raises(DivisionByZeroError):
            interp.eval(ast)

    def test_comparison_equal(self):
        """Test equality: (= 5 5) = True."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="="),
                ASTNode(NodeType.NUMBER, value=5),
                ASTNode(NodeType.NUMBER, value=5),
            ],
        )
        assert interp.eval(ast) is True

    def test_comparison_less_than(self):
        """Test less than: (< 3 5) = True."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="<"),
                ASTNode(NodeType.NUMBER, value=3),
                ASTNode(NodeType.NUMBER, value=5),
            ],
        )
        assert interp.eval(ast) is True


class TestControlFlow:
    """Test control flow structures."""

    def test_if_true_branch(self):
        """Test if statement - true branch: (if (= 5 5) 1 2) = 1."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.IF,
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value="="),
                        ASTNode(NodeType.NUMBER, value=5),
                        ASTNode(NodeType.NUMBER, value=5),
                    ],
                ),
                ASTNode(NodeType.NUMBER, value=1),
                ASTNode(NodeType.NUMBER, value=2),
            ],
        )
        assert interp.eval(ast) == 1

    def test_if_false_branch(self):
        """Test if statement - false branch: (if (< 5 3) 1 2) = 2."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.IF,
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value="<"),
                        ASTNode(NodeType.NUMBER, value=5),
                        ASTNode(NodeType.NUMBER, value=3),
                    ],
                ),
                ASTNode(NodeType.NUMBER, value=1),
                ASTNode(NodeType.NUMBER, value=2),
            ],
        )
        assert interp.eval(ast) == 2


class TestVariableBinding:
    """Test variable binding with let."""

    def test_let_single_binding(self):
        """Test let with single binding: (let ((x 5)) x) = 5."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.LET,
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.SYMBOL, value="x"),
                                ASTNode(NodeType.NUMBER, value=5),
                            ],
                        ),
                    ],
                ),
                ASTNode(NodeType.SYMBOL, value="x"),
            ],
        )
        assert interp.eval(ast) == 5

    def test_let_multiple_bindings(self):
        """Test let with multiple bindings: (let ((x 1) (y 2)) (+ x y)) = 3."""
        interp = MiniLispInterpreter()
        ast = ASTNode(
            NodeType.LET,
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.SYMBOL, value="x"),
                                ASTNode(NodeType.NUMBER, value=1),
                            ],
                        ),
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.SYMBOL, value="y"),
                                ASTNode(NodeType.NUMBER, value=2),
                            ],
                        ),
                    ],
                ),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value="+"),
                        ASTNode(NodeType.SYMBOL, value="x"),
                        ASTNode(NodeType.SYMBOL, value="y"),
                    ],
                ),
            ],
        )
        assert interp.eval(ast) == 3


class TestLambda:
    """Test lambda expressions."""

    def test_lambda_simple(self):
        """Test lambda: ((lambda (x) (* x 2)) 5) = 10."""
        interp = MiniLispInterpreter()

        # Build AST: ((lambda (x) (* x 2)) 5)
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(
                    NodeType.LAMBDA,
                    children=[
                        ASTNode(
                            NodeType.LIST,
                            children=[ASTNode(NodeType.SYMBOL, value="x")],
                        ),
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
        assert interp.eval(ast) == 10

    def test_lambda_two_params(self):
        """Test lambda with two parameters: ((lambda (x y) (+ x y)) 3 4) = 7."""
        interp = MiniLispInterpreter()

        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(
                    NodeType.LAMBDA,
                    children=[
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.SYMBOL, value="x"),
                                ASTNode(NodeType.SYMBOL, value="y"),
                            ],
                        ),
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value="+"),
                                ASTNode(NodeType.SYMBOL, value="x"),
                                ASTNode(NodeType.SYMBOL, value="y"),
                            ],
                        ),
                    ],
                ),
                ASTNode(NodeType.NUMBER, value=3),
                ASTNode(NodeType.NUMBER, value=4),
            ],
        )
        assert interp.eval(ast) == 7


class TestDefine:
    """Test function definitions."""

    def test_define_function(self):
        """Test function definition and call: (define (add1 x) (+ x 1)), (add1 5) = 6."""
        interp = MiniLispInterpreter()
        env = Environment(parent=interp.global_env)

        # Define function
        define_ast = ASTNode(
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
        interp.eval(define_ast, env)

        # Call function
        call_ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.SYMBOL, value="add1"),
                ASTNode(NodeType.NUMBER, value=5),
            ],
        )
        assert interp.eval(call_ast, env) == 6

    def test_define_recursive(self):
        """Test recursive function: factorial."""
        interp = MiniLispInterpreter()
        env = Environment(parent=interp.global_env)

        # (define (fact n) (if (= n 0) 1 (* n (fact (- n 1)))))
        define_ast = ASTNode(
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
        interp.eval(define_ast, env)

        # Test fact(5) = 120
        call_ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.SYMBOL, value="fact"),
                ASTNode(NodeType.NUMBER, value=5),
            ],
        )
        assert interp.eval(call_ast, env) == 120


class TestTestExecution:
    """Test running tests on programs."""

    def test_run_tests_passing(self):
        """Test running passing tests."""
        interp = MiniLispInterpreter()

        # Define simple function: (define (add1 x) (+ x 1))
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

        # Create tests
        tests = [
            TestCase(inputs=[5], expected_output=6),
            TestCase(inputs=[0], expected_output=1),
            TestCase(inputs=[-1], expected_output=0),
        ]

        # Run tests
        results = interp.run_tests(program, tests)

        assert all(results)
        assert all(test.passing for test in tests)

    def test_run_tests_failing(self):
        """Test running failing tests."""
        interp = MiniLispInterpreter()

        # Define WRONG function: (define (add1 x) (+ x 2))  # Bug: should be + 1
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
                        ASTNode(NodeType.NUMBER, value=2),  # BUG!
                    ],
                ),
            ],
        )

        # Create tests (expecting correct behavior)
        tests = [
            TestCase(inputs=[5], expected_output=6),
        ]

        # Run tests
        results = interp.run_tests(program, tests)

        assert not all(results)
        assert not tests[0].passing
        assert tests[0].actual_output == 7  # Got +2 instead of +1


class TestExecutionTracing:
    """Test execution tracing for test feedback."""

    def test_trace_simple_execution(self):
        """Test tracing nodes visited during execution."""
        interp = MiniLispInterpreter()

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

        # Trace execution
        traced_nodes = interp.trace_execution(program, test_inputs=[5])

        # Should have traced multiple nodes
        assert len(traced_nodes) > 0

        # Trace should include visited nodes (exact count depends on implementation)
        # At minimum, should have traced: DEFINE, body nodes, etc.
