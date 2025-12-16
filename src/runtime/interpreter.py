"""Mini-Lisp interpreter with execution tracing for test-driven refinement."""

from dataclasses import dataclass
from typing import Any, Callable

from src.data.asg_builder import ASTNode, NodeType


class InterpreterError(Exception):
    """Base class for interpreter errors."""

    pass


class UndefinedVariableError(InterpreterError):
    """Variable not found in environment."""

    pass


class TypeMismatchError(InterpreterError):
    """Type error during evaluation."""

    pass


class DivisionByZeroError(InterpreterError):
    """Division by zero error."""

    pass


@dataclass
class TestCase:
    """Represents a test case for a program."""

    inputs: list[Any]
    expected_output: Any
    actual_output: Any | None = None
    passing: bool | None = None

    def __repr__(self) -> str:
        status = "✓" if self.passing else "✗" if self.passing is False else "?"
        return f"{status} f({self.inputs}) = {self.expected_output} (got: {self.actual_output})"


class Environment:
    """Environment for variable bindings with lexical scoping."""

    def __init__(self, parent: "Environment | None" = None):
        """Create environment, optionally with a parent scope."""
        self.bindings: dict[str, Any] = {}
        self.parent = parent

    def define(self, name: str, value: Any) -> None:
        """Define a variable in this environment."""
        self.bindings[name] = value

    def lookup(self, name: str) -> Any:
        """Look up variable value, checking parent scopes if needed."""
        if name in self.bindings:
            return self.bindings[name]
        elif self.parent is not None:
            return self.parent.lookup(name)
        else:
            raise UndefinedVariableError(f"Undefined variable: {name}")

    def exists(self, name: str) -> bool:
        """Check if variable exists in this scope or parent scopes."""
        return name in self.bindings or (self.parent is not None and self.parent.exists(name))


class MiniLispInterpreter:
    """Interpreter for Mini-Lisp programs with execution tracing.

    Supports:
    - Primitives: +, -, *, /, <, >, =
    - Control flow: if
    - Variable binding: let, define
    - Functions: lambda
    - Recursion
    - Execution tracing for test feedback
    """

    def __init__(self, max_steps: int = 1000):
        """Initialize interpreter.

        Args:
            max_steps: Maximum execution steps to prevent infinite loops
        """
        self.max_steps = max_steps
        self.step_count = 0
        self.trace_mode = False
        self.traced_nodes: set[int] = set()
        self.node_id_map: dict[int, ASTNode] = {}  # Maps object ID to node

        # Global environment with built-in primitives
        self.global_env = Environment()
        self._define_primitives()

    def _define_primitives(self) -> None:
        """Define built-in primitive operations."""
        # Arithmetic
        self.global_env.define("+", lambda *args: sum(args))
        self.global_env.define("-", lambda a, b: a - b if len([a, b]) == 2 else -a)
        self.global_env.define("*", lambda *args: eval_multiply(args))
        self.global_env.define("/", lambda a, b: safe_divide(a, b))

        # Comparison
        self.global_env.define("<", lambda a, b: a < b)
        self.global_env.define(">", lambda a, b: a > b)
        self.global_env.define("=", lambda a, b: a == b)

        # Higher-order functions (simplified)
        self.global_env.define("map", lambda f, lst: [f(x) for x in lst])
        self.global_env.define("filter", lambda pred, lst: [x for x in lst if pred(x)])
        self.global_env.define("reduce", lambda f, lst: reduce_impl(f, lst))

    def eval(self, node: ASTNode, env: Environment | None = None) -> Any:
        """Evaluate AST node in environment.

        Args:
            node: AST node to evaluate
            env: Environment for variable bindings (default: global)

        Returns:
            Result of evaluation

        Raises:
            InterpreterError: On evaluation errors
        """
        if env is None:
            env = self.global_env

        # Check execution limit
        self.step_count += 1
        if self.step_count > self.max_steps:
            raise InterpreterError(
                f"Execution exceeded {self.max_steps} steps (possible infinite loop)"
            )

        # Trace this node if in trace mode
        if self.trace_mode:
            self.traced_nodes.add(id(node))
            self.node_id_map[id(node)] = node

        # Evaluate based on node type
        if node.node_type == NodeType.NUMBER:
            return node.value

        elif node.node_type == NodeType.SYMBOL:
            # Variable lookup
            return env.lookup(node.value)

        elif node.node_type == NodeType.DEFINE:
            # (define (func_name param1 param2) body)
            # or (define var value)
            if not node.children:
                raise InterpreterError("DEFINE requires children")

            # First child is either symbol or list (func_name params)
            first_child = node.children[0]

            if first_child.node_type == NodeType.LIST:
                # Function definition: (define (name params...) body)
                func_name = first_child.children[0].value
                params = [child.value for child in first_child.children[1:]]
                body = node.children[1] if len(node.children) > 1 else None

                # Create lambda and bind to name
                func = self._make_lambda(params, body, env)
                env.define(func_name, func)
                return func
            else:
                # Variable definition: (define name value)
                name = first_child.value
                value = self.eval(node.children[1], env) if len(node.children) > 1 else None
                env.define(name, value)
                return value

        elif node.node_type == NodeType.LAMBDA:
            # (lambda (params...) body)
            params_list = node.children[0]
            body = node.children[1]
            params = [child.value for child in params_list.children]
            return self._make_lambda(params, body, env)

        elif node.node_type == NodeType.IF:
            # (if condition then_expr else_expr)
            if len(node.children) != 3:
                raise InterpreterError("IF requires exactly 3 children")

            condition = self.eval(node.children[0], env)
            if condition:
                return self.eval(node.children[1], env)
            else:
                return self.eval(node.children[2], env)

        elif node.node_type == NodeType.LET:
            # (let ((var1 val1) (var2 val2)) body)
            bindings_list = node.children[0]
            body = node.children[1]

            # Create new environment with bindings
            let_env = Environment(parent=env)
            for binding in bindings_list.children:
                var_name = binding.children[0].value
                var_value = self.eval(binding.children[1], env)
                let_env.define(var_name, var_value)

            return self.eval(body, let_env)

        elif node.node_type == NodeType.LIST:
            # Function application: (func arg1 arg2 ...)
            if not node.children:
                return []  # Empty list

            # Evaluate operator
            operator = node.children[0]

            # Check if it's a special form keyword
            if operator.node_type == NodeType.OPERATOR:
                # Built-in operator
                op_func = env.lookup(operator.value)
                args = [self.eval(child, env) for child in node.children[1:]]
                return op_func(*args)
            else:
                # Function application
                func = self.eval(operator, env)
                args = [self.eval(child, env) for child in node.children[1:]]

                if callable(func):
                    return func(*args)
                else:
                    raise TypeMismatchError(f"Cannot call non-function: {func}")

        elif node.node_type == NodeType.OPERATOR:
            # Operator symbol by itself (for lookup)
            return env.lookup(node.value)

        else:
            raise InterpreterError(f"Unknown node type: {node.node_type}")

    def _make_lambda(self, params: list[str], body: ASTNode, closure_env: Environment) -> Callable:
        """Create a lambda function with closure."""

        def lambda_func(*args):
            if len(args) != len(params):
                raise TypeMismatchError(f"Lambda expects {len(params)} arguments, got {len(args)}")

            # Create new environment for function scope
            func_env = Environment(parent=closure_env)
            for param, arg in zip(params, args):
                func_env.define(param, arg)

            return self.eval(body, func_env)

        return lambda_func

    def run_tests(self, program: ASTNode, tests: list[TestCase]) -> list[bool]:
        """Execute program on test inputs, return pass/fail for each.

        Args:
            program: AST of the program (should be a DEFINE)
            tests: List of test cases

        Returns:
            List of booleans indicating test pass/fail
        """
        results = []

        # Execute program to define function
        test_env = Environment(parent=self.global_env)
        try:
            self.step_count = 0
            self.eval(program, test_env)
        except InterpreterError:
            # If program doesn't execute, all tests fail
            for test in tests:
                test.actual_output = None
                test.passing = False
            return [False] * len(tests)

        # Get function name (from DEFINE node)
        if program.node_type == NodeType.DEFINE:
            if program.children[0].node_type == NodeType.LIST:
                func_name = program.children[0].children[0].value
            else:
                func_name = program.children[0].value
        else:
            raise InterpreterError("run_tests expects DEFINE node")

        # Run each test
        for test in tests:
            try:
                self.step_count = 0
                func = test_env.lookup(func_name)
                actual = func(*test.inputs)
                test.actual_output = actual
                test.passing = actual == test.expected_output
                results.append(test.passing)
            except InterpreterError:
                test.actual_output = None
                test.passing = False
                results.append(False)

        return results

    def trace_execution(self, program: ASTNode, test_inputs: list[Any]) -> set[int]:
        """Track which nodes were executed during a test run.

        Args:
            program: AST of the program
            test_inputs: Inputs for a single test

        Returns:
            Set of node IDs (object IDs) that were visited during execution
        """
        # Reset trace state
        self.traced_nodes.clear()
        self.node_id_map.clear()
        self.trace_mode = True

        try:
            # Execute program to define function
            test_env = Environment(parent=self.global_env)
            self.step_count = 0
            self.eval(program, test_env)

            # Get function and execute with test inputs
            if program.node_type == NodeType.DEFINE:
                if program.children[0].node_type == NodeType.LIST:
                    func_name = program.children[0].children[0].value
                else:
                    func_name = program.children[0].value

                func = test_env.lookup(func_name)
                self.step_count = 0
                func(*test_inputs)
        except InterpreterError:
            # Even on error, return traced nodes up to failure point
            pass
        finally:
            self.trace_mode = False

        return self.traced_nodes.copy()


# Helper functions


def eval_multiply(args):
    """Multiply all arguments."""
    result = 1
    for arg in args:
        result *= arg
    return result


def safe_divide(a, b):
    """Safe division with zero check."""
    if b == 0:
        raise DivisionByZeroError("Division by zero")
    return a / b


def reduce_impl(f, lst):
    """Reduce implementation (fold left)."""
    if not lst:
        return None
    result = lst[0]
    for item in lst[1:]:
        result = f(result, item)
    return result
