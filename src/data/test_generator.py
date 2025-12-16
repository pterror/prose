"""Test case generation utilities for Mini-Lisp programs."""

import random
from typing import Any, Callable

from src.data.asg_builder import ASTNode, NodeType
from src.runtime.interpreter import MiniLispInterpreter, TestCase, InterpreterError


class TestGenerator:
    """Generate test cases for Mini-Lisp programs."""

    def __init__(self, interpreter: MiniLispInterpreter | None = None):
        """Initialize test generator.

        Args:
            interpreter: Interpreter instance (creates new one if None)
        """
        self.interpreter = interpreter or MiniLispInterpreter()

    def generate_tests_for_function(
        self,
        program: ASTNode,
        func_name: str,
        num_params: int,
        rng: random.Random,
        num_tests: int = 3,
        param_range: tuple[int, int] = (-10, 10),
    ) -> list[TestCase]:
        """Generate tests by executing program with random inputs.

        Args:
            program: The DEFINE AST node
            func_name: Function name to test
            num_params: Number of parameters the function takes
            rng: Random number generator for determinism
            num_tests: Number of test cases to generate
            param_range: Range for random parameter values

        Returns:
            List of TestCase instances
        """
        tests = []

        # Execute program to define function
        from src.runtime.interpreter import Environment

        env = Environment(parent=self.interpreter.global_env)

        try:
            self.interpreter.step_count = 0
            self.interpreter.eval(program, env)
        except InterpreterError:
            # If program doesn't compile, return empty tests
            return []

        # Get the function
        try:
            func = env.lookup(func_name)
        except:
            return []

        # Generate test cases
        for _ in range(num_tests):
            # Generate random inputs
            inputs = [rng.randint(*param_range) for _ in range(num_params)]

            # Execute to get expected output
            try:
                self.interpreter.step_count = 0
                expected_output = func(*inputs)

                # Create test case
                test = TestCase(
                    inputs=inputs,
                    expected_output=expected_output,
                )
                tests.append(test)
            except InterpreterError:
                # Skip tests that cause errors
                continue
            except (RecursionError, ValueError, TypeError):
                # Skip tests that cause Python errors
                continue

        return tests

    def generate_manual_tests(self, test_specs: list[dict[str, Any]]) -> list[TestCase]:
        """Generate tests from manual specifications.

        Args:
            test_specs: List of dicts with 'inputs' and 'expected_output' keys

        Returns:
            List of TestCase instances
        """
        return [
            TestCase(inputs=spec["inputs"], expected_output=spec["expected_output"])
            for spec in test_specs
        ]


def generate_arithmetic_tests(
    program: ASTNode, operation: str, rng: random.Random, num_tests: int = 3
) -> list[TestCase]:
    """Generate tests for simple arithmetic expressions.

    This is for non-function expressions like (+ a b).
    We need to evaluate with specific variable bindings.
    """
    # For simple expressions, we generate tests with different variable values
    # This is a simplified implementation - in practice, you'd extract variables
    # from the AST and test different combinations

    # Extract variables from program
    variables = _extract_variables(program)

    if not variables:
        return []

    tests = []
    interpreter = MiniLispInterpreter()

    for _ in range(num_tests):
        # Generate random values for variables
        from src.runtime.interpreter import Environment

        env = Environment(parent=interpreter.global_env)

        inputs = {}
        for var in variables:
            value = rng.randint(-10, 10)
            inputs[var] = value
            env.define(var, value)

        # Evaluate expression
        try:
            interpreter.step_count = 0
            result = interpreter.eval(program, env)

            # Create test case (for expressions, inputs are dict of var values)
            test = TestCase(inputs=list(inputs.values()), expected_output=result)
            tests.append(test)
        except InterpreterError:
            continue

    return tests


def _extract_variables(node: ASTNode, variables: set[str] | None = None) -> set[str]:
    """Extract all variable names (SYMBOL nodes) from AST."""
    if variables is None:
        variables = set()

    if node.node_type == NodeType.SYMBOL and node.value:
        variables.add(node.value)

    for child in node.children or []:
        _extract_variables(child, variables)

    return variables
