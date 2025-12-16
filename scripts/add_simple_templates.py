#!/usr/bin/env python3
"""Add simple templates to synthetic_gen.py"""

# Read synthetic_gen.py
with open("/home/me/git/prose/src/data/synthetic_gen.py", "r") as f:
    content = f.read()

# Find the templates list initialization
templates_start = content.index("self.templates = [")
templates_end = content.index("]", templates_start)

# Add the new templates to the end of the list
new_templates = """self.templates = [
            ArithmeticTemplate("arithmetic", "Basic arithmetic expressions with varying structure"),
            RecursionTemplate("recursion", "Recursive function definitions"),
            LambdaTemplate("lambda", "Higher-order functions with lambdas"),
            LetBindingTemplate("let", "Variable scoping with varying bindings"),
            NestedLetTemplate("nested_let", "Nested let bindings"),
            ConditionalChainTemplate("conditional", "Nested conditional expressions"),
            MixedArithmeticTemplate("mixed_arith", "Mixed arithmetic with asymmetric structures"),
        ]
        
        # Import and add simple templates for diversity
        from src.data.simple_templates import SimpleArithTemplate, DefineSimpleTemplate, SimpleIfTemplate
        self.templates.extend([
            SimpleArithTemplate("simple_arith", "Simple arithmetic expressions"),
            DefineSimpleTemplate("simple_define", "Simple variable definitions"),
            SimpleIfTemplate("simple_if", "Simple conditional expressions"),
        ])"""

# Replace
content = content[:templates_start] + new_templates + content[templates_end + 1 :]

# Write back
with open("/home/me/git/prose/src/data/synthetic_gen.py", "w") as f:
    f.write(content)

print("Added 3 simple templates to synthetic_gen.py")
