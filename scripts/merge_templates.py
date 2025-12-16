#!/usr/bin/env python3
"""Quick script to merge new templates into synthetic_gen.py"""

# Read the scratch templates
with open("/home/me/git/prose/src/data/new_templates_scratch.py", "r") as f:
    templates_code = f.read()

# Read the original synthetic_gen.py
with open("/home/me/git/prose/src/data/synthetic_gen.py", "r") as f:
    original_code = f.read()

# Find where to insert (before "class SyntheticGenerator:")
insert_marker = "class SyntheticGenerator:"
insert_index = original_code.index(insert_marker)

# Extract just the class definitions (skip imports)
templates_start = templates_code.index("class NestedLetTemplate")
templates_to_insert = templates_code[templates_start:] + "\n\n"

# Insert the new templates
new_code = original_code[:insert_index] + templates_to_insert + original_code[insert_index:]

# Also update the templates list in __init__
# Find the templates list
init_start = new_code.index("self.templates = [")
init_end = new_code.index("]", init_start) + 1

# Replace with updated list
new_templates_list = """self.templates = [
            ArithmeticTemplate("arithmetic", "Basic arithmetic expressions with varying structure"),
            RecursionTemplate("recursion", "Recursive function definitions"),
            LambdaTemplate("lambda", "Higher-order functions with lambdas"),
            LetBindingTemplate("let", "Variable scoping with varying bindings"),
            NestedLetTemplate("nested_let", "Nested let bindings"),
            ConditionalChainTemplate("conditional", "Nested conditional expressions"),
            MixedArithmeticTemplate("mixed_arith", "Mixed arithmetic with asymmetric structures"),
        ]"""

new_code = new_code[:init_start] + new_templates_list + new_code[init_end:]

# Write back
with open("/home/me/git/prose/src/data/synthetic_gen.py", "w") as f:
    f.write(new_code)

print("Templates integrated successfully!")
print(
    f"Added 3 new templates: NestedLetTemplate, ConditionalChainTemplate, MixedArithmeticTemplate"
)
