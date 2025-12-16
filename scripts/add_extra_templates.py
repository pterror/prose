#!/usr/bin/env python3
"""Add extra templates to synthetic_gen.py"""

# Read synthetic_gen.py
with open("/home/me/git/prose/src/data/synthetic_gen.py", "r") as f:
    content = f.read()

# Find the templates list initialization (look for where simple templates are added)
marker = "from src.data.simple_templates import"
if marker in content:
    # Find the extend call
    extend_start = content.index("self.templates.extend([")
    extend_end = content.index("])", extend_start) + 2

    # Replace with updated extend that includes extra templates
    new_extend = """from src.data.simple_templates import SimpleArithTemplate, DefineSimpleTemplate, SimpleIfTemplate
        from src.data.extra_templates import (
            SingleSymbolTemplate, SingleNumberTemplate, ComparisonOnlyTemplate,
            NestedComparisonTemplate, ListLiteralTemplate, MultiDefineTemplate,
            IfWithArithBranchesTemplate, LambdaWithLetTemplate
        )
        self.templates.extend([
            SimpleArithTemplate("simple_arith", "Simple arithmetic expressions"),
            DefineSimpleTemplate("simple_define", "Simple variable definitions"),
            SimpleIfTemplate("simple_if", "Simple conditional expressions"),
            SingleSymbolTemplate("single_sym", "Single symbol"),
            SingleNumberTemplate("single_num", "Single number"),
            ComparisonOnlyTemplate("comparison", "Comparison expressions"),
            NestedComparisonTemplate("nested_comp", "Nested comparisons"),
            ListLiteralTemplate("list_lit", "List literals"),
            MultiDefineTemplate("multi_define", "Multiple definitions"),
            IfWithArithBranchesTemplate("if_arith", "IF with arithmetic branches"),
            LambdaWithLetTemplate("lambda_let", "Lambda with LET"),
        ])"""

    # Find the start of the import line
    import_start = content.index(marker)
    # Find the end of the extend block (after the ])

    content = content[:import_start] + new_extend + content[extend_end:]

    # Write back
    with open("/home/me/git/prose/src/data/synthetic_gen.py", "w") as f:
        f.write(content)

    print("Added 8 extra templates to synthetic_gen.py")
else:
    print("Marker not found - simple_templates import missing")
