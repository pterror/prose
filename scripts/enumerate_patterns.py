#!/usr/bin/env python3
"""Generate a mega-varied ArithmeticTemplate with 20+ structural patterns."""

# All the pattern generators
patterns = []

# Pattern 1-4: Simple n-ary (2, 3, 4, 5 operands)
for n in [2, 3, 4, 5]:
    patterns.append(f"simple_{n}ary")

# Pattern 5-8: Nested left at different depths
for depth in [2, 3, 4]:
    patterns.append(f"nested_left_d{depth}")

# Pattern 9-11: Nested right at different depths
for depth in [2, 3]:
    patterns.append(f"nested_right_d{depth}")

# Pattern 12-14: Balanced binary trees
for depth in [2, 3]:
    patterns.append(f"balanced_d{depth}")

# Pattern 15-17: Asymmetric (3-operand with one nested)
patterns.extend(["asym_left", "asym_right", "asym_mid"])

# Pattern 18-20: Chain (all left-associated or all right-associated)
patterns.extend(["chain_left_3", "chain_left_4", "chain_right_3"])

# Pattern 21-23: Mixed depths
patterns.extend(["mixed_2_3", "mixed_3_4", "wide_shallow"])

print(f"# Generated {len(patterns)} structural patterns for ArithmeticTemplate")
print(f"# Patterns: {patterns}")
print(f"\nTotal unique structures: {len(patterns)}")
print(f"\nNow I need to implement generation code for each pattern...")

# Let me create a template that randomly picks from these 23 patterns
