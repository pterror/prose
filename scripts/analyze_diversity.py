#!/usr/bin/env python3
"""Analyze theoretical diversity of current template system."""

# Template diversity analysis:

# 1. ArithmeticTemplate: (op1 (op2 a b) (op3 c d))
#    - 4 operators for each of 3 positions
#    - Unique programs: 4^3 = 64

# 2. RecursionTemplate: (define (func_name n) (if (= n base_val) 1 (* n (func (- n 1)))))
#    - 3 function names × 3 base values (0, 1, 2)
#    - Unique programs: 3 × 3 = 9

# 3. LambdaTemplate: (hof (lambda (x) (op x const)) list)
#    - 3 HOF (map, filter, reduce) × 3 operators (+, *, -) × 10 constants (1-10)
#    - Unique programs: 3 × 3 × 10 = 90

# 4. LetBindingTemplate: (let ((x val1) (y val2)) (op x y))
#    - 3 operators (+, -, *) × 10 val1 × 10 val2
#    - Unique programs: 3 × 10 × 10 = 300

# Total unique programs across all templates:
total_unique = 64 + 9 + 90 + 300

print("=" * 70)
print("Theoretical Program Diversity Analysis")
print("=" * 70)
print()
print("Current Templates:")
print(f"  1. ArithmeticTemplate:   64 unique programs (4³ operator combinations)")
print(f"  2. RecursionTemplate:     9 unique programs (3 functions × 3 base values)")
print(f"  3. LambdaTemplate:       90 unique programs (3 HOF × 3 ops × 10 constants)")
print(f"  4. LetBindingTemplate:  300 unique programs (3 ops × 10² value pairs)")
print()
print(f"TOTAL UNIQUE PROGRAMS: {total_unique}")
print()
print("=" * 70)
print("Implications for Dataset Size:")
print("=" * 70)
print()
print(f"With {total_unique} unique programs:")
print()
print("  • 100K training samples = ~217x duplication per program")
print("  • 10K validation samples = ~22x duplication per program")
print("  • 10K test samples = ~22x duplication per program")
print()
print("This level of duplication is EXCESSIVE and unlikely to improve")
print("model performance significantly.")
print()
print("=" * 70)
print("Recommendations:")
print("=" * 70)
print()
print("Option 1: START SMALL (Recommended for Phase 1)")
print("  • 2K training samples (~4x duplication)")
print("  • 500 validation samples (1x coverage)")
print("  • 500 test samples (1x coverage)")
print("  • Storage: ~150MB")
print("  • Training time: ~30 minutes for 50 epochs")
print()
print("Option 2: EXPAND TEMPLATES FIRST")
print("  Add more template variety to justify larger datasets:")
print("  • Nested arithmetic (3+ depth)")
print("  • Multiple function definitions")
print("  • Conditional expressions with varied predicates")
print("  • List operations (cons, car, cdr)")
print("  • Mixed let+lambda patterns")
print("  → Could reach 10K-50K unique programs")
print()
print("Option 3: HYBRID APPROACH")
print("  • Start with 2K/500/500 split")
print("  • Train initial model")
print("  • Add more templates based on what model struggles with")
print("  • Scale up dataset as template variety increases")
print()
print("=" * 70)
