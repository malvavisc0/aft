# Model fine-tuning and quantization

## Communication
You are talking to seniors. Be terse — no preamble, no recap, no
hand-holding, no "here's what I did" summaries. State results, not
intentions. Skip explanations unless explicitly asked. Never narrate
tool use. Answer the question; move on.

## Project
aft — Aria Finetuner. Standalone fine-tuning and quantization pipeline 
for instruction-following models.

## Stack
- python 3.12, uv, pyproject.toml (setuptools build)
- Type checker: **basedpyright** (`standard` mode)

## Commands
```bash
uv sync                          # install deps
uv run ruff check --fix .        # lint
uv run ruff format .             # format
uv run basedpyright              # type check
uvx radon cc src/aft -nc         # complexity (reject >= C)
uvx vulture src/aft              # dead code
uv run pytest                    # tests (co-located per module under tests/)
```
Run lint + basedpyright after every change. Reject any radon grade >= C. Fix all vulture findings.

## Priority
Quality over quantity. Clean, simple, readable, maintainable code. Nothing else.
Never trade correctness or clarity for speed or fewer lines of diff.

Less is more. Every line of code must earn its place; if a line can be
deleted without losing behavior, delete it. Do not add speculative code,
defensive branches, or abstractions for futures that may never come — you
maintain what you write, so every line is a liability. Write the fewest
lines that fully solve the problem, and nothing more.

## Rules
- DRY, KISS
- No unnecessary complexity
- No defensive fallbacks / over-engineering
- Explicit > implicit
- Type hints always
- PEP 8 (ruff enforces, line-length 88)
- Meaningful names, small functions, single responsibility
- Fail fast and clearly
- Prefer standard library
- No clever tricks
- Docstrings for public APIs
- No unnecessary inline comments
- Use uv (never pip / system python)
- No mutable module-level globals
- No dead code
- No `type: ignore` comments

## Depth Before Breadth

Read before you edit. Code has memory — a change in module X may silently
break module Y via shared state, event channels, or API contracts.

- Read the surrounding system (callers, consumers, shared state, imports,
  API contracts) before choosing an edit location. Scan broad, then narrow.
- Name the blast radius before editing — list every file and flow the change
  touches, including indirect ones. If you can't, keep reading.
- The smallest diff is not the goal. The correct edit may touch several
  files. Touching fewer files because you searched less is a failure.
- The source is ground truth. Verify behavior against the code, never
  against your summary of it.
- Trivial change? Keep it trivial — no caveats, no over-engineering.

## File size
- No file may exceed **600 lines**. Hard cap **660 lines** (10% margin).
- Approaching the limit? Split first — by responsibility, extract helpers,
  or carve out a class. Never merge concerns to stay under it.
- `find src/aft -name '*.py' -exec wc -l {} + | sort -rn | head` before
  extending a file.
- Prefer many small, well-named files over one large file.

## Workflow

Use `uv`, never `python` directly.

### Verify before you code
Assumptions are the most common source of bugs. Before writing anything:
- **Project code**: read the relevant file(s) first. Never edit from memory.
- **Dependencies**: don't invent APIs. Confirm against installed source
  (`uv run python -c "import <pkg>; print(<pkg>.__file__)"`), stubs, or a
  3-line reproduction. Adapt to the real API — no fallbacks.
- **Bugs**: write the smallest reproduction first. Confirm the failure
  before fixing.

### Checklist
1. Read the relevant source; check file size; split before extending.
2. Map the change — name touched files and dependents. Can't? Read more.
3. Verify any third-party API against installed source/stubs.
4. Reproduce bugs minimally before fixing.
5. Run linters/type check before and after: `uv run ruff check .`,
   `uv run basedpyright`, `uvx vulture src/aft`.
6. Review `git diff` before finalizing.
7. Don't commit unless asked.

## Style
Simplest correct solution. Delete anything unneeded.

Tests: pin behavior that can regress, not implementation details or
trivialities. A test that always passes or restates the code is dead
weight. A few real-behavior tests beat a thick suite that pins nothing.
