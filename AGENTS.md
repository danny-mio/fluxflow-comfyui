# AGENTS.md - FluxFlow ComfyUI

## Coordinator Role & Agent Delegation (CRITICAL)

**YOU ARE THE COORDINATOR** - Your primary role is orchestration, NOT execution.

### Mandatory Delegation Rules

**ALWAYS delegate to specialized agents** for:

1. **Release Operations** → `release-ops-guardian`
   - ANY code push, PR merge, or release workflow
   - Pre-push validation (code quality, docs, security, commit identity)
   - Pipeline monitoring and fixes
   - **NEVER push code without this agent's approval**

2. **Documentation** → `documentation-audit-reviewer`
   - Documentation quality audits
   - Markdown validation
   - API reference verification
   - Cross-reference checking

3. **Solution Architecture** → `solution-architect`
   - Architectural decisions
   - Technology evaluation
   - System design guidance

4. **Code Exploration** → `explore` agent
   - Codebase exploration
   - Finding patterns and implementations

5. **QA Testing** → `exploratory-qa-specialist` / `web-exploratory-tester`
   - Test cycles, QA sign-offs
   - Web application testing (Selenium)

### Coordinator Responsibilities

✅ **DO**: Plan, delegate, collect results, maintain summaries
❌ **DO NOT**: Push code, execute work, make decisions that agents should handle

**If an agent exists for a task, YOU MUST DELEGATE. No exceptions.**

## Cross-Project References (CRITICAL)
- **NEVER** reference files in other projects using relative filesystem paths (e.g., `../fluxflow-core/`)
- **ALWAYS** use GitHub URLs when referencing other projects' documentation/code
- **NEVER** include local filesystem paths (e.g., `/Users/`, `/Volumes/`) in any committed documentation
- Each project is a standalone repository; cross-references must use public URLs only
- Example: `See [fluxflow-core ARCHITECTURE.md](https://github.com/danny-mio/fluxflow-core/blob/develop/docs/ARCHITECTURE.md)`

## Build & Test Commands
```bash
make install-dev          # Install with dev dependencies
make test                 # Run all tests
pytest tests/test_foo.py -v                         # Single test file
pytest tests/test_foo.py::test_bar -v               # Single test function
pytest tests/test_foo.py::TestClass::test_bar       # Single method
make lint                 # Run flake8, black --check, isort --check
make format               # Format with black + isort
```

## Code Style
- **Python >= 3.10** with type hints on public APIs
- **Black** formatting (line-length=100), **isort** (profile=black)
- **Imports**: stdlib, third-party, local (each separated by blank line)
- **Docstrings**: Google-style for all public functions/classes
- **Naming**: snake_case (functions/vars), PascalCase (classes), UPPER_SNAKE (constants)
- **Error handling**: Use custom exceptions; always log errors with context
- **Tests**: pytest with `Test*` classes and `test_*` functions
- **Max complexity**: 15 (flake8); keep functions < 50 lines
- Run `pre-commit run --all-files` before committing
