# AI Hermes Self-Evolve Skill

Portable Codex skill for auditing and maintaining AI Hermes routing memory.

## What It Does

- Checks whether changed files are covered by AI Hermes routing memory.
- Validates `repo_map`, `task_routes`, `pitfalls`, and self-evolution governance files.
- Guides minimal routing-memory updates for stale facts, hidden contracts, and pre-submit checks.
- Bundles fallback scripts so the skill can run even when a target repository does not include project-local AI Hermes skill scripts.

## Contents

- `SKILL.md` - workflow and operating rules.
- `agents/openai.yaml` - Codex skill metadata.
- `scripts/evolve_ai_routing.py` - routing coverage checker.
- `scripts/validate_ai_routing.py` - routing schema/path/reference validator.
- `scripts/route_task.py` - route resolver smoke checker for selected route ids.

## Target Repository Requirements

Use this skill in a repository that follows the AI Hermes routing layout:

- `AGENTS.md`
- `docs/repo_map.json`
- `docs/task_routes.json`
- `docs/pitfalls.json`
- `docs/ai_routing_evolution_policy.json` for full validation

The bundled scripts use only Python standard library modules. Python 3.10+ is recommended.

## Usage

Prefer project-local skill scripts when they exist:

```bash
python3 skills/ai-hermes-self-evolve/scripts/evolve_ai_routing.py --changed-file <path> --json
python3 skills/ai-hermes-self-evolve/scripts/validate_ai_routing.py
```

If the target repo does not have those project-local scripts, run the bundled scripts from this skill directory and pass the target repository explicitly:

```bash
SKILL_DIR=<directory-containing-this-README>
python3 "$SKILL_DIR/scripts/evolve_ai_routing.py" --project-root "$PWD" --changed-file <path> --json
python3 "$SKILL_DIR/scripts/validate_ai_routing.py" --project-root "$PWD"
python3 "$SKILL_DIR/scripts/route_task.py" --project-root "$PWD" --route-id <route-id> --mode context
```

For routing-only checks:

```bash
python3 "$SKILL_DIR/scripts/evolve_ai_routing.py" --project-root "$PWD" --routing-only --changed-file <routing-file> --json
```

## Portability Notes

- The skill does not require host-specific absolute paths.
- The bundled scripts do not modify files; they only report coverage or validation results.
- The skill is portable across repositories that use compatible AI Hermes routing JSON schemas.
- Repositories with custom docset locations or without `docs/ai_routing_evolution_policy.json` may need local adaptation before full validation passes.
