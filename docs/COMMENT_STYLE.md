# Comment Style Guide

This guide defines how we write comments across the repository.
Short version: write fewer comments, and make the ones that remain useful.

## 1) Core principles

1. **Default to no comments.** Prefer clear names and structure.
2. **Explain why, not what.** The code already shows what it does.
3. **If deletion causes no confusion, delete it.**
4. **Skip decorative comments.** No banners, ASCII boxes, or filler TODOs.
5. **Write comments in English.**
6. **Don’t tie comments to tasks/PRs/issues/callers.** That belongs in git history.

## 2) Allowed comment patterns

### 2.1 One-line module docstring
A one-line module docstring at the top is usually enough.
Multi-line module docstrings should be rare.

### 2.2 Block comments for non-obvious constraints
Use block comments for things a reader cannot infer from code:
platform quirks, library bugs, version-specific behavior, import-order traps, etc.

When you write one:
- Put the trigger condition first.
- Describe the symptom concretely (message/code/behavior).
- Explain why the fix must be in this exact spot.
- Note exclusions and trade-offs when relevant.

### 2.3 Inline notes for magic values
If a literal has hidden meaning, add a short inline note.
Keep it one line. Longer explanation belongs in a nearby block comment.

### 2.4 Linter directives (optional reason)
Keep directives like `noqa`, `type: ignore`, or something important.
If the reason is obvious, no note needed. If not, add a short parenthetical.

## 3) Exceptions where normal commenting is encouraged

The strict rules are intentionally relaxed in these cases.
Moderate explanatory comments are welcome.

### 3.1 Mathematical formulas
If code implements a non-trivial formula, document the formula itself.
- Use a recognizable textbook form.
- Name symbols and units.
- Cite a source only when needed.

### 3.2 Parameters of imported mathematical/external routines
For fitters/models/detectors/macros where parameter roles are not obvious,
explain each parameter briefly (meaning + unit).
Also note important constraints (`bounds`, `maxfev`, weights, etc.) when they affect outcomes.

### 3.3 Multi-step calculations
For conceptually layered calculations, add short step headers.
- No comment for obvious steps, but add one when heavy calculation or non-obvious math is involved.
- Include numeric thresholds/windows.
- Use known algorithm names when applicable (e.g., Otsu, RANSAC).

### 3.4 ui layout sections
For builder functions with multiple UI sections, two short comment forms are
allowed:

- **Section header** — `# <position> <orientation> layout: <c1>, <c2>, <c3>.`
  `<position>` describes where *this section itself* sits inside its parent
  layout — not where its children sit. Allowed forms:
  - one vertical token: `VTop` / `VCenter` / `VBot`
  - one horizontal token: `HLeft` / `HCenter` / `HRight`
  - one of each, combined: e.g. `VTop HLeft`
  The same token may repeat across sections in different parents.
  `<orientation>` is `grid`, `vertical`, or `horizontal` and describes how
  this section lays out its own children.
- **Row / column label** — `# Row N: A | B | C.` (or `# Column N: ...`),
  listing components left-to-right or top-to-bottom. Drop the `|` when the
  row has a single component.

Name what each widget is *for*, not what the call literally constructs
(restating code is still forbidden — §4). Do not reference identifiers that
no longer exist in the code.

## 4) Forbidden comments

Remove these when you see them:
- Comments that simply restate code.
- References to issues, tasks, callers, or PR context.
- Decorative separators and banners.
- “Removed” markers next to dead code.
- Long what-docstrings for args/returns already visible in the signature.
- Mixed-language comments or emoji.

## 5) Refactoring checklist

1. Delete obvious noise (restatements, decorations, issue/caller references).
2. Normalize language to English (or delete low-value comments).
3. Keep only comments that prevent future confusion.
4. Rename first, comment second.
5. Add inline notes for true magic values.
6. Add rationale for unclear linter suppressions.
7. Preserve useful comments for formulas, external parameters, and multi-step math.

## 6) Quick do/don’t

| Situation | Do | Don’t |
|---|---|---|
| Explain function behavior | Use clear names/signatures | Add a what-docstring |
| Non-obvious import order | Explain why in a block comment | Leave future readers guessing |
| Platform-specific branch | Comment only when needed | Add empty labels |
| Magic constant | One-line meaning + unit | Spread explanation across many lines |
| Temporary code | Finish or delete it | Leave “TODO: later” clutter |
| Dead code | Delete it | Comment it out |
| Formula implementation | Show formula/symbols/units | Hide behind vague wording |
| External routine params | Brief per-parameter notes | Force readers to infer intent |
| Multi-step algorithm | Add short step signposts | Leave a wall of opaque code |
