# Naming Guide

Good names do most of the explaining for us.
When you rename something, make sure a reader can understand the function’s
intent, inputs, outputs, and approach just from the call site.

## 1) Naming rules

### 1.1 Prefer full words
Use full words unless an abbreviation is truly standard in this domain.

### 1.2 Use a clear verb + object
The verb should describe the real action, and the object should name the target.
Try to avoid vague verbs like `process` or `handle` when a sharper one exists.

### 1.3 Reflect the output
Name what the function returns (or computes), not just a generic action.
If the name and return value disagree, treat it as a bug.

### 1.4 Include the method when it matters
If multiple algorithms could apply, make the chosen method explicit in the name.

### 1.5 Keep domain context visible
If a helper is tied to a specific pipeline or tool, say that in the name.
That context helps readers place it correctly.

### 1.6 Be consistent across modules
The same concept should use the same term everywhere.
Two helpers that produce the same thing should not have unrelated names.

## 2) What must stay unchanged during renaming

- **Signature**: keep parameters, defaults, and return types the same.
- **Implementation**: no logic edits, inlining, or extraction.
- **Behavior**: no user-visible changes.
- **Imports/exports**: update all call sites and `__init__.py` exports.
  Do not leave temporary alias shims behind.
