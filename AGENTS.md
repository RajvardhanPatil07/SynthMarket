# SynthMarket Design System Rules

## Product Direction

- SynthMarket is a research workspace, not a marketing page and not an exchange clone.
- UI should feel minimal, precise, and functional: calm surfaces, dense information, clear controls, restrained color.
- Avoid decorative gradients, glowing effects, novelty styling, oversized type, and unnecessary visual noise.
- Prioritize fast scanning for quant workflows: configuration, run status, generated data, evaluation, backtest, exports.

## Frontend Architecture

- The web UI is plain HTML, CSS, and JavaScript in `synthmarket/static/`.
- Do not add React, Node build steps, CSS frameworks, icon packages, or external frontend dependencies for this dashboard.
- Keep behavior in `synthmarket/static/app.js`; keep styling in `synthmarket/static/styles.css`.
- Preserve the existing API contracts in `synthmarket/web.py` unless a backend change is explicitly needed.

## Design Tokens

- Use CSS custom properties in `:root` and `:root[data-theme="dark"]`.
- IMPORTANT: Do not hardcode colors outside token definitions except transparent/white/black utility values.
- Use neutral surfaces first, then a single restrained accent for primary actions.
- Status colors:
  - Positive/success: green token.
  - Warning/running: amber token.
  - Failed/risk: red token.
- Charts must read from `--chart-*` tokens so dark/light mode remains legible.

## Layout Rules

- Use compact 8px radius or less.
- Use panels only for actual tool surfaces, repeated metric cards, tables, and output containers.
- Keep form controls dense but readable; labels should be short and uppercase.
- Avoid cards inside cards unless the inner item is a repeated library/template item.
- Side rail should contain run progress and outputs only.
- Navigation should be functional tabs with clear active state, not decorative pills.

## Component Conventions

- HTML must remain accessible:
  - Buttons for actions.
  - Labels associated with form controls.
  - `aria-label` for icon/text toggles when the visible label is short.
- Use semantic sections and tables for result data.
- Any new interactive control must have a disabled/error/empty state when applicable.
- Theme state must persist in `localStorage` and redraw canvases after switching.

## Figma-To-Code Workflow

- If Figma MCP tools are available, first fetch design context and screenshot for exact nodes before implementing.
- Treat Figma output as design reference, not final code style.
- Translate all Figma values into this project’s CSS tokens and plain HTML/CSS/JS conventions.
- Validate in the browser after every meaningful frontend change.

## Testing And Verification

- Run `./.venv/bin/ruff check . --no-cache` after Python changes.
- Run `./.venv/bin/python -m pytest -p no:cacheprovider` after behavior/API changes.
- Run `./.venv/bin/python -m compileall synthmarket tests examples` before final handoff.
- For UI changes, reload `http://127.0.0.1:8765/`, check browser console errors, and verify dark/light theme and at least one result-heavy tab.
