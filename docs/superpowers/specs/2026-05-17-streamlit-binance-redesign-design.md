# Streamlit Binance-Inspired Redesign — Design Spec

**Date:** 2026-05-17
**Status:** Draft — awaiting user approval before plan-writing
**Scope:** `streamlit_page_en.py`, `streamlit_page_ko.py`, supporting modules; targets the user-reported "Save Changes doesn't update JSON view" bug (symptom 1) and a Binance-inspired visual overhaul of the strategy editor.

---

## 1. Goals

1. **Fix the save bug.** Editing an existing strategy and clicking *Save Changes* must update `st.session_state["strategies"]` and the visible JSON view. Uploaded JSONs must populate form widgets correctly.
2. **Reskin to a Binance-inspired dark theme.** Apply the supplied design system (Binance Yellow accent on near-black canvas, trading-up/down semantics, Inter + JetBrains Mono substitutes for BinanceNova / BinancePlex) within Streamlit's component constraints.
3. **Restructure the layout** from form-on-left / raw-JSON-on-right into form-on-left / strategy-summary-cards-on-right, with raw JSON kept behind a "View Raw JSON" expander.
4. **Refactor shared structure** into a `streamlit_ui/` package so EN and KO pages stop duplicating ~300 lines of identical form logic.
5. **No engine changes.** This spec touches the Streamlit layer only. The pybacktest engine on `main` and the upcoming engine-core-upgrade work are untouched.

## 2. Non-Goals

- Pixel-perfect Binance reproduction. Streamlit's widget internals limit what CSS can override.
- Live market data / trading-platform features beyond what the backtest engine already supports.
- Visual regression testing infrastructure (deferred — manual smoke tests suffice for this round).
- Refactoring the pybacktest engine, models, or strategy schema.
- Mobile-first design. Layout must not break at narrow widths, but desktop is the primary target.

## 3. Architecture

### 3.1 Module Layout

```
streamlit_page.py            # entry router, unchanged
streamlit_page_en.py         # ~90 lines, thin localization wrapper
streamlit_page_ko.py         # ~90 lines, thin localization wrapper

streamlit_ui/                # new package
  __init__.py
  theme.py                   # CSS string + inject_global_styles()
  components.py              # render_strategy_card, render_metric, render_section_header,
                             # render_primary_button, render_buy_sell_selector
  forms.py                   # input_strategy_details, _collect_form_dict,
                             # _save_strategy_callback, _apply_uploaded_json
  i18n.py                    # LABELS dict + T() lookup

.streamlit/
  config.toml                # base dark theme + Binance Yellow primaryColor

tests/
  test_streamlit_forms.py    # new — pure-Python tests for form state machine
  test_streamlit_i18n.py     # new — both languages have all required keys
```

### 3.2 Module Responsibilities

**`theme.py`** — Single responsibility: visual styling.
- Exports `inject_global_styles() -> None` that writes one `st.markdown(unsafe_allow_html=True)` block per page render. Idempotent within a session (uses an `st.session_state` flag to avoid duplicate injection on rerun, though duplicate `<style>` tags are harmless).
- Exports color, spacing, and radius tokens as Python constants for components that need them inline.

**`components.py`** — Single responsibility: visual presentation of data.
- `render_strategy_card(ticker: str, strategy: dict, lang: str) -> None` — emits a strategy summary card via `st.markdown(html, unsafe_allow_html=True)`, then renders the Edit / Delete `st.button` widgets below.
- `render_metric(label: str, value: str, kind: Literal["neutral", "up", "down"]) -> None` — renders a metric callout card.
- `render_section_header(text: str) -> None` — renders an h2 with the yellow left rule.
- `render_buy_sell_selector(state_key: str, lang: str) -> Literal["buy", "sell"]` — renders the two-pill selector and returns the active selection.

**`forms.py`** — Single responsibility: form state machine + save logic.
- `input_strategy_details(state_key_prefix, default_ticker, saved_data, allowed_qty_types, lang)` — the shared form helper (currently duplicated between EN and KO). Seeds widget session_state from `saved_data` when needed, renders widgets, returns nothing (the save callback reads widget state directly).
- `_collect_form_dict(prefix: str, allowed_qty: tuple[str, ...], state: dict | None = None) -> dict` — pure function that reads widget session_state by prefix and returns a dict matching `StrategyWrapper`'s buy/sell schema. The `state` parameter defaults to `st.session_state` and is overridable in tests.
- `_save_strategy_callback(lang: str, main_ticker: str) -> None` — Streamlit `on_click` callback invoked by the Save button. Reads widget state, builds a fresh strategy dict, writes it to `st.session_state["strategies"][main_ticker]`, fires `st.toast`.
- `_apply_uploaded_json(loaded_data: dict) -> None` — clears stale widget session_state keys, then writes `st.session_state["strategies"] = loaded_data`. Bound as the `on_click` handler of the "Apply Data" sidebar button.

**`i18n.py`** — Single responsibility: localized strings.
- `LABELS: dict[Literal["en", "ko"], dict[str, str]]` — maps every UI string the page files reference.
- `T(key: str, lang: str) -> str` — lookup helper. Raises `KeyError` at call time if a key is missing — caught by `test_streamlit_i18n.py`.

### 3.3 Why this layout

- Each module answers one of "what does it look like" (`theme`, `components`), "how does it behave" (`forms`), or "what does it say" (`i18n`). This matches the design-for-isolation principle the brainstorming skill calls out: each unit has a clear interface, can be understood independently, and is testable in isolation.
- The bug fix lives entirely in `forms.py`. Fixing it once fixes both languages and prevents future drift.
- A future third language is a strings-only change in `i18n.py`.

## 4. Visual System

### 4.1 Theme Config

`.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#FCD535"
backgroundColor = "#0b0e11"
secondaryBackgroundColor = "#1e2329"
textColor = "#eaecef"
font = "sans serif"
```

### 4.2 Token Mapping

| Binance design system token | Streamlit usage |
|---|---|
| `colors.primary` (#FCD535) | All primary CTAs, section header accent rule, active tab indicators on the "Buy" semantic |
| `colors.primary-active` (#f0b90b) | Primary button hover/press state |
| `colors.canvas-dark` (#0b0e11) | Page background (via theme config) |
| `colors.surface-card-dark` (#1e2329) | Strategy summary cards, metric callouts, input field backgrounds |
| `colors.surface-elevated-dark` (#2b3139) | Nested surfaces (form section group backgrounds, expander interior) |
| `colors.hairline-on-dark` (#2b3139) | All 1px borders on dark canvas |
| `colors.body` (#eaecef) | Default text |
| `colors.muted` (#707a8a) | Captions, helper text, field labels |
| `colors.trading-up` (#0ecb81) | Buy tab indicator + border, profit-rate metric when > 1.0 |
| `colors.trading-down` (#f6465d) | Sell tab indicator + border, profit-rate metric when < 1.0 |
| `colors.on-primary` (#181a20) | Black text on yellow primary buttons |
| `typography.hero-display` (Inter 700, 48px) | Page title "📈 Automated Trading Strategy" |
| `typography.display-md` (Inter 600, 40px) | Section headers ("📝 Edit Strategy", "Backtest") |
| `typography.title-lg` (Inter 600, 24px) | Sub-section headers, card ticker labels |
| `typography.number-display` (JetBrains Mono 700, 40px) | Final Profit Rate, Initial Capital metric values |
| `typography.number-md` (JetBrains Mono 500, 16px) | Quantity / threshold values inside strategy cards |
| `typography.body-md` (Inter 400, 14px) | All running text |
| `rounded.md` (6px) | Primary buttons |
| `rounded.lg` (8px) | Inputs, secondary surfaces |
| `rounded.xl` (12px) | Strategy cards, metric callouts |
| `rounded.pill` (9999px) | Main Ticker input, Buy/Sell selector pills |
| `spacing.section` (80px) | Vertical padding between page tabs |

### 4.3 Accepted Visual Constraints

- Streamlit `selectbox` dropdown panels render in a portal outside the page DOM; the trigger is styleable but the open menu retains some default styling.
- The Streamlit slider track accepts a primary-color accent but not a full Binance-chart-handle treatment without an iframe component.
- BinanceNova and BinancePlex are proprietary; Inter and JetBrains Mono substitute per the design system's "Note on Font Substitutes" section.

## 5. Layout & Components

### 5.1 Page Header

Full-width 64px-tall band at the top of every page render:
- Yellow "📈" glyph + page title in `typography.display-md`.
- 1px `hairline-on-dark` divider below.

### 5.2 Sidebar (data management)

- Section labels in uppercase `colors.muted` 12px.
- "Apply Data" → yellow `button-primary`.
- "Reset All" → transparent + 1px hairline `button-secondary-on-dark`.
- Both bound to `on_click` callbacks (Apply Data → `_apply_uploaded_json`).

### 5.3 Edit Strategy Tab — Left Column (~60%)

1. Section header "📝 Edit Strategy" with yellow 4px vertical accent rule.
2. **Main Ticker input**: pill-shaped, 48px tall, JetBrains Mono font for ticker character. Below: `colors.muted` caption "Editing existing" or "Creating new".
3. **Portfolio Weight pair**: slider on left, yellow metric callout on right showing percent in `typography.number-md`.
4. **Buy/Sell selector**: two horizontal pill buttons (replaces native `st.tabs`).
   - Buy: `trading-up` text + border when active, hairline border when inactive.
   - Sell: `trading-down` text + border when active.
   - Selection lives in `st.session_state[f"{lang}_active_side"]`; only the active side's form renders below.
5. **Form fields** in a single-column stack of grouped sections:
   - `BASE` — Target Ticker + Aggregation + Field selectboxes.
   - `PERIOD` — checkbox + conditional number input.
   - `CRITERIA` — criteria type + value.
   - `QUANTITY` — unit + value (split disallowed for sell, per existing behavior).
   - `PRICE BASIS` — single selectbox.
6. **Save button**: full-width yellow `button-primary`, 48px tall, label "💾 Save Changes" or "➕ Add Strategy" depending on whether the ticker already has a saved strategy. Bound to `_save_strategy_callback` via `on_click`.

### 5.4 Edit Strategy Tab — Right Column (~40%)

Renamed from "Current JSON Data" to "Portfolio Composition".

1. **Summary callout** at top:
   - Strategy count in JetBrains Mono 40px.
   - Total portfolio weight, colored green if `== 1.0`, yellow if `< 1.0`, red if `> 1.0`.
2. **Strategy card stack** — one card per ticker, vertical layout, 16px gap:
   - Header row: ticker (bold, 24px) + weight pill (right-aligned).
   - `trading-up` BUY section: 1-line human-readable rule ("When current Close drops 0.50%"), 1-line action ("Buy 10 shares at Close").
   - `trading-down` SELL section: same shape.
   - Footer row: `[ Edit ]` + `[ Delete ]` buttons.
3. **"View Raw JSON" expander** at the bottom — collapses the existing `st.code(json_str)` + download button. Preserves debug/export workflow.

### 5.5 Backtest Tab

1. **Backtest form (left ~33%)**: existing structure (start/end date, initial capital, run button) restyled.
   - Initial Capital value live-previewed above the input as a yellow metric callout.
   - "Start Backtest!" → yellow `button-primary-pill`.
2. **Results panel (right ~67%)**:
   - Top: 4 metric cards via `st.columns(4)` — Final Value / Profit Rate / # Trades / # Tickers. Profit Rate cell colored by direction.
   - Performance matplotlib chart in a dark card.
   - Trade history `st.dataframe`, with buy rows getting a `trading-up` indicator and sell rows getting `trading-down` (via styler).
   - Date slider + portfolio-value-at-date stays; value renders in a metric callout.
   - Monthly snapshot dataframe at the bottom, styled.

### 5.6 Strategy Card Rendering Approach

```python
def render_strategy_card(ticker: str, strategy: dict, lang: str) -> None:
    # 1. Emit the HTML card (header + buy/sell summary blocks) via st.markdown.
    # 2. Render two adjacent st.button widgets below (Edit, Delete) — clickable Streamlit widgets,
    #    not <button> tags inside the HTML, because we need real callbacks.
    # 3. Edit's on_click sets st.session_state[f"{lang}_main_ticker"] = ticker; st.rerun().
    # 4. Delete's on_click pops the ticker from st.session_state["strategies"]; st.rerun().
```

The human-readable rule strings are generated by a `_describe_rule(side: dict, lang: str) -> str` helper in `components.py` that interprets the indicator/threshold/quantity tuples into a sentence.

## 6. Bug Fix Strategy (Symptom 1)

### 6.1 Root-Cause Hypotheses (ranked)

**Primary — widget state outliving JSON uploads.** Streamlit's `st.text_input("...", value=default, key="k")` ignores `value=` once `k` is in session_state. When a JSON is uploaded, `st.session_state["strategies"]` updates but the widget-level keys (`buy_AAPL_en_*`) do not. The form keeps displaying pre-upload values; the user "edits" what they think is the uploaded JSON but is actually stale state; Save writes back near-identical content; JSON view appears not to change.

**Secondary — mutated dict reference.** `current_data = st.session_state["strategies"].get(main_ticker, {})` is a reference. If any code mutates `current_data["buy"]` rather than rebuilding, the dict identity is preserved across reruns and the JSON `st.code` block (serializing the same object) shows no visible diff.

### 6.2 Fixes (defend against both hypotheses)

**Change 1 — Centralize Save through `on_click`.** Replace the `if st.button(...):` block with `st.button(label, on_click=_save_strategy_callback, args=(lang, main_ticker))`. Callbacks run before the next rerun, and they read directly from `st.session_state[widget_key]` — always the user's latest input.

**Change 2 — Clear stale widget state on JSON upload.** `_apply_uploaded_json` purges every session_state key that matches the form's widget-key prefixes (`buy_*`, `sell_*`, `en_weight_*`, `kr_weight_*`) before writing the new `strategies` dict. Next render seeds widgets fresh from the uploaded data.

**Change 3 — Drop the ticker from widget keys.** Switch keys from `f"buy_{main_ticker}_en_crit_val"` to stable per-language slots like `f"en_buy_crit_val"`, and explicitly seed `st.session_state[key] = saved_value` *before* the widget is rendered when the user switches Main Ticker. This eliminates orphaned widget state on ticker rename and follows Streamlit's idiomatic seed-then-render pattern.

**Change 4 — Always rebuild the strategy dict.** Inside `_save_strategy_callback`, the dict written to `st.session_state["strategies"][main_ticker]` is always freshly constructed via `_collect_form_dict()`. Never a mutation of `current_data`. Defensive against the secondary hypothesis.

### 6.3 Verification

Implementer must reproduce the bug first by running `uv run streamlit run streamlit_page.py`, uploading `strategy_test_format.json`, editing AAPL's buy threshold, clicking Save, and confirming the JSON view does not update. After applying Changes 1–4, the same flow must update the JSON view. If reproduction fails (the bug is not what we hypothesized), implementer pauses and reports back before shipping the fix.

## 7. Data Flow

1. **Initial load** — `st.session_state` initialized with `{"strategies": {}, "backtest": None}` by the entry router (existing behavior, unchanged).
2. **JSON upload** — User uploads file → sidebar's Apply Data button calls `_apply_uploaded_json(loaded_dict)` → widget keys purged, `strategies` replaced → `st.rerun()` → form widgets seed from the new strategies dict.
3. **Form edit** — User changes a widget → Streamlit updates `st.session_state[widget_key]` automatically.
4. **Save** — User clicks Save → `_save_strategy_callback(lang, main_ticker)` reads widget state via `_collect_form_dict`, writes a fresh dict to `st.session_state["strategies"][main_ticker]` → next render shows updated card + raw JSON.
5. **Edit from card** — User clicks Edit on a card → callback sets `st.session_state[f"{lang}_main_ticker"] = ticker` → `st.rerun()` → Main Ticker input picks up new value, widgets re-seed from that ticker's saved data.
6. **Delete from card** — User clicks Delete → callback `pop`s ticker from `strategies` → `st.rerun()`.

## 8. Error Handling

- **Invalid JSON upload** — `_apply_uploaded_json` rejects non-dict roots via `isinstance(loaded_data, dict)` check (existing behavior); shows `st.error`. No state mutation on failure.
- **Missing i18n key** — `T(key, lang)` raises `KeyError` at call site; caught by `test_streamlit_i18n.py` at CI time, not at runtime.
- **Save with empty Main Ticker** — Save button is hidden when `main_ticker` is empty (existing behavior preserved).
- **Schema mismatch on save** — The dict produced by `_collect_form_dict` is validated by `test_saved_strategy_validates_against_StrategyWrapper`. We do not validate at runtime in the form — the form's widget options are constrained to valid values, so a runtime validation failure would indicate a code bug, not user error.

## 9. Testing

### 9.1 Automated (added to `tests/`)

**`tests/test_streamlit_forms.py`**:
- `test_collect_form_dict_assembles_schema` — seed a fake state dict, assert returned dict matches `StrategyWrapper`'s buy/sell schema.
- `test_collect_form_dict_coerces_count_to_int` — `qty_val=10.0` + `qty_type="count"` → `["count", 10]` (int, not float).
- `test_collect_form_dict_period_disabled_returns_false` — `use_period=False` → returned `window` is `False`, not `0` or `None`.
- `test_apply_uploaded_json_clears_widget_state` — pre-populate fake state with stale widget keys, call `_apply_uploaded_json(new_data)`, assert stale keys are gone and `strategies` updated.
- `test_saved_strategy_validates_against_StrategyWrapper` — round-trip the output of `_collect_form_dict` through `StrategyWrapper.model_validate`. Contract test between form and engine.

`_collect_form_dict` and `_apply_uploaded_json` accept an optional `state` parameter that defaults to `st.session_state`. Tests pass a plain `dict` for `state` to avoid spinning up Streamlit.

**`tests/test_streamlit_i18n.py`**:
- Loads `LABELS` from `streamlit_ui.i18n`, asserts every key referenced by the page files exists in both language dicts.
- Build the reference set by grep'ing `T("...", lang)` calls at test time.

### 9.2 Manual Smoke Tests

Implementer runs each in 30 seconds before claiming task complete:

1. **Fresh load** — Add AAPL strategy → card + JSON both populate.
2. **Edit existing** — Change buy criteria value → Save → card + JSON reflect new value.
3. **Upload JSON** — Apply `strategy_test_format.json` → cards for AAPL + TQQQ appear; form populates with AAPL when Main Ticker = AAPL.
4. **Upload → edit → save** — After step 3, change AAPL's quantity → Save → JSON shows new quantity. (This is the symptom-1 regression check.)
5. **Edit from card** — Click Edit on TQQQ card → Main Ticker switches to TQQQ; form shows TQQQ values without AAPL leftovers.
6. **Delete from card** — Click Delete on AAPL card → ticker removed from both card stack and raw JSON.
7. **Language toggle** — Switch to Korean → data persists; Korean strings render; yellow CTAs unchanged.
8. **Backtest** — Run a backtest → metric cards render; Profit Rate colored by direction; chart renders.
9. **Narrow viewport (~600px)** — Layout does not break; cards stack vertically; CTAs stay full-width.

### 9.3 Existing Tests

`uv run pytest -q` must continue to pass — the 28 tests on `main` cover the engine, not the UI. New tests add to that count.

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Streamlit CSS escape hatches stop working after a Streamlit version bump | Pin Streamlit version in `pyproject.toml`; document the CSS selectors used in `theme.py` as version-coupled |
| Bug-fix hypothesis is wrong, "fix" masks the real cause | Implementer reproduces the bug *before* fixing; pauses + reports if reproduction fails |
| Card render produces wrong human-readable rule for an indicator/threshold combination | `_describe_rule` is a pure function; add `tests/test_streamlit_forms.py::test_describe_rule_*` cases for each combination |
| EN and KO drift again after refactor | Single shared `forms.py` and `components.py`; i18n test ensures keys stay in sync |
| Korean translations may not exist for new card UI text ("Editing existing", "Portfolio Composition", etc.) | Implementer asks user for translations before shipping; falls back to English string if `LABELS["ko"]` missing a key only as a development scaffold, with a `# TODO ko translation` marker |

## 11. Out of Scope (Explicit)

- Markets table component, trader-row, arena-gradient hero (Binance design system components without a use case in this app).
- Visual regression / screenshot diffing infrastructure.
- Changes to `streamlit_page.py` entry router.
- Changes to the language toggle behavior (still a sidebar radio).
- Mobile-first design (must not break, but desktop is primary).
- Engine, model, or strategy schema changes.

## 12. Open Questions Resolved During Brainstorming

| Question | Decision |
|---|---|
| Visual scope | Heavy reskin + restructure |
| Page coverage | EN + KO both, in one pass |
| JSON viewer fate | Replace with summary cards; raw JSON behind expander |
| Bug fix sequencing | Fix as part of redesign (first commit on the branch) |
| Implementation approach | Approach A — pure Streamlit + CSS injection |
| Refactor scope | Extract shared `streamlit_ui/` module |
| Visual diff tooling | Skip |
