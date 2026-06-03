"""
explore_tui.py — a Textual TUI to *play with* the all-states stack.

Two modes:
  [S] Search   — query the all-states API directly (employment search / officers-by-name).
  [P] Pipeline — run the REAL entity-resolution pipeline on ONE mention you type in
                 (Stage 0 early-filter -> candidates -> XGBoost score -> exact-name gate
                 -> agency validation), via PostMatcher.find_canonical_stint([mention]).

WHY IT LIVES HERE (resolve/src) AND RUNS UNDER THE VENV:
  - Pipeline mode imports features.py -> sentence_transformers; under a Python that also has
    TensorFlow (the global one) that deadlocks at import in a non-TTY context. The venv has no
    TensorFlow. So: run with the venv.
  - find_canonical_stint reads ../data/input/common_last_names.csv and
    ../models/best_model_xgboost.pkl by RELATIVE path, so cwd must be resolve/src.

RUN:
    # 1. start the all-states API (separate terminal), e.g. on :8001
    cd server_all_states && /Users/ayyubibrahim/bin/python3 src.py
    # 2. run this TUI from resolve/src, pointed at that API
    cd resolve/src && NPI_API_URL=http://localhost:8001 ../../venv/bin/python explore_tui.py

KEYS:
  - Click a field (or Tab between fields) and type. Press Enter or the button to submit.
  - Results land in a table: it is FOCUSED automatically, so ↑/↓ move row-by-row,
    PgUp/PgDn page, Ctrl+Home/Ctrl+End jump to top/bottom. Tab back to the form to edit.

Pipeline mode also needs OPENAI_API_KEY in resolve/src/.env (agency validation LLM).
"""
import os
import sys

# Make sibling modules (api, features, helpers) and the repo root (models.src) importable
# regardless of cwd, and pin cwd to this dir so the pipeline's relative paths resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root for `models.src`

# Default the API target to the all-states server if the user didn't set it.
os.environ.setdefault("NPI_API_URL", "http://localhost:8001")
API_URL = os.environ["NPI_API_URL"]

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, RichLog, Static, Switch,
)

from api import NPIClient


# --------------------------------------------------------------------------------------
# Reusable labeled input (official compound-widget pattern) — guarantees every field
# renders with a clear label and a full-width box.
# --------------------------------------------------------------------------------------
class Field(Widget):
    DEFAULT_CSS = """
    Field { layout: horizontal; height: 3; }
    Field Label { width: 22; padding: 1 1 0 1; text-align: right; color: $text-muted; }
    Field Input { width: 1fr; }
    """

    def __init__(self, label: str, field_id: str, placeholder: str = "") -> None:
        super().__init__(id=field_id)  # the id lives on the Field container
        self._label = label
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        yield Label(self._label)
        yield Input(placeholder=self._placeholder)

    @property
    def value(self) -> str:
        return self.query_one(Input).value

    def focus_input(self) -> None:
        self.query_one(Input).focus()


def style_table(t: DataTable) -> None:
    """Consistent, readable, keyboard-scrollable table."""
    t.cursor_type = "row"
    t.zebra_stripes = True


# --------------------------------------------------------------------------------------
# Mode selection screen
# --------------------------------------------------------------------------------------
class ModeScreen(Screen):
    BINDINGS = [("s", "go_search", "Search"), ("p", "go_pipeline", "Pipeline"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="mode-box"):
            yield Label("NPI all-states explorer", id="title")
            yield Static(f"API: {API_URL}", id="api-line")
            yield Static("checking API…", id="health")
            yield Static("")
            yield Button("Search the database   [ S ]", id="btn-search", variant="primary")
            yield Static("Direct API queries: employment search, or all officers with a name.",
                         classes="hint")
            yield Static("")
            yield Button("Run the full pipeline   [ P ]", id="btn-pipeline", variant="success")
            yield Static("Entity resolution on ONE mention. Needs name + state + incident year.",
                         classes="hint")
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._check_health, thread=True)

    def _check_health(self) -> None:
        ok = NPIClient(base_url=API_URL).health_check()
        self.app.call_from_thread(self._show_health, ok)

    def _show_health(self, ok: bool) -> None:
        h = self.query_one("#health", Static)
        if ok:
            h.update("[green]● API reachable[/green]")
        else:
            h.update(f"[red]● API NOT reachable at {API_URL}[/red] — start server_all_states (:8001)")

    def action_go_search(self) -> None:
        self.app.push_screen(SearchScreen())

    def action_go_pipeline(self) -> None:
        self.app.push_screen(PipelineScreen())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-search":
            self.action_go_search()
        elif event.button.id == "btn-pipeline":
            self.action_go_pipeline()


# --------------------------------------------------------------------------------------
# Search screen — direct API calls (no ML)
# --------------------------------------------------------------------------------------
class SearchScreen(Screen):
    BINDINGS = [
        ("escape", "back", "Back"),
        ("ctrl+r", "run", "Search"),
        ("ctrl+f", "focus_form", "Edit fields"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with VerticalScroll(id="form"):
                yield Label("Database search", classes="screen-title")
                yield Field("First name", "first", "e.g. John")
                yield Field("Last name", "last", "e.g. Smith")
                yield Field("State", "state", "CA / california")
                yield Field("Agency", "agency", "optional")
                yield Field("Limit", "limit", "default 25")
                with Horizontal(id="switch-row"):
                    yield Label("All officers by name:", id="switch-lbl")
                    yield Switch(id="by-name")
                with Horizontal(id="btn-row"):
                    yield Button("Search", id="run", variant="primary")
                    yield Button("Back", id="back")
                yield Static("Enter a name or agency, then Search.", id="status")
            yield DataTable(id="results")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#results", DataTable)
        t.add_columns("person_nbr", "first", "middle", "last", "agency", "type", "start", "end", "state")
        style_table(t)
        self.query_one("#first", Field).focus_input()

    # --- navigation -------------------------------------------------------------------
    def action_back(self) -> None:
        self.app.pop_screen()

    def action_focus_form(self) -> None:
        self.query_one("#first", Field).focus_input()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "run":
            self.action_run()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_run()

    # --- search ----------------------------------------------------------------------
    def action_run(self) -> None:
        first = self.query_one("#first", Field).value.strip()
        last = self.query_one("#last", Field).value.strip()
        state = self.query_one("#state", Field).value.strip() or None
        agency = self.query_one("#agency", Field).value.strip() or None
        limit_raw = self.query_one("#limit", Field).value.strip()
        by_name = self.query_one("#by-name", Switch).value
        if not first and not last and not agency:
            self._status("[yellow]Enter at least a name or agency.[/yellow]")
            return
        try:
            limit = int(limit_raw) if limit_raw else 25
        except ValueError:
            self._status("[red]Limit must be a number.[/red]")
            return
        self._status("searching…")
        self.run_worker(
            lambda: self._search(first, last, state, agency, limit, by_name),
            thread=True, exclusive=True,
        )

    def _search(self, first, last, state, agency, limit, by_name):
        client = NPIClient(base_url=API_URL)
        try:
            if by_name:
                recs = client.get_officers_by_name(first_name=first, last_name=last, state=state)
            else:
                recs = client.get_post_employment_records(
                    first_name=first or None, last_name=last or None,
                    agency=agency, state=state, limit=limit,
                )
            self.app.call_from_thread(self._show_results, recs, None)
        except Exception as e:  # noqa: BLE001 — surface anything to the UI
            self.app.call_from_thread(self._show_results, [], str(e))

    def _show_results(self, recs, err):
        t = self.query_one("#results", DataTable)
        t.clear()
        if err:
            self._status(f"[red]Error: {err}[/red]")
            return
        for r in recs:
            t.add_row(
                r.post_person_nbr, r.post_first_name, r.post_middle_name or "",
                r.post_last_name, r.post_agency_name,
                getattr(r.post_agency_type, "value", r.post_agency_type),
                _d(r.post_start_date), _d(r.post_end_date), r.state or "",
            )
        n_persons = len({r.post_person_nbr for r in recs})
        self._status(
            f"[green]{len(recs)} record(s), {n_persons} distinct person(s).[/green]  "
            f"[dim]↑/↓ scroll · PgUp/PgDn page · Ctrl+F to edit fields[/dim]"
        )
        if recs:
            t.focus()  # focused table => arrow keys scroll it natively

    def _status(self, msg: str) -> None:
        self.query_one("#status", Static).update(msg)


# --------------------------------------------------------------------------------------
# Pipeline screen — runs the real entity-resolution on one mention
# --------------------------------------------------------------------------------------
class PipelineScreen(Screen):
    BINDINGS = [
        ("escape", "back", "Back"),
        ("ctrl+r", "run", "Run"),
        ("ctrl+f", "focus_form", "Edit fields"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with VerticalScroll(id="form"):
                yield Label("Pipeline ( * required )", classes="screen-title")
                yield Field("First name *", "first", "e.g. Scott")
                yield Field("Last name *", "last", "e.g. Lunger")
                yield Field("Middle name", "middle", "optional")
                yield Field("State *", "state", "CA")
                yield Field("Incident year *", "year", "e.g. 2019")
                yield Field("Source agency", "agency", "optional")
                yield Field("Mentioned agencies", "mentioned", "comma-separated")
                with Horizontal(id="btn-row"):
                    yield Button("Run", id="run", variant="success")
                    yield Button("Back", id="back")
                yield Static("Fill required fields, then Run.", id="verdict")
            with Vertical(id="out"):
                yield Label("Scored candidates", classes="pane-title")
                yield DataTable(id="cands")
                yield Label("Pipeline log", classes="pane-title")
                yield RichLog(id="log", wrap=True, highlight=False, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#cands", DataTable)
        t.add_columns("person_nbr", "first", "last", "agency", "start", "end", "prob")
        style_table(t)
        self.query_one("#first", Field).focus_input()

    # --- navigation -------------------------------------------------------------------
    def action_back(self) -> None:
        self.app.pop_screen()

    def action_focus_form(self) -> None:
        self.query_one("#first", Field).focus_input()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "run":
            self.action_run()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_run()

    # --- run pipeline ----------------------------------------------------------------
    def action_run(self) -> None:
        first = self.query_one("#first", Field).value.strip()
        last = self.query_one("#last", Field).value.strip()
        middle = self.query_one("#middle", Field).value.strip()
        state = self.query_one("#state", Field).value.strip()
        year_raw = self.query_one("#year", Field).value.strip()
        agency = self.query_one("#agency", Field).value.strip()
        mentioned = self.query_one("#mentioned", Field).value.strip()

        missing = [n for n, v in [("first name", first), ("last name", last),
                                  ("state", state), ("incident year", year_raw)] if not v]
        if missing:
            self._verdict(f"[yellow]Required: {', '.join(missing)}.[/yellow]")
            return
        try:
            year = int(year_raw)
        except ValueError:
            self._verdict("[red]Incident year must be a number.[/red]")
            return

        self._verdict("running pipeline… loads the model + makes 1 LLM call (~10–30s)")
        self.query_one("#cands", DataTable).clear()
        self.query_one("#log", RichLog).clear()
        self.run_worker(
            lambda: self._run_pipeline(first, last, middle, state, year, agency, mentioned),
            thread=True, exclusive=True,
        )

    def _run_pipeline(self, first, last, middle, state, year, agency, mentioned):
        # Run the pipeline in a SEPARATE PROCESS (pipeline_runner.py). This isolates ALL of
        # its output — including the fd-level writes from torch/transformers/tqdm/warnings —
        # into a captured log file, so none of it can bleed onto the Textual terminal.
        import hashlib
        import json
        import tempfile
        import subprocess

        uid = hashlib.sha256(f"{first}|{last}|{state}|{year}|{agency}".encode()).hexdigest()[:16]
        mention = {
            "uid": uid, "first": first, "last": last, "middle": middle,
            "state": state, "year": year, "agency": agency, "mentioned": mentioned,
        }
        in_f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(mention, in_f); in_f.close()
        out_path = in_f.name + ".out"
        log_path = in_f.name + ".log"

        try:
            with open(log_path, "w") as logf:
                proc = subprocess.run(
                    [sys.executable, os.path.join(_HERE, "pipeline_runner.py"), in_f.name, out_path],
                    cwd=_HERE, env=os.environ.copy(),
                    stdout=logf, stderr=subprocess.STDOUT, timeout=180,
                )
            log = _read(log_path)
            if proc.returncode != 0 or not os.path.exists(out_path):
                tail = "\n".join(log.splitlines()[-8:])
                self.app.call_from_thread(
                    self._show_pipeline, None, log,
                    f"pipeline exited with code {proc.returncode}. {tail}")
                return
            with open(out_path) as f:
                result = json.load(f)
            self.app.call_from_thread(self._show_pipeline, result, log, None)
        except subprocess.TimeoutExpired:
            self.app.call_from_thread(self._show_pipeline, None, _read(log_path),
                                      "pipeline timed out after 180s")
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self._show_pipeline, None, _read(log_path), str(e))
        finally:
            for p in (in_f.name, out_path, log_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _show_pipeline(self, result, log, err):
        rlog = self.query_one("#log", RichLog)
        if log:
            rlog.write(log[-8000:])

        if err:
            self._verdict(f"[red]Pipeline error: {err}[/red]")
            return

        matched = result.get("matched") or []
        if matched:
            m = matched[0]
            self._verdict(
                f"[b green]AUTO-MATCHED[/b green] → {m['post_first_name']} {m['post_last_name']} "
                f"| {m['post_agency_name']} | POST {m['post_person_nbr']} "
                f"| prob {m['match_probability']:.3f}"
            )
        else:
            self._verdict(f"[b yellow]ROUTED TO REVIEW[/b yellow] — {result.get('reason') or 'No match'}")

        t = self.query_one("#cands", DataTable)
        t.clear()
        cands = result.get("candidates") or []
        for c in cands:
            prob = c.get("match_probability")
            t.add_row(
                c["post_person_nbr"], c["post_first_name"], c["post_last_name"],
                c["post_agency_name"], c["post_start_date"], c["post_end_date"],
                f"{prob:.3f}" if prob is not None else "",
            )
        if cands:
            t.focus()  # focused => arrow keys scroll the candidates table

    def _verdict(self, msg: str) -> None:
        self.query_one("#verdict", Static).update(msg)


def _d(v):
    """Format a date-ish value compactly for a cell."""
    if v is None or isinstance(v, float):
        return ""
    s = str(v)
    return s[:10] if s and s != "NaT" else ""


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


class ExploreApp(App):
    CSS = """
    Screen { background: $surface; }

    #mode-box { padding: 1 3; width: 70; margin: 1 2; }
    #title { text-style: bold; color: $accent; }
    #api-line { color: $text-muted; }
    .hint { color: $text-muted; padding: 0 0 0 2; }

    /* Side-by-side: a fixed-width form on the left, output filling the rest on the right.
       This keeps the results table FULL HEIGHT and always visible, no matter how tall the
       form is or how short the terminal window is. */
    #body { height: 1fr; }
    #form { width: 46; height: 1fr; padding: 1 2; border-right: vkey $primary-darken-2; }
    .screen-title { text-style: bold; color: $accent; padding: 0 0 1 0; }
    #switch-row { height: 3; align-vertical: middle; }
    #switch-lbl { width: auto; padding: 1 1 0 1; color: $text-muted; }
    #btn-row { height: auto; padding: 1 0 0 0; }
    #btn-row Button { margin: 0 2 0 0; }
    #status, #verdict { padding: 1 0; height: auto; }

    #results { width: 1fr; height: 1fr; border: round $primary; }

    #out { width: 1fr; height: 1fr; }
    .pane-title { text-style: bold; color: $accent; padding: 0 1; }
    #cands { height: 1fr; border: round $primary; }
    #log { height: 1fr; border: round $primary-darken-1; background: $panel; }

    DataTable:focus { border: round $accent; }
    """
    BINDINGS = [
        # `q` quits whenever you're not typing in a text field (mode screen, or after a
        # search when the table/buttons have focus). Ctrl+C quits ALWAYS — priority=True
        # makes it fire even while an Input is focused, so it's the guaranteed one-key exit.
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", priority=True, show=False),
        Binding("ctrl+q", "quit", "Quit", show=False),
    ]

    def on_mount(self) -> None:
        self.title = "NPI all-states explorer"
        self.push_screen(ModeScreen())


if __name__ == "__main__":
    ExploreApp().run()
