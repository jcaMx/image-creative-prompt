# app_dynamic.py
"""
Dynamic NiceGUI app driven by JSON config.

Requirements:
pip install requests beautifulsoup4 langchain langchain-openai openai nicegui


"""

import os
import asyncio
import json
import re
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nicegui import ui
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import hashlib

# --------------------------
# CONFIG JSON (dynamic form)
# --------------------------
# You can reuse / adapt this CONFIG for other apps.
CONFIG_JSON = {
    "title": "AI Image Prompt Generator (Dynamic)",
    "subtitle": "Generate structured AI image prompts for marketing campaigns, ensuring high quality visuals. Enter a business URL, let the model summarize, propose value themes, and generate visual concepts + image prompts.",
    "sections": [
        {
            "title": "1) Business Site",
            "fields": [
                {
                    "id": "business_site_url",
                    "label": "Business Site URL",
                    "type": "text",
                    "placeholder": "https://example.com",
                    "help": "Enter the business homepage URL. This triggers scraping."
                }
            ]
        },
        {
            "title": "2) Business Summary",
            "fields": [
                {
                    "id": "business_summary",
                    "label": "Business Summary",
                    "type": "textarea",
                    "placeholder": "(auto-generated)",
                    "prompt": "Generate a concise structured business summary for {business_name} based on the scraped site content below. Focus on: operations, unique value propositions, industry.\n\nScraped Content:\n{scraped_text}\n\nStructured summary:",
                    "depends_on": ["scraped_text"],
                    "readonly": True
                }
            ]
        },
        {
            "title": "3) Business Value Themes",
            "fields": [
                {
                    "id": "business_value_themes",
                    "label": "Business Value Themes",
                    "type": "button_list",
                    "prompt": "Based on the business summary below, list 3 unique value themes (short phrases) that would resonate with potential customers. Return them as newline-separated list.\n\nBusiness Summary:\n{business_summary}",
                    "depends_on": ["business_summary"]
                }
            ]
        },
        {
            "title": "4) Visual Concepts",
            "fields": [
                {
                    "id": "visual_concepts",
                    "label": "Visual Concepts",
                    "type": "button_list",
                    "prompt": "Based on the selected business value theme below, list 3 visually evocative ad concepts (short phrases). Return them as newline-separated list.\n\nSelected Theme:\n{business_value_themes}",
                    "depends_on": ["business_value_themes"]
                }
            ]
        },
        {
            "title": "5) Image Prompts",
            "fields": [
                {
                    "id": "image_prompts",
                    "label": "Image Prompts",
                    "type": "textarea",
                    "placeholder": "(auto-generated)",
                    "prompt": "Based on the selected visual concept below, write 3 specific detailed image prompts suitable for an AI image generator. Include style, subject, composition, lighting, and photography/illustration directions. Return as newline-separated list.\n\nSelected Visual Concept:\n{visual_concepts}",
                    "depends_on": ["visual_concepts"],
                    "readonly": True
                }
            ]
        }
    ]
}

# --------------------------
# HTTP + LLM setup
# --------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable")

# LangChain ChatOpenAI client (sync generate)
llm = ChatOpenAI(temperature=0, model="gpt-4")


# requests session with retries
session = requests.Session()
session.headers.update({"User-Agent": "Dynamic-Form-Scraper/1.0"})
retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

# keywords (used if you want to expand link discovery)
TARGET_KEYWORDS = [
    "about", "team", "mission", "values", "services", "solutions", "products",
    "industries", "clients", "case-studies", "projects", "blog", "insights",
    "resources", "news", "careers", "jobs", "contact"
]

# --------------------------
# Utilities
# --------------------------
def parse_list_response(text: str) -> list:
    """Parse LLM list outputs into a Python list (strip numbering/bullets)."""
    if not text:
        return []
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # strip bullets or numbering like "1. ", "- ", "• "
        line = re.sub(r'^\s*[\-\*\u2022\d\.\)\:]+\s*', '', line)
        if line:
            lines.append(line)
    if not lines:
        # fallback: split by commas
        parts = [p.strip() for p in re.split(r'[,\n]+', text) if p.strip()]
        lines = parts
    # dedupe while preserving order
    seen = set()
    out = []
    for l in lines:
        kl = l.lower()
        if kl not in seen:
            out.append(l)
            seen.add(kl)
    return out[:20]

def extract_llm_text(generation_result) -> str:
    """Safely extract text from langchain-openai generate response."""
    try:
        return generation_result.generations[0][0].text
    except Exception:
        return str(generation_result)

# --------------------------
# LLM call helper
# --------------------------
async def call_llm_text(prompt_text: str) -> str:
    """Call ChatOpenAI safely and return generated text (robust extraction)."""
    loop = asyncio.get_running_loop()

    # call the llm in a thread so we don't block the event loop
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: llm([HumanMessage(content=prompt_text)])
        )
    except Exception as e:
        err = f"LLM call failed: {e}"
        print("[DEBUG] call_llm_text error:", err)
        return err

    # extract text from a variety of possible response shapes
    text = ""
    try:
        # easiest cases first
        if resp is None:
            text = ""
        elif isinstance(resp, str):
            text = resp
        # ChatOpenAI sometimes returns an AIMessage-like object with .content
        elif hasattr(resp, "content"):
            text = resp.content
        # older/langchain llm.generate style LLMResult -> resp.generations[0][0].text
        elif hasattr(resp, "generations"):
            try:
                text = resp.generations[0][0].text
            except Exception:
                text = str(resp)
        # openai-like choices
        elif hasattr(resp, "choices"):
            try:
                choice = resp.choices[0]
                # choice might contain .message or .text
                if hasattr(choice, "message") and isinstance(choice.message, dict):
                    text = choice.message.get("content") or choice.message.get("text") or str(choice.message)
                else:
                    text = getattr(choice, "text", str(choice))
            except Exception:
                text = str(resp)
        else:
            # fallback to stringifying the response
            text = str(resp)
    except Exception as e:
        text = f"LLM extract error: {e}"

    # debug: print what we received
    try:
        print(f"[DEBUG] call_llm_text returned (len={len(text)}): {repr(text)[:1000]}")
    except Exception:
        print("[DEBUG] call_llm_text returned (unable to print length)")

    return text


def show_message(title: str, message: str):
    message_label.set_text(f"{title}\n{message}")
    message_dialog.open()


def start_progress(msg: str):
    progress_label.text = msg
    progress_dialog.open()

def stop_progress():
    try:
        progress_dialog.close()
    except Exception:
        pass


# --------------------------
# Scraping helpers
# --------------------------
def scrape_text_from_url(url: str) -> str:
    """Basic page text extraction using BeautifulSoup."""
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
    except Exception:
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def discover_relevant_links(root_url: str, homepage_html: str = None):
    """Find internal links and rank by keyword presence (basic)."""
    try:
        if homepage_html:
            soup = BeautifulSoup(homepage_html, "html.parser")
        else:
            r = session.get(root_url, timeout=10); r.raise_for_status(); soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return {}
    all_links = [urljoin(root_url, a['href']) for a in soup.find_all('a', href=True)]
    internal = [l for l in all_links if urlparse(l).netloc == urlparse(root_url).netloc]
    unique = list(dict.fromkeys(internal))
    def score(u):
        ul = u.lower()
        return sum(kw in ul for kw in TARGET_KEYWORDS)
    scored = sorted(unique, key=score, reverse=True)
    results = {}
    for kw in TARGET_KEYWORDS:
        for u in scored:
            if kw in u.lower():
                if kw not in results:
                    results[kw] = u
    return results

# --------------------------
# App dynamic engine state
# --------------------------
# Stores field values (selected values or text)
state_values = {}
# Stores generated options for button_list fields: field_id -> [options]
state_options = {}
# Widget references: field_id -> widget or container
widgets = {}
# Dependency map: field_id -> list of fields that depend on it
dependency_map = {}

# Build dependency map from config
for sec in CONFIG_JSON["sections"]:
    for fld in sec["fields"]:
        for dep in fld.get("depends_on", []):
            dependency_map.setdefault(dep, []).append(fld["id"])

# Helper: safe formatter using state_values + state_options
def build_prompt_from_template(template: str) -> str:
    """
    Format the prompt template with values from state_values and state_options.
    If an expected key is missing, substitute an empty string.
    """
    # build a dict that includes both values and options (options joined by newline)
    fmt = {}
    fmt.update({k: (v if isinstance(v, str) else ("\n".join(v) if v else "")) for k, v in state_values.items()})
    fmt.update({k: ("\n".join(v) if isinstance(v, list) else str(v)) for k, v in state_options.items()})
    # Also include 'scraped_text' if present in state_values
    try:
        return template.format(**fmt)
    except Exception:
        # fallback: safe replace braces to avoid crash
        return template

# --------------------------
# UI rendering from CONFIG
# --------------------------
with ui.card().classes(
    "w-full max-w-none mx-auto shadow-lg rounded-xl "
    "bg-gradient-to-r from-gray-50 to-gray-100 border border-gray-200"
):
    with ui.column().classes("w-full p-4 sm:p-6 md:p-8 space-y-3 sm:space-y-4"):
        # Main title with responsive text sizing
        ui.label(CONFIG_JSON.get("title", "Dynamic App")).classes(
            "text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center "
            "bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent "
            "leading-tight"
        )
        
        # Subtitle with responsive spacing and styling
        if CONFIG_JSON.get("subtitle", ""):
            ui.label(CONFIG_JSON.get("subtitle", "")).classes(
                "text-xs sm:text-sm md:text-base text-center text-gray-600 "
                "max-w-2xl mx-auto leading-relaxed px-2"
            )
        
        # Status indicator with icon-like styling
        with ui.row().classes("justify-center items-center mt-2 sm:mt-4"):
            ui.icon("circle", size="xs").classes("text-gray-500")
            status = ui.label("Loading...").classes(
                "text-xs sm:text-sm text-gray-700 font-medium ml-1"
            )   

@ui.page("/result")
def result_page():
    with ui.card().classes(
        "w-full max-w-3xl mx-auto shadow-lg rounded-xl bg-gradient-to-r "
        "from-gray-50 to-gray-100 p-6 space-y-6"
    ):
        # Title
        ui.label("✅ Final Results").classes(
            "text-2xl sm:text-3xl font-bold text-center text-gray-800"
        )

        # Subtitle / note
        ui.label("Here are the details generated for your business:").classes(
            "text-sm text-gray-600 text-center mb-4"
        )

        # Iterate sections
        for sec in CONFIG_JSON["sections"]:
            sec_title = sec["title"]

            with ui.card().classes(
                "bg-white shadow-sm rounded-lg p-4 border border-gray-200 space-y-2"
            ):
                ui.label(sec_title).classes("text-lg font-semibold text-gray-700")

                for fld in sec["fields"]:
                    fid = fld["id"]
                    val = state_values.get(fid) or state_options.get(fid)

                    if not val:
                        ui.label("(no data)").classes("text-gray-400 text-sm")
                        continue

                    # Lists (button_list outputs)
                    if isinstance(val, list):
                        with ui.column().classes("space-y-1"):
                            for v in val:
                                ui.label(f"• {v}").classes("text-gray-700 text-sm")
                    else:
                        # Single values (text or textarea)
                        ui.textarea(value=str(val), readonly=True) \
                            .props("filled") \
                            .classes("w-full text-sm text-gray-700")

        # ✅ Pre-create dialog for messages
        with ui.dialog() as result_dialog, ui.card().classes(
            "p-6 rounded-xl shadow-lg bg-white text-center"
        ):
            dialog_label = ui.label("").classes("text-lg font-semibold text-gray-800")
            ui.button("OK", on_click=result_dialog.close).classes(
                "mt-4 bg-black text-white w-full"
            )

        # Save JSON button
        def save_state_to_file():
            payload = {
                "values": state_values,
                "options": state_options,
            }
            fname = f"{state_values.get('business_name','business')}_output.json".replace(" ", "_")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            # ✅ Just update and open the dialog
            dialog_label.set_text(f"Saved to {fname}")
            result_dialog.open()

        ui.button(
            "💾 Save as JSON",
            on_click=save_state_to_file,
        ).classes(
            "w-full bg-gray-800 text-white mt-4 py-3 rounded-lg hover:bg-gray-700"
        )



# --------------------------
# Progress dialog (global overlay)
# --------------------------
with ui.dialog() as progress_dialog, ui.card().classes(
    "p-6 rounded-xl shadow-lg bg-white text-center"
):
    progress_label = ui.label("Working...").classes(
        "text-lg font-semibold text-gray-800"
    )
    ui.spinner(size="lg", color="gray").classes("mt-4")


# --- define once, globally ---
# inside your main page build code (not inside a function)
with ui.dialog() as message_dialog, ui.card().classes("p-6 rounded-xl shadow-lg bg-white text-center"):
    message_label = ui.label("").classes("text-lg font-semibold text-gray-800")
    ok_button = ui.button("OK", on_click=message_dialog.close).classes("mt-4 bg-black text-white w-full")


# Containers for sections (so UI looks grouped)
section_containers = {}

# track fields currently being processed to avoid concurrent duplicate calls
_in_flight_fields = set()

# store last prompt hash for each field so we can skip if prompt unchanged
_last_prompt_hash = {}


def make_on_change(_fid):
    async def _handler(event):
        # Prefer to read the authoritative widget value if widget exists
        widget = widgets.get(_fid)
        val = None

        if widget is not None:
            # ui.input widgets expose current value via .value or .get_value()
            try:
                # modern NiceGUI: widget.value
                val = getattr(widget, "value", None)
            except Exception:
                val = None

            # fallback: some widget types implement get_value()
            if val is None:
                try:
                    val = widget.get_value()
                except Exception:
                    val = None

        # If we couldn't get value from widget, fall back to event extraction
        if val is None:
            if hasattr(event, "value"):
                val = event.value
            elif hasattr(event, "args") and event.args:
                val = event.args[0]
            elif isinstance(event, str):
                val = event
            else:
                try:
                    val = str(event)
                except Exception:
                    val = ""

        # Normalise value to string and trim
        val = val or ""
        if isinstance(val, (list, dict)):
            try:
                val = json.dumps(val, ensure_ascii=False)
            except Exception:
                val = str(val)
        if isinstance(val, str):
            val = val.strip()

        # Store the value (no auto-resolve here; Next will trigger resolution)
        state_values[_fid] = val
        status.set_text(f"Set {_fid}: {val[:60]}")  # show first 60 chars for UX
        print("EVENT RAW:", repr(event))
        print("WIDGET VALUE:", getattr(widget, "value", None))

    return _handler


for section in CONFIG_JSON["sections"]:
    sec_title = section.get("title", "")
    display = "block" if sec_title.startswith("1)") else "none"
    container = ui.card().classes("p-4 mb-4 shadow-sm rounded-lg bg-white w-full").style(f"display:{display}")



    section_containers[sec_title] = container
    
    with container:
        ui.markdown(f"### {sec_title}")

        # build fields inside this section
        for fld in section["fields"]:
            fid = fld["id"]
            ftype = fld.get("type", "text")
            label = fld.get("label", fid)
            placeholder = fld.get("placeholder", "")
            # create widgets according to type
            if ftype == "text":
                w = ui.input(label=label, placeholder=placeholder, value="").classes("w-full p-2 border rounded-md")

                # when user changes value, trigger dependency resolution
                widgets[fid] = w
                w.on('change', make_on_change(fid))
                

            elif ftype == "textarea":
                # Create textarea and store it in widgets so we can update it later
                w = ui.textarea(label=label, placeholder=placeholder, value="").classes("w-full p-3 border rounded-md")
                w._props["autogrow"] = True

                widgets[fid] = w   # <<< IMPORTANT: keep reference so resolve_dependents_for can update it

                # If config requests readonly, set via props (best-effort)
                if fld.get("readonly"):
                    try:
                        w.props("readonly")
                    except Exception:
                        pass

                # If config specifies rows (int) or height (string), apply safely
                if "rows" in fld and isinstance(fld["rows"], int):
                    try:
                        w.props(f"rows={fld['rows']}")
                    except Exception:
                        pass
                if "height" in fld and isinstance(fld["height"], str):
                    w.style(f"height:{fld['height']}")

                # make it auto-grow instead of scroll
                w.props("autogrow")

                if fld.get("readonly"):
                    w.props("readonly")

                w.style("width:100%")
                widgets[fid] = w
            elif ftype == "button_list":
                # create a column that will hold stacked buttons
                col = ui.column().style("width:100%")
                widgets[fid] = col
                # if config provides static options, render them now
                static_opts = fld.get("options", [])
                if static_opts:
                    # create buttons stacked full width
                    for opt in static_opts:
                        def make_btn_handler(_fid=fid, _opt=opt):
                            async def _on_click():
                                state_values[_fid] = _opt
                                status.set_text(f"Selected {_opt} for {_fid}")
                                await resolve_dependents_for(_fid)
                            return _on_click
                        with col:
                            ui.button(opt, on_click=make_btn_handler()).style("width:100%; margin-bottom:6px")
            else:
                widgets[fid] = ui.label(f"Unknown field type: {ftype}")

# --------------------------
# Step control & single Next button (create AFTER all sections are built)
# --------------------------
# ordered containers in the same order as CONFIG_JSON
ordered_containers = [section_containers[sec.get("title")] for sec in CONFIG_JSON["sections"]]

# step state
_step = {"i": 0}

def show_step(index: int):
    """Make all steps up to index visible (stacked UI)."""
    total = len(ordered_containers)
    if index < 0:
        index = 0
    if index >= total:
        index = total - 1
    _step["i"] = index

    # 🔑 Instead of hiding old steps, keep them visible
    for idx, container in enumerate(ordered_containers):
        try:
            if idx <= index:
                container.style("display:block")
            else:
                container.style("display:none")
        except Exception:
            pass

    # Update button label
    try:
        if index == total - 1:
            next_button.set_text("Finish")
        else:
            next_button.set_text("Next")
    except Exception:
        pass

    status.set_text(f"Step {index+1} of {total}")


# guard to prevent re-entrancy
_running_next = {"busy": False}

# Track which sections have been completed
completed_sections = set()

async def on_next_click():
    if _running_next["busy"]:
        show_message("Wait", "Already processing, please wait...")
        return
    _running_next["busy"] = True

    try:
        idx = _step["i"]
        section = CONFIG_JSON["sections"][idx]
        sec_title = section.get("title", f"Step {idx+1}")

        # ✅ If this section already completed, just advance
        if sec_title in completed_sections:
            if idx < len(ordered_containers) - 1:
                show_step(idx + 1)
            else:
                # 🚀 Go to /result when Finish is clicked
                # disable Next to avoid duplicate triggers and navigate client-side (safe from background tasks)
                try:
                    next_button.disable()
                except Exception:
                    pass
                ui.run_javascript("window.location.href = '/result';")

            return

        status.set_text(f"Processing {sec_title} ... ⏳")

        # Process each field in this section
        for fld in section.get("fields", []):
            fid = fld["id"]

            # Special case: scraping for business URL
            if fid == "business_site_url" and state_values.get("business_site_url"):
                await handle_scrape_and_set("business_site_url")

            # Resolve dependencies
            for dep in fld.get("depends_on", []):
                if dep == "scraped_text" and not state_values.get("scraped_text"):
                    await handle_scrape_and_set("business_site_url")
                await resolve_dependents_for(dep)

            # Resolve the field itself
            await resolve_dependents_for(fid)

        # ✅ Mark as complete
        completed_sections.add(sec_title)

        # Next step or finish
        if idx < len(ordered_containers) - 1:
            show_step(idx + 1)
        else:
            next_button.set_text("Finish")
            show_message("Done", "All steps completed 🎉")
            # 🚀 Redirect when Finish is clicked
            # disable Next to avoid duplicate triggers and navigate client-side (safe from background tasks)
            try:
                next_button.disable()
            except Exception:
                pass
            ui.run_javascript("window.location.href = '/result';")


    except Exception as e:
        show_message("Error", str(e))
    finally:
        _running_next["busy"] = False



# create one global Next button (place it where you want it on the page)
next_button = ui.button(
    "➡ Next",
    on_click=lambda: asyncio.create_task(on_next_click())
).classes("w-full bg-black text-white mt-4 p-3 rounded-lg")

# initialize: show only first step
show_step(0)





# Add Save JSON button
def save_state_to_file():
    payload = {
        "values": state_values,
        "options": state_options
    }
    fname = f"{state_values.get('business_name','business')}_dynamic_output.json".replace(" ", "_")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    status.set_text(f"Saved to {fname}")

ui.button(
    "💾 Save JSON",
    on_click=lambda: save_state_to_file()
).props("flat").classes("w-full bg-gray-800 text-white mt-2 p-3 rounded-lg")



# --------------------------
# Special handlers & dependency resolution
# --------------------------
async def handle_scrape_and_set(field_id: str):
    """
    If business_site_url changed, scrape homepage + a few internal pages,
    set 'scraped_text' and 'business_name' in state_values, then resolve dependents.
    """
    url = state_values.get(field_id, "").strip()
    if not url:
        status.set_text("No URL to scrape")
        return
    status.set_text("Scraping site (homepage + a few internal pages)... ⏳")
    try:
        r = session.get(url, timeout=10); r.raise_for_status()
        homepage_html = r.text
        soup = BeautifulSoup(homepage_html, "html.parser")
        title_tag = soup.title.string.strip() if soup.title and soup.title.string else ""
        business_name = title_tag.split("|")[0].strip() if title_tag else urlparse(url).netloc
        # build combined text
        combined = soup.get_text(separator="\n", strip=True)
        # grab relevant links and append up to 4 internal pages
        relevant = discover_relevant_links(url, homepage_html=homepage_html)
        for i, (k, link) in enumerate(list(relevant.items())[:4]):
            txt = scrape_text_from_url(link)
            if txt:
                combined += "\n\n" + txt
        # store scraped_text and business_name
        state_values["scraped_text"] = combined
        state_values["business_name"] = business_name
        status.set_text(f"Scraped site, extracted business_name='{business_name}'")
        # auto-fill a visible business_name widget if one exists
        if "business_name" in widgets:
            try:
                widgets["business_name"].value = business_name
            except Exception:
                pass
    except Exception as e:
        state_values["scraped_text"] = ""
        status.set_text(f"Scrape failed: {e}")
    # after scraping, resolve dependents of 'scraped_text' and of the URL itself
    await resolve_dependents_for("scraped_text")
    await resolve_dependents_for(field_id)

# keep this set at the top of your file (global)
_in_flight_fields: set[str] = set()

async def resolve_dependents_for(changed_field_id: str):
    """
    Resolve all fields that depend on changed_field_id.
    Each dependent will only be processed if all its dependencies are ready.
    Marks the section as complete once all its fields have values,
    and enables/disables the Next button accordingly.
    """
    dependents = dependency_map.get(changed_field_id, [])
    for dep_field_id in dependents:
        # --- skip if already in-flight (prevents duplicates)
        if dep_field_id in _in_flight_fields:
            continue
        _in_flight_fields.add(dep_field_id)

        try:
            # --- locate field config
            target_cfg = None
            for sec in CONFIG_JSON["sections"]:
                for f in sec["fields"]:
                    if f["id"] == dep_field_id:
                        target_cfg = f
                        break
                if target_cfg:
                    break
            if not target_cfg:
                continue

            # --- check dependencies
            deps = target_cfg.get("depends_on", [])
            ready = True
            for d in deps:
                v = state_values.get(d, None)
                if v:
                    continue
                if d in state_options and state_options[d]:
                    continue
                if d == "scraped_text" and state_values.get("scraped_text", ""):
                    continue
                ready = False
                break
            if not ready:
                continue

            # --- build prompt
            prompt_template = target_cfg.get("prompt", "")
            prompt_text = build_prompt_from_template(prompt_template)

            # --- dedupe by prompt hash (skip if identical prompt already produced a result)
            prompt_hash = hashlib.sha256((prompt_text or "").encode("utf-8")).hexdigest()
            prev_hash = _last_prompt_hash.get(dep_field_id)
            has_existing_result = bool(state_values.get(dep_field_id)) or bool(state_options.get(dep_field_id))

            # if prompt unchanged and we already have a result, skip LLM
            if prev_hash == prompt_hash and has_existing_result:
                print(f"[DEBUG] Skipping {dep_field_id}: prompt unchanged and result exists")
                # ensure section is visible and continue to resolve downstream deps
                for sec in CONFIG_JSON["sections"]:
                    for f in sec["fields"]:
                        if f["id"] == dep_field_id:
                            try:
                                section_containers[sec["title"]].style("display:block")
                            except Exception:
                                pass
                continue

            # mark as in-flight to avoid concurrent duplicate processing (already present in your loop)
            _in_flight_fields.add(dep_field_id)

            # --- call LLM
            target_type = target_cfg.get("type", "text")
            start_progress(f"Running LLM for {dep_field_id} ...")
            try:
                result_text = await call_llm_text(prompt_text)
            finally:
                stop_progress()

            # record the prompt hash after successful call (even empty result)
            _last_prompt_hash[dep_field_id] = prompt_hash


            # --- handle text / textarea
            if target_type in ("text", "textarea"):
                # store raw trimmed result
                cleaned = (result_text or "").strip()
                state_values[dep_field_id] = cleaned

                # debug: show what LLM returned in server console
                print(f"[DEBUG] LLM result for {dep_field_id!r}: {repr(cleaned)[:800]}")

                widget = widgets.get(dep_field_id)
                if widget:
                    try:
                        # ✅ update textarea value
                        widget.value = cleaned

                        # ✅ auto-resize (hug text height)
                        ui.run_javascript(f"""
                        const el = document.querySelector('#{widget.id} textarea');
                        if (el) {{
                            el.style.height = 'auto';
                            el.style.height = el.scrollHeight + 'px';
                        }}
                        """)
                    except Exception as e:
                        status.set_text(f"Update failed for {dep_field_id}: {e}")
                        print("Widget update error:", e)
                else:
                    # fallback: render inside section if widget missing
                    print(f"[WARN] widget for {dep_field_id} not found in widgets dict. Rendering fallback output.")
                    # for sec in CONFIG_JSON["sections"]:
                    #     for f in sec["fields"]:
                    #         if f["id"] == dep_field_id:
                    #             try:
                    #                 with section_containers[sec["title"]]:
                    #                     ui.textarea(value=cleaned, readonly=True) \
                    #                         .props("filled autogrow") \
                    #                         .classes("w-full text-sm text-gray-700")
                    #             except Exception:
                    #                 pass

                # unhide the section this field belongs to
                for sec in CONFIG_JSON["sections"]:
                    for f in sec["fields"]:
                        if f["id"] == dep_field_id:
                            try:
                                section_containers[sec["title"]].style("display:block")
                            except Exception:
                                pass

                # recurse to resolve deeper dependents
                await resolve_dependents_for(dep_field_id)


            # --- handle button_list
            elif target_type == "button_list":
                options = parse_list_response(result_text)
                state_options[dep_field_id] = options
                container = widgets.get(dep_field_id)
                if container:
                    try:
                        container.clear()
                    except Exception:
                        pass

                    with container:
                        if not options:
                            ui.label("No options generated.")
                        else:
                            base_classes = "w-full py-2 px-4 mb-2 rounded-lg transition"
                            default_classes = "bg-gray-800 text-white hover:bg-gray-700"
                            selected_classes = "bg-blue-700 text-white"

                            for opt in options:
                                def make_btn_click(_fid=dep_field_id, _opt=opt, _opts=options, _container=container):
                                    async def _on_click():
                                        state_values[_fid] = _opt
                                        status.set_text(f"Selected: {_opt}")

                                        try:
                                            _container.clear()
                                        except Exception:
                                            pass
                                        with _container:
                                            for o in _opts:
                                                cls = base_classes + " " + (selected_classes if o == _opt else default_classes)
                                                def mk(_o=o):
                                                    return make_btn_click(_fid=_fid, _opt=_o, _opts=_opts, _container=_container)
                                                ui.button(o, on_click=mk()).props("flat").classes(cls)

                                        await resolve_dependents_for(_fid)

                                        # auto-advance to next section
                                        current_idx = None
                                        for idx_s, s in enumerate(CONFIG_JSON["sections"]):
                                            if any(fld["id"] == _fid for fld in s["fields"]):
                                                current_idx = idx_s
                                                break
                                        if current_idx is not None and current_idx < len(ordered_containers) - 1:
                                            show_step(current_idx + 1)
                                            try:
                                                next_button.enable()
                                            except Exception:
                                                pass
                                    return _on_click

                                ui.button(opt, on_click=make_btn_click()).props("flat").classes(f"{base_classes} {default_classes}")

            # --- handle unknown type
            else:
                state_values[dep_field_id] = (result_text or "").strip()
                w = widgets.get(dep_field_id)
                if w:
                    try:
                        w.value = state_values[dep_field_id]
                    except Exception:
                        pass
                await resolve_dependents_for(dep_field_id)

        finally:
            # ✅ always clear in-flight flag so the same field can run again later
            _in_flight_fields.discard(dep_field_id)

    # --- after processing dependents, check if current section is complete
    for sec in CONFIG_JSON["sections"]:
        if any(f["id"] == changed_field_id for f in sec["fields"]):
            complete = True
            for fld in sec["fields"]:
                fid = fld["id"]
                if fld["type"] in ("text", "textarea"):
                    if not state_values.get(fid, "").strip():
                        complete = False
                elif fld["type"] == "button_list":
                    if not state_options.get(fid) and not state_values.get(fid):
                        complete = False
            if complete:
                next_button.enable()
                status.set_text(f"✅ Section '{sec['title']}' is complete")
            else:
                next_button.disable()
            break


# Wire manual trigger: let user click a Run/Refresh button for the entire form (optional)
async def run_all_resolvers():
    # trigger resolution for all fields that have depends_on defined (attempt in order)
    for sec in CONFIG_JSON["sections"]:
        for f in sec["fields"]:
            for dep in f.get("depends_on", []):
                # if dep is scraped_text and scraped_text not present but business_site_url present, do scraping
                if dep == "scraped_text" and not state_values.get("scraped_text", "") and state_values.get("business_site_url", ""):
                    await handle_scrape_and_set("business_site_url")
                await resolve_dependents_for(dep)

ui.button("Run / Refresh All", on_click=lambda: asyncio.create_task(run_all_resolvers()))

# Set initial focus or instructions
status.set_text("Enter a Business Site URL and the form will auto-generate results.")

# --------------------------
# Start NiceGUI app
# --------------------------
if __name__ in {"__main__", "__mp_main__"}:
    print(">>> Starting UI now...")
    ui.run(title=CONFIG_JSON.get("title", "Dynamic App"), port=8080)
