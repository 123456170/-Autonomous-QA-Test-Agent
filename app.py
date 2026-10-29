import io
import os
import json
import time
import uuid
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


st.set_page_config(
    page_title="Autonomous QA Test Agent",
    page_icon="🧪",
    layout="wide",
)


ALLOWED_HOSTS = {
    "todomvc-demo.staging.internal",
    "conduit-demo.staging.internal",
    "juice-shop-demo.staging.internal",
    "localhost",
    "127.0.0.1",
}


TEST_1 = """from playwright.sync_api import sync_playwright, expect


def test_completed_todos_hidden_after_reload():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("https://todomvc-demo.staging.internal")

        page.fill("#new-todo", "Synthetic milk")
        page.press("#new-todo", "Enter")
        page.click("ul.todo-list li:nth-child(1) .toggle")
        page.reload()

        healed = page.locator("[data-testid='todo-item'][data-status='complete']")
        expect(healed).to_have_count(0)

        browser.close()
"""


TEST_2 = """from playwright.sync_api import sync_playwright, expect


def test_signup_rejects_invalid_email():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("https://conduit-demo.staging.internal/register")

        page.fill("input[name='username']", "synthetic_user")
        page.fill("input[name='email']", "not-an-email")
        page.fill("input[name='password']", "SyntheticPass123!")

        healed_submit = page.locator("[data-testid='register-submit']")
        healed_submit.click()

        expect(page.get_by_text("Enter a valid email")).to_be_visible()

        browser.close()
"""


TEST_3 = """from playwright.sync_api import sync_playwright, expect


def test_checkout_coupon_changes_total():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("https://juice-shop-demo.staging.internal/checkout")

        page.click("button.add-to-cart")
        page.click("a[href='/cart']")

        healed_coupon = page.locator("[data-testid='coupon-input']")
        healed_coupon.fill("SAVE10")
        page.click("#applyCoupon")

        total = page.locator("div.order-total")
        expect(total).to_contain_text("$90.00")

        browser.close()
"""


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "TodoMVC: completed items reappear after reload": {
        "url": "https://todomvc-demo.staging.internal",
        "elements": [
            {"selector": "#new-todo", "role": "textbox", "name": "What needs to be done?"},
            {"selector": "ul.todo-list li .toggle", "role": "checkbox", "name": "Toggle todo"},
            {"selector": "a[href='#/active']", "role": "link", "name": "Active"},
            {"selector": "button.clear-completed", "role": "button", "name": "Clear completed"},
        ],
        "bug": {
            "title": "Completed todos reappear after page reload",
            "severity": "High",
            "expected": "Completed todos stay hidden when the Active filter is selected or after reload.",
            "actual": "After marking a todo complete and reloading, the completed todo is visible again.",
            "console_errors": [
                "Uncaught TypeError: cannot read property 'completed' of undefined",
                "Warning: todo state hydration mismatch detected",
            ],
        },
        "repro_steps": [
            "Open the TodoMVC staging URL.",
            "Add a synthetic todo named 'Synthetic milk'.",
            "Mark the todo as complete.",
            "Reload the page.",
            "Observe that the completed item is visible again in the Active view.",
        ],
        "test_name": "test_completed_todos_hidden_after_reload.py",
        "test_code": TEST_1,
        "steps": [
            {"type": "open"},
            {"type": "list"},
            {
                "type": "action",
                "selector": "#new-todo",
                "action": "fill",
                "value": "Synthetic milk",
                "message": "Fill new todo input with synthetic data",
            },
            {
                "type": "action",
                "selector": "#new-todo",
                "action": "press",
                "value": "Enter",
                "message": "Submit synthetic todo",
            },
            {
                "type": "action",
                "selector": "ul.todo-list li:nth-child(1) .toggle",
                "action": "click",
                "value": None,
                "message": "Mark first todo complete",
            },
            {
                "type": "heal",
                "original": "ul.todo-list li.completed",
                "failed": "locator resolved to 0 elements",
                "healed": "[data-testid='todo-item'][data-status='complete']",
                "strategy": "data-testid + state attribute fallback",
                "confidence": 0.91,
                "message": "Self-healed completed-item selector",
            },
            {
                "type": "reload",
                "message": "Reload page to verify persistence",
            },
            {
                "type": "assert",
                "condition": "completed item is not visible after reload in Active view",
                "passed": False,
                "message": "Assertion failed: completed item reappeared",
            },
            {
                "type": "gate",
                "selector": "a[href='/settings/account']",
                "action": "click",
                "destructive": True,
                "message": "Navigate to account settings (potentially destructive)",
            },
            {
                "type": "test",
                "message": "Generate regression test",
            },
        ],
    },
    "Signup: missing email validation": {
        "url": "https://conduit-demo.staging.internal/register",
        "elements": [
            {"selector": "input[name='username']", "role": "textbox", "name": "Username"},
            {"selector": "input[name='email']", "role": "textbox", "name": "Email"},
            {"selector": "input[name='password']", "role": "textbox", "name": "Password"},
            {"selector": "button[type='submit']", "role": "button", "name": "Sign Up"},
            {"selector": "[data-testid='register-submit']", "role": "button", "name": "Register submit"},
        ],
        "bug": {
            "title": "Signup form accepts invalid email address",
            "severity": "Medium",
            "expected": "The form blocks submission and shows an inline validation error for invalid email.",
            "actual": "The invalid email 'not-an-email' is accepted and the signup request is attempted.",
            "console_errors": [
                "POST /users 422 Unprocessable Entity",
                "Unhandled promise rejection in form submit handler",
            ],
        },
        "repro_steps": [
            "Open the registration page.",
            "Enter a valid username and password.",
            "Enter an invalid email address.",
            "Click the signup button.",
            "Observe that no inline validation error is shown and the request is attempted.",
        ],
        "test_name": "test_signup_rejects_invalid_email.py",
        "test_code": TEST_2,
        "steps": [
            {"type": "open"},
            {"type": "list"},
            {
                "type": "action",
                "selector": "input[name='username']",
                "action": "fill",
                "value": "synthetic_user",
                "message": "Fill synthetic username",
            },
            {
                "type": "action",
                "selector": "input[name='email']",
                "action": "fill",
                "value": "not-an-email",
                "message": "Fill intentionally invalid email",
            },
            {
                "type": "action",
                "selector": "input[name='password']",
                "action": "fill",
                "value": "SyntheticPass123!",
                "message": "Fill synthetic password",
            },
            {
                "type": "action",
                "selector": "button[type='submit']",
                "action": "click",
                "value": None,
                "message": "Attempt signup with invalid email",
            },
            {
                "type": "heal",
                "original": "button[type='submit']",
                "failed": "element is not stable / intercepted by overlay",
                "healed": "[data-testid='register-submit']",
                "strategy": "role + accessible name + data-testid fallback",
                "confidence": 0.88,
                "message": "Self-healed signup submit selector",
            },
            {
                "type": "assert",
                "condition": "inline validation error is shown for invalid email",
                "passed": False,
                "message": "Assertion failed: no validation error was displayed",
            },
            {
                "type": "gate",
                "selector": "button[data-action='delete-account']",
                "action": "click",
                "destructive": True,
                "message": "Delete synthetic account (destructive cleanup)",
            },
            {
                "type": "test",
                "message": "Generate regression test",
            },
        ],
    },
    "Checkout: coupon ignored in total": {
        "url": "https://juice-shop-demo.staging.internal/checkout",
        "elements": [
            {"selector": "button.add-to-cart", "role": "button", "name": "Add to cart"},
            {"selector": "a[href='/cart']", "role": "link", "name": "Cart"},
            {"selector": "#couponCode", "role": "textbox", "name": "Coupon code"},
            {"selector": "#applyCoupon", "role": "button", "name": "Apply coupon"},
            {"selector": "div.order-total", "role": "text", "name": "Order total"},
        ],
        "bug": {
            "title": "Checkout total ignores coupon SAVE10",
            "severity": "High",
            "expected": "Applying coupon SAVE10 reduces the order total by 10%.",
            "actual": "The coupon is accepted visually, but the order total remains unchanged.",
            "console_errors": [
                "Coupon applied event emitted but pricing service returned stale total",
                "TypeError: discount is undefined",
            ],
        },
        "repro_steps": [
            "Open the checkout staging URL.",
            "Add a synthetic item to the cart.",
            "Navigate to the cart.",
            "Apply coupon SAVE10.",
            "Observe that the order total does not change.",
        ],
        "test_name": "test_checkout_coupon_changes_total.py",
        "test_code": TEST_3,
        "steps": [
            {"type": "open"},
            {"type": "list"},
            {
                "type": "action",
                "selector": "button.add-to-cart",
                "action": "click",
                "value": None,
                "message": "Add synthetic item to cart",
            },
            {
                "type": "action",
                "selector": "a[href='/cart']",
                "action": "click",
                "value": None,
                "message": "Open cart page",
            },
            {
                "type": "heal",
                "original": "#couponCode",
                "failed": "selector matched hidden legacy input",
                "healed": "[data-testid='coupon-input']",
                "strategy": "visible input + data-testid fallback",
                "confidence": 0.93,
                "message": "Self-healed coupon input selector",
            },
            {
                "type": "action",
                "selector": "[data-testid='coupon-input']",
                "action": "fill",
                "value": "SAVE10",
                "message": "Apply synthetic coupon",
            },
            {
                "type": "action",
                "selector": "#applyCoupon",
                "action": "click",
                "value": None,
                "message": "Trigger coupon application",
            },
            {
                "type": "assert",
                "condition": "order total reflects 10% discount after coupon",
                "passed": False,
                "message": "Assertion failed: total did not change after coupon",
            },
            {
                "type": "gate",
                "selector": "button.place-order",
                "action": "click",
                "destructive": True,
                "message": "Place synthetic order (side-effectful checkout action)",
            },
            {
                "type": "test",
                "message": "Generate regression test",
            },
        ],
    },
}


@st.cache_resource
def get_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_allowed(target_url: str) -> bool:
    try:
        host = urlparse(target_url).hostname or ""
    except Exception:
        return False
    return host in ALLOWED_HOSTS or host.endswith(".staging.internal")


def make_screenshot(
    title: str,
    url: str,
    lines: List[str],
    footer: str,
    tone: str = "info",
) -> Image.Image:
    width, height = 960, 540
    img = Image.new("RGB", (width, height), "#0b1220")
    draw = ImageDraw.Draw(img)

    font_small = get_font(14)
    font = get_font(16)
    font_big = get_font(22)

    draw.rectangle([0, 0, width, 64], fill="#111827")

    for i, color in enumerate(["#ef4444", "#f59e0b", "#22c55e"]):
        draw.ellipse([18 + i * 24, 22, 30 + i * 24, 34], fill=color)

    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle([100, 16, 860, 48], radius=10, fill="#1f2937")
    else:
        draw.rectangle([100, 16, 860, 48], fill="#1f2937")

    draw.text((112, 24), str(url)[:96], fill="#e5e7eb", font=font_small)
    draw.text((20, 72), str(title)[:80], fill="#f9fafb", font=font_big)

    tone_map = {
        "ok": ("#166534", "#dcfce7"),
        "warn": ("#6d28d9", "#ede9fe"),
        "fail": ("#b91c1c", "#fee2e2"),
        "pending": ("#b45309", "#fef3c7"),
        "info": ("#1d4ed8", "#dbeafe"),
    }
    border, _ = tone_map.get(tone, tone_map["info"])

    draw.rectangle([18, 104, width - 18, height - 62], fill="#f8fafc", outline=border, width=3)

    y = 126
    for line in lines[:11]:
        text = str(line)
        color = "#111827"
        if text.startswith("BUG") or text.startswith("Result: FAIL"):
            color = "#b91c1c"
        elif text.startswith("OK") or text.startswith("Status: executed"):
            color = "#15803d"
        elif text.startswith("Healed") or text.startswith("Confidence"):
            color = "#6d28d9"
        elif text.startswith("Human gate"):
            color = "#b45309"

        draw.text((38, y), text[:112], fill=color, font=font)
        y += 28

    draw.rectangle([0, height - 54, width, height], fill="#111827")
    draw.text((18, height - 38), str(footer)[:120], fill="#93c5fd", font=font_small)

    return img


def generate_gif(frames: List[Tuple[Image.Image, str]]) -> bytes:
    buf = io.BytesIO()
    images = [img.convert("RGB") for img, _ in frames]
    images[0].save(
        buf,
        format="GIF",
        append_images=images[1:],
        save_all=True,
        duration=800,
        loop=0,
    )
    return buf.getvalue()


def generate_video(frames: List[Tuple[Image.Image, str]]) -> Optional[Dict[str, Any]]:
    if not frames:
        return None

    if len(frames) > 14:
        step = max(1, len(frames) // 14)
        frames = frames[::step][:14]

    if cv2 is None or np is None:
        try:
            return {
                "bytes": generate_gif(frames),
                "mime": "image/gif",
                "ext": ".gif",
            }
        except Exception:
            return None

    candidates = [
        ("vp80", ".webm", "video/webm"),
        ("avc1", ".mp4", "video/mp4"),
        ("mp4v", ".mp4", "video/mp4"),
    ]

    for fourcc, ext, mime in candidates:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        path = tmp.name
        tmp.close()

        try:
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*fourcc),
                8,
                (960, 540),
            )

            if not writer.isOpened():
                if os.path.exists(path):
                    os.unlink(path)
                continue

            try:
                for img, _ in frames:
                    arr = np.asarray(img.convert("RGB"))
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    for _ in range(6):
                        writer.write(bgr)
            finally:
                writer.release()

            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, "rb") as f:
                    data = f.read()
                os.unlink(path)
                return {
                    "bytes": data,
                    "mime": mime,
                    "ext": ext,
                }

            if os.path.exists(path):
                os.unlink(path)

        except Exception:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
            continue

    try:
        return {
            "bytes": generate_gif(frames),
            "mime": "image/gif",
            "ext": ".gif",
        }
    except Exception:
        return None


def display_video(container, video: Optional[Dict[str, Any]], key_prefix: str) -> None:
    if not video or not video.get("bytes"):
        container.info("Video artifact will appear after the live run completes.")
        return

    mime = video.get("mime", "video/mp4")
    ext = video.get("ext", ".mp4")

    if mime.startswith("video/"):
        container.video(video["bytes"], format=mime)
    else:
        container.image(video["bytes"])

    run_id = st.session_state.get("run_id", "run")
    container.download_button(
        label="Download demo video",
        data=video["bytes"],
        file_name=f"autonomous-qa-demo-{run_id}{ext}",
        mime=mime,
        key=f"{key_prefix}_download_video",
    )


def init_state() -> None:
    defaults = {
        "started": False,
        "result": None,
        "events": [],
        "screenshots": {},
        "pending": [],
        "video": None,
        "scenario": None,
        "run_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state(scenario_name: str) -> None:
    st.session_state.started = False
    st.session_state.result = None
    st.session_state.events = []
    st.session_state.screenshots = {}
    st.session_state.pending = []
    st.session_state.video = None
    st.session_state.scenario = scenario_name
    st.session_state.run_id = None


def render_completed_live(status, video_ph, result: Dict[str, Any]) -> None:
    status.update(
        label=f"Run complete · {len(result.get('bugs', []))} bug(s) · {result.get('latency_seconds', 0)}s",
        state="complete",
        expanded=False,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bugs found", len(result.get("bugs", [])))
    c2.metric("Actions", len(result.get("actions", [])))
    c3.metric("States visited", len(result.get("visited_states", [])))
    c4.metric("Latency", f"{result.get('latency_seconds', 0)}s")

    display_video(video_ph, st.session_state.get("video"), "live_completed")


def apply_pending_decision(
    idx: int,
    approved: bool,
    allow_destructive: bool,
    dry_run: bool,
) -> None:
    pending = st.session_state.get("pending", [])
    if idx < 0 or idx >= len(pending):
        return

    gate = pending[idx]
    if gate.get("handled"):
        return

    if approved:
        if gate.get("destructive") and not allow_destructive:
            decision = "blocked_by_guardrail"
            status = "blocked"
            message = "Destructive action blocked by guardrail."
        elif dry_run:
            decision = "simulated"
            status = "simulated"
            message = "Dry run: approved action was simulated without real side effects."
        else:
            decision = "approved"
            status = "executed"
            message = "Approved action executed in sandbox."
    else:
        decision = "rejected"
        status = "rejected"
        message = "Human rejected pending action."

    gate["handled"] = True
    gate["decision"] = decision

    evidence_id = f"EV-{len(st.session_state.events) + 1:03d}"
    event = {
        "timestamp": now_iso(),
        "node": "human_gate",
        "tool": "guardrail.decision",
        "status": status,
        "message": message,
        "evidence_id": evidence_id,
        "gate_id": gate.get("id"),
        "decision": decision,
    }
    st.session_state.events.append(event)

    result = st.session_state.get("result")
    if result:
        result["actions"].append(
            {
                "index": len(result["actions"]) + 1,
                "tool": "guardrail.decision",
                "args": {
                    "gate_id": gate.get("id"),
                    "decision": decision,
                },
                "status": status,
                "evidence_id": evidence_id,
                "timestamp": now_iso(),
            }
        )
        st.session_state.result = result


def render_pending(container, allow_destructive: bool, dry_run: bool) -> None:
    pending = st.session_state.get("pending", [])
    if not pending:
        return

    container.subheader("Human gate decisions")
    container.caption("Nothing with side effects is executed until a human decision is recorded.")

    for idx, gate in enumerate(pending):
        c1, c2, c3, c4 = container.columns([4, 1, 1, 2])
        c1.markdown(f"**{gate.get('id')}** · {gate.get('message')}")
        c1.caption(
            f"Destructive: {gate.get('destructive')} · "
            f"Handled: {gate.get('handled')} · "
            f"Decision: {gate.get('decision', 'pending')}"
        )

        if not gate.get("handled"):
            if c2.button("Approve", key=f"approve_{idx}_{gate.get('id')}"):
                apply_pending_decision(idx, True, allow_destructive, dry_run)
                st.rerun()

            if c3.button("Reject", key=f"reject_{idx}_{gate.get('id')}"):
                apply_pending_decision(idx, False, allow_destructive, dry_run)
                st.rerun()
        else:
            c4.metric("State", gate.get("decision", "handled"))


def render_bug_report(result: Optional[Dict[str, Any]]) -> None:
    if not result:
        st.info("Run the simulation to generate a bug report.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bugs", len(result.get("bugs", [])))
    c2.metric("Actions", len(result.get("actions", [])))
    c3.metric("States", len(result.get("visited_states", [])))
    c4.metric("Tests", len(result.get("generated_tests", [])))
    c5.metric("Cost", f"${result.get('cost_usd', 0.0):.2f}")

    st.divider()
    st.subheader("Detected bugs")

    bugs = result.get("bugs", [])
    if not bugs:
        st.success("No bugs were detected in this run.")
    else:
        df = pd.DataFrame(
            [
                {
                    "Title": b.get("title"),
                    "Severity": b.get("severity"),
                    "Expected": b.get("expected"),
                    "Actual": b.get("actual"),
                    "Screenshot": b.get("screenshot"),
                }
                for b in bugs
            ]
        )
        st.dataframe(df, use_container_width=True)

        for bug in bugs:
            st.markdown(f"### {bug.get('title')}")
            st.write(f"**Severity:** {bug.get('severity')}")

            st.write("**Reproduction steps:**")
            for i, step in enumerate(bug.get("repro_steps", []), start=1):
                st.write(f"{i}. {step}")

            st.write("**Expected:**", bug.get("expected"))
            st.write("**Actual:**", bug.get("actual"))
            st.write("**Console errors:**", ", ".join(bug.get("console_errors", [])) or "none")

            screenshot_id = bug.get("screenshot")
            img = st.session_state.get("screenshots", {}).get(screenshot_id)
            if img:
                st.image(img, caption=f"Bug evidence · {screenshot_id}", use_container_width=True)

            st.divider()

    st.subheader("Generated Playwright tests")
    tests = result.get("generated_tests", [])
    if not tests:
        st.info("No tests were generated.")
    else:
        for test in tests:
            st.markdown(f"**{test.get('name')}**")
            st.code(test.get("code", ""), language="python")

    st.subheader("Full JSON report")
    st.json(result)

    st.download_button(
        label="Download JSON report",
        data=json.dumps(result, indent=2).encode("utf-8"),
        file_name="autonomous-qa-report.json",
        mime="application/json",
        key="json_report_download",
    )


def render_audit_log() -> None:
    events = st.session_state.get("events", [])
    if not events:
        st.info("No audit events yet.")
        return

    df = pd.DataFrame(events)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="Download audit CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="autonomous-qa-audit.csv",
        mime="text/csv",
        key="audit_csv_download",
    )

    st.download_button(
        label="Download audit JSON",
        data=json.dumps(events, indent=2).encode("utf-8"),
        file_name="autonomous-qa-audit.json",
        mime="application/json",
        key="audit_json_download",
    )


def render_about() -> None:
    st.markdown(
        """
        ### What this demo does
        This single-file Streamlit app simulates an **Autonomous QA Test Agent** that:
        - explores a web app,
        - discovers interactive elements,
        - performs synthetic user actions,
        - detects UI bugs,
        - self-heals fragile selectors,
        - generates Playwright regression tests,
        - and produces a bug report with screenshots.

        ### Important safety model
        - No real browser is launched in this demo.
        - No external API key is required.
        - The run is a realistic local simulation designed to demonstrate the product behavior.
        - The generated video is produced inside the app from simulated screenshots.

        ### Guardrails shown in the UI
        - URL allow-list enforcement.
        - Destructive actions require an explicit allow flag.
        - Dry-run mode simulates side effects instead of executing them.
        - Human gate captures approve/reject decisions.
        - Audit log records every node, tool, status, and evidence id.

        ### Self-healing selectors
        The signature capability is visible in the audit log and screenshots:
        - original selector fails,
        - agent records the failure reason,
        - fallback strategy is selected,
        - healed selector is emitted with confidence,
        - later actions use the healed selector.
        """
    )

    st.warning(
        "This app is a portfolio-grade simulation. For production use, connect it to "
        "Playwright, a real Gemini-compatible model gateway, persistent storage, and "
        "environment-scoped allow-lists."
    )


def run_live(
    status,
    progress,
    log_ph,
    screen_ph,
    video_ph,
    scenario_name: str,
    url: str,
    strictness: float,
    dry_run: bool,
    allow_destructive: bool,
) -> None:
    scenario = SCENARIOS[scenario_name]
    session_id = str(uuid.uuid4())
    st.session_state.run_id = session_id

    start_total = time.time()
    started_at = now_iso()

    events: List[Dict[str, Any]] = []
    screenshots: Dict[str, Image.Image] = {}
    frames: List[Tuple[Image.Image, str]] = []
    visited: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    bugs: List[Dict[str, Any]] = []
    tests: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []

    executable_steps = [s for s in scenario["steps"] if s.get("type") not in ("open", "list")]
    total_steps = len(executable_steps) + 5
    step_counter = 0

    def emit(
        node: str,
        tool: str,
        event_status: str,
        message: str,
        **kwargs: Any,
    ) -> str:
        nonlocal step_counter

        evidence_id = f"EV-{len(events) + 1:03d}"
        event = {
            "timestamp": now_iso(),
            "node": node,
            "tool": tool,
            "status": event_status,
            "message": message,
            "evidence_id": evidence_id,
        }
        event.update(kwargs)
        events.append(event)

        step_counter += 1
        pct = min(100, int(100 * step_counter / max(1, total_steps)))
        progress.progress(pct / 100.0, text=f"{pct}% · {node} · {tool}")

        log_rows = [
            {
                "Time": e["timestamp"][11:19],
                "Node": e["node"],
                "Tool": e["tool"],
                "Status": e["status"],
                "Message": e["message"],
            }
            for e in events[-10:]
        ]
        log_ph.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        status.update(label=f"{node}: {message}", state="running", expanded=True)
        time.sleep(0.32)

        return evidence_id

    def add_screenshot(img: Image.Image, caption: str) -> str:
        screenshot_id = f"SCREEN-{len(screenshots) + 1:03d}"
        screenshots[screenshot_id] = img
        frames.append((img, caption))
        screen_ph.image(img, use_container_width=True, caption=caption)
        return screenshot_id

    allowed = is_allowed(url)
    effective_url = url if allowed else scenario["url"]

    emit(
        "ingest",
        "guardrail.validate_url",
        "blocked" if not allowed else "ok",
        "URL allow-list check",
        url=effective_url,
        allowed=allowed,
    )

    if not allowed:
        emit(
            "ingest",
            "guardrail.fallback",
            "warning",
            "Non-allow-listed URL replaced with staged demo URL",
            original_url=url,
            fallback_url=effective_url,
        )

    img = make_screenshot(
        title=scenario_name,
        url=effective_url,
        lines=[
            "Loading page...",
            "DOM ready",
            "Synthetic session started",
            "No real credentials used",
        ],
        footer="Perception: open_page",
        tone="info",
    )
    screenshot_id = add_screenshot(img, "Initial page state")
    visited.append(
        {
            "state_id": "STATE-001",
            "url": effective_url,
            "title": scenario_name,
            "screenshot": screenshot_id,
        }
    )
    emit(
        "perception",
        "open_page",
        "ok",
        "Page opened and screenshot captured",
        screenshot=screenshot_id,
        url=effective_url,
    )

    elements = scenario.get("elements", [])
    emit(
        "perception",
        "list_interactive_elements",
        "ok",
        f"Discovered {len(elements)} interactive elements",
        element_count=len(elements),
    )

    element_lines = [
        f"{e.get('role')}: {e.get('name')} ({e.get('selector')})"
        for e in elements[:7]
    ]
    img = make_screenshot(
        title="Accessibility tree",
        url=effective_url,
        lines=element_lines or ["No elements discovered"],
        footer="Perception: interactive element inventory",
        tone="info",
    )
    add_screenshot(img, "Interactive elements discovered")

    emit(
        "plan",
        "agent.plan",
        "ok",
        f"Planned synthetic journey with strictness {strictness:.2f}",
        confidence=strictness,
    )

    action_index = 0

    for step in executable_steps:
        step_type = step.get("type")
        action_index += 1

        if step_type == "action":
            selector = step.get("selector")
            action = step.get("action")
            value = step.get("value")

            lines = [
                f"Action: {action}",
                f"Selector: {selector}",
                f"Value: {value if value is not None else ''}",
                "Side effects: sandboxed synthetic data",
            ]
            img = make_screenshot(
                title=scenario_name,
                url=effective_url,
                lines=lines,
                footer=step.get("message", "Action executed"),
                tone="ok",
            )
            screenshot_id = add_screenshot(img, step.get("message", "Action executed"))
            evidence_id = emit(
                "act",
                "perform_action",
                "ok",
                step.get("message", "Action executed"),
                selector=selector,
                action=action,
                value=value,
                screenshot=screenshot_id,
            )
            actions.append(
                {
                    "index": action_index,
                    "tool": "perform_action",
                    "args": {
                        "selector": selector,
                        "action": action,
                        "value": value,
                    },
                    "status": "ok",
                    "evidence_id": evidence_id,
                    "screenshot": screenshot_id,
                    "timestamp": now_iso(),
                }
            )

        elif step_type == "heal":
            original = step.get("original")
            healed = step.get("healed")
            confidence = float(step.get("confidence", 0.0))

            lines = [
                "Original selector failed:",
                str(original),
                f"Failure: {step.get('failed')}",
                "Healed selector:",
                str(healed),
                f"Strategy: {step.get('strategy')}",
                f"Confidence: {confidence:.2f}",
            ]
            img = make_screenshot(
                title="Self-healing selector",
                url=effective_url,
                lines=lines,
                footer=step.get("message", "Selector healed"),
                tone="warn",
            )
            screenshot_id = add_screenshot(img, step.get("message", "Selector healed"))
            evidence_id = emit(
                "self_healing",
                "selector.heal",
                "warning",
                step.get("message", "Selector healed"),
                original_selector=original,
                healed_selector=healed,
                confidence=confidence,
                screenshot=screenshot_id,
            )
            actions.append(
                {
                    "index": action_index,
                    "tool": "selector.heal",
                    "args": {
                        "original": original,
                        "healed": healed,
                        "strategy": step.get("strategy"),
                    },
                    "status": "healed",
                    "evidence_id": evidence_id,
                    "screenshot": screenshot_id,
                    "timestamp": now_iso(),
                }
            )

        elif step_type == "reload":
            lines = [
                "Tool: perform_action",
                "Action: reload",
                "Checking persisted UI state",
            ]
            img = make_screenshot(
                title=scenario_name,
                url=effective_url,
                lines=lines,
                footer=step.get("message", "Reload executed"),
                tone="info",
            )
            screenshot_id = add_screenshot(img, step.get("message", "Reload executed"))
            evidence_id = emit(
                "act",
                "perform_action",
                "ok",
                step.get("message", "Reload executed"),
                selector="page",
                action="reload",
                screenshot=screenshot_id,
            )
            actions.append(
                {
                    "index": action_index,
                    "tool": "perform_action",
                    "args": {
                        "selector": "page",
                        "action": "reload",
                        "value": None,
                    },
                    "status": "ok",
                    "evidence_id": evidence_id,
                    "screenshot": screenshot_id,
                    "timestamp": now_iso(),
                }
            )
            visited.append(
                {
                    "state_id": f"STATE-{len(visited) + 1:03d}",
                    "url": effective_url,
                    "title": "After reload",
                    "screenshot": screenshot_id,
                }
            )

        elif step_type == "assert":
            passed = bool(step.get("passed"))
            condition = step.get("condition")
            tone = "ok" if passed else "fail"

            lines = [
                f"Assertion: {condition}",
                f"Result: {'PASS' if passed else 'FAIL'}",
                "Evidence: screenshot + console buffer",
            ]
            if not passed:
                lines.append("BUG: expected behavior not observed")

            img = make_screenshot(
                title=scenario_name,
                url=effective_url,
                lines=lines,
                footer=step.get("message", "Assertion checked"),
                tone=tone,
            )
            screenshot_id = add_screenshot(img, step.get("message", "Assertion checked"))
            evidence_id = emit(
                "verify",
                "assert_state",
                "pass" if passed else "fail",
                step.get("message", "Assertion checked"),
                condition=condition,
                passed=passed,
                screenshot=screenshot_id,
            )

            if not passed:
                bug_payload = scenario.get("bug", {})
                bugs.append(
                    {
                        "title": bug_payload.get("title", "Unknown bug"),
                        "severity": bug_payload.get("severity", "Medium"),
                        "repro_steps": scenario.get("repro_steps", []),
                        "expected": bug_payload.get("expected", ""),
                        "actual": bug_payload.get("actual", ""),
                        "screenshot": screenshot_id,
                        "console_errors": bug_payload.get("console_errors", []),
                        "evidence_id": evidence_id,
                    }
                )

        elif step_type == "gate":
            lines = [
                "Human gate: waiting for approval",
                str(step.get("message")),
                f"Destructive: {step.get('destructive', False)}",
                "No side effect executed yet",
            ]
            img = make_screenshot(
                title=scenario_name,
                url=effective_url,
                lines=lines,
                footer="Human gate: pending decision",
                tone="pending",
            )
            screenshot_id = add_screenshot(img, "Human gate pending")
            evidence_id = emit(
                "human_gate",
                "guardrail.pending_action",
                "pending",
                step.get("message", "Pending human approval"),
                destructive=step.get("destructive", False),
                screenshot=screenshot_id,
            )
            pending.append(
                {
                    "id": f"GATE-{len(pending) + 1:03d}",
                    "message": step.get("message", "Pending human approval"),
                    "selector": step.get("selector"),
                    "action": step.get("action", "click"),
                    "destructive": bool(step.get("destructive", False)),
                    "side_effect": True,
                    "handled": False,
                    "decision": None,
                    "evidence_id": evidence_id,
                }
            )

        elif step_type == "test":
            test_name = scenario.get("test_name", "generated_test.py")
            code = scenario.get("test_code", "")

            lines = [
                "Generated regression test",
                test_name,
                "Framework: pytest-playwright",
                "Includes self-healing selector fallback",
            ]
            img = make_screenshot(
                title=scenario_name,
                url=effective_url,
                lines=lines,
                footer="Finalize: write_test",
                tone="info",
            )
            screenshot_id = add_screenshot(img, "Generated test artifact")
            evidence_id = emit(
                "finalize",
                "write_test",
                "ok",
                f"Wrote {test_name}",
                test_name=test_name,
                screenshot=screenshot_id,
            )
            tests.append(
                {
                    "name": test_name,
                    "language": "python-pytest-playwright",
                    "code": code,
                    "evidence_id": evidence_id,
                }
            )

    assertions_executed = sum(1 for e in events if e.get("tool") == "assert_state")
    self_healed_selectors = sum(1 for e in events if e.get("node") == "self_healing")

    coverage_estimate = {
        "pages_visited": len(visited),
        "interactive_elements_discovered": len(elements),
        "assertions_executed": assertions_executed,
        "self_healed_selectors": self_healed_selectors,
        "estimated_percent": round(
            min(
                96.0,
                28.0 + len(visited) * 8 + assertions_executed * 5 + self_healed_selectors * 4,
            ),
            1,
        ),
    }

    latency_seconds = round(time.time() - start_total, 2)

    result = {
        "session_id": session_id,
        "started_at": started_at,
        "finished_at": now_iso(),
        "latency_seconds": latency_seconds,
        "cost_usd": 0.0,
        "mode": "local_simulation",
        "target_url": effective_url,
        "visited_states": visited,
        "actions": actions,
        "bugs": bugs,
        "generated_tests": tests,
        "coverage_estimate": coverage_estimate,
        "guardrails": {
            "allow_list_enforced": True,
            "destructive_actions_allowed": allow_destructive,
            "dry_run": dry_run,
            "max_actions_per_session": 200,
            "actions_executed": len(actions),
        },
        "self_healing": {
            "attempts": self_healed_selectors,
            "successes": self_healed_selectors,
            "observability": "Each healing event includes original selector, failure reason, healed selector, strategy, and confidence.",
        },
    }

    video_artifact = generate_video(frames)
    st.session_state.video = video_artifact
    display_video(video_ph, video_artifact, "live_run")

    progress.progress(1.0, text="100% · complete")
    status.update(
        label=f"Run complete · {len(bugs)} bug(s) · {latency_seconds}s",
        state="complete",
        expanded=False,
    )

    st.session_state.result = result
    st.session_state.events = events
    st.session_state.screenshots = screenshots
    st.session_state.pending = pending


init_state()

st.sidebar.title("⚙️ Agent controls")

scenario_names = list(SCENARIOS.keys())
scenario_name = st.sidebar.selectbox(
    "Demo scenario",
    scenario_names,
    index=0,
)
scenario = SCENARIOS[scenario_name]

url = st.sidebar.text_input(
    "Target web app URL",
    scenario["url"],
)
strictness = st.sidebar.slider(
    "Plan strictness",
    min_value=0.0,
    max_value=1.0,
    value=0.72,
    step=0.01,
)
dry_run = st.sidebar.toggle(
    "Dry run",
    value=True,
)
allow_destructive = st.sidebar.toggle(
    "Allow destructive actions",
    value=False,
)
auto_start = st.sidebar.toggle(
    "Auto-start demo",
    value=True,
)

if st.sidebar.button("Reset and run"):
    reset_state(scenario_name)
    st.rerun()

st.sidebar.caption(
    "No API key required. This demo simulates Gemini + Playwright locally "
    "and generates an MP4/WebM/GIF artifact inside the app."
)

if st.session_state.get("scenario") != scenario_name:
    reset_state(scenario_name)

st.title("🧪 Autonomous QA Test Agent")
st.caption(
    "Self-healing selectors · simulated Gemini-style reasoning · Playwright test generation · "
    "bug report with screenshots · live demo video generated in-app"
)

tabs = st.tabs(
    [
        "Live Run",
        "Video Demo",
        "Bug Report",
        "Audit Log",
        "About",
    ]
)

with tabs[0]:
    st.subheader("Live autonomous exploration")

    status = st.status("Idle", expanded=True)
    progress = st.progress(0.0, text="0%")
    log_ph = st.empty()
    screen_ph = st.empty()
    video_ph = st.empty()
    control_ph = st.container()

    if not st.session_state.started and auto_start:
        st.session_state.started = True
        run_live(
            status=status,
            progress=progress,
            log_ph=log_ph,
            screen_ph=screen_ph,
            video_ph=video_ph,
            scenario_name=scenario_name,
            url=url,
            strictness=strictness,
            dry_run=dry_run,
            allow_destructive=allow_destructive,
        )
    elif st.session_state.result:
        render_completed_live(status, video_ph, st.session_state.result)
    else:
        status.update(label="Auto-start disabled. Press Reset and run.", state="idle")
        if st.button("Start simulation"):
            reset_state(scenario_name)
            st.rerun()

    render_pending(control_ph, allow_destructive, dry_run)

with tabs[1]:
    st.subheader("Generated live demo video")
    st.caption(
        "This video is generated from the simulated screenshots produced during the run. "
        "It is not a static mockup."
    )
    display_video(st.container(), st.session_state.get("video"), "video_tab")

with tabs[2]:
    st.subheader("QA bug report")
    render_bug_report(st.session_state.get("result"))

with tabs[3]:
    st.subheader("Audit log")
    render_audit_log()

with tabs[4]:
    st.subheader("About and responsible use")
    render_about()