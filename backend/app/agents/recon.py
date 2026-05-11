"""Production-β.1 — Reconnaissance walker ("Scout this app").

Before any test plan executes for the first time on a new app, the
user can hit "Scout this app" — this module visits the target URL,
walks 2-3 levels deep through the most likely test paths (homepage,
primary nav, search, account surface), and writes structured notes
to the AKB so subsequent runs have a mental model of the app.

Strategy
--------
1. Open the headed (or headless) browser and navigate to ``target_url``.
2. Capture the homepage: AX tree, key affordances (nav, search, login
   link), screenshot, body text.
3. Run a vision call against the screenshot to summarize "what is
   this app?" + identify the auth surface and primary CTAs.
4. Walk a small breadth-first list of links from the homepage:
   nav items, footer links to "About / Help / Contact / Sign in"
   are skipped (low-value); product-y / feature-y links are
   followed one level.
5. STOP at any auth wall (per the user's locked policy v1) — write
   "auth surface at /signin" + skip protected pages.
6. Persist the App Profile + each visited page's note to AKB
   under ``kind="recon_note"``.

Cost
----
Bounded by ``max_pages`` (default 8). Each page = 1 vision call +
small DOM scrape. Total ~8 vision calls + a few text-only LLM calls
+ a few seconds of browser walking. Caches in AKB for ``N`` days
(refresh button rebuilds).

Auth-wall detection
-------------------
Re-uses the screen classifier (``classify_popup`` for v1; will
swap to ``detect_screen_intent`` once Phase 2 lands cleanly).
``required_step`` + URL contains login/signin OR a password input
visible → treat as auth wall.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urljoin, urlparse

if TYPE_CHECKING:
    from playwright.sync_api import Page
    from sqlalchemy.orm import Session

    from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ReconResult:
    target_url: str
    pages_visited: int = 0
    pages: list[dict[str, Any]] = field(default_factory=list)
    auth_surface: str | None = None
    primary_nav_items: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    vision_calls: int = 0


_AUTH_URL_HINTS = (
    "signin", "sign-in", "login", "log-in", "auth", "account/login",
    "session", "oauth", "saml", "okta", "sso",
)
_NAV_KEYWORDS = (
    "home", "products", "categories", "shop", "browse", "search",
    "deals", "today", "menu", "explore",
)


def _norm_host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_same_host(a: str, b: str) -> bool:
    ah, bh = _norm_host(a), _norm_host(b)
    if not ah or not bh:
        return False
    if ah == bh:
        return True
    # Allow www / non-www swap.
    return ah.lstrip("www.") == bh.lstrip("www.")


def run_recon(
    page: "Page",
    db: "Session",
    *,
    target_url: str,
    provider: "LLMProvider | None" = None,
    cheap_provider: "LLMProvider | None" = None,
    max_pages: int = 8,
    emit_event: Callable[[str, dict], None] | None = None,
) -> ReconResult:
    """Scout the app + write AKB chunks.

    Returns a ``ReconResult`` that the caller renders to the user.
    On any exception, returns a partial result with ``error_message``
    set — non-fatal so the user sees what was learned before the
    walker hit a wall.
    """
    out = ReconResult(target_url=target_url)
    if not target_url:
        out.error_message = "target_url is required"
        return out

    visited: set[str] = set()
    queue: list[str] = [target_url]
    home_host = _norm_host(target_url)

    def _emit(t: str, d: dict) -> None:
        if emit_event:
            try:
                emit_event(t, d)
            except Exception:
                pass

    from app.agents.page_intel import (  # noqa: PLC0415
        capture_screenshot_for_vision,
    )
    from app.services.akb import write_chunk  # noqa: PLC0415

    _emit("recon_started", {"target_url": target_url})

    # β.2 — autoload pattern pack BEFORE walking. URL-based detection
    # works pre-navigation; DOM-signature detection runs after the
    # homepage loads (below).
    try:
        from app.agents.patterns import autoload_pack  # noqa: PLC0415

        pack = autoload_pack(db, target_url=target_url)
        if pack is not None:
            out.notes.append(
                f"Pattern pack loaded: {pack.label} "
                f"({len(pack.rules)} rules)",
            )
            _emit("recon_pack_loaded", {
                "pack_id": pack.id,
                "label": pack.label,
                "rule_count": len(pack.rules),
            })
    except Exception as e:
        logger.debug("recon: pattern pack autoload skipped: %s", e)

    while queue and out.pages_visited < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        if not _is_same_host(url, target_url):
            continue
        visited.add(url)

        try:
            page.goto(
                url, wait_until="domcontentloaded", timeout=20_000,
            )
        except Exception as e:
            logger.debug("recon: navigate failed for %s: %s", url, e)
            continue

        # Settle.
        try:
            page.wait_for_load_state("networkidle", timeout=8_000)
        except Exception:
            pass

        try:
            cur_url = page.url
        except Exception:
            cur_url = url

        # Auth-wall detection — heuristic: URL contains a login hint
        # OR there's a visible password input on the page.
        looks_like_auth = any(
            h in cur_url.lower() for h in _AUTH_URL_HINTS
        )
        try:
            has_pwd = page.locator("input[type='password']").count() > 0
        except Exception:
            has_pwd = False
        if looks_like_auth or has_pwd:
            if out.auth_surface is None:
                out.auth_surface = cur_url
                out.notes.append(
                    f"Authentication surface at {cur_url} — protected "
                    f"pages skipped during recon.",
                )
            _emit("recon_auth_skip", {"url": cur_url})
            continue

        # Page summary — small scrape of nav / heading / primary CTAs
        # via the existing page_intel JS. Cheap.
        try:
            from app.agents.page_intel import (  # noqa: PLC0415
                _PAGE_SUMMARY_JS,
            )
            summary = page.evaluate(_PAGE_SUMMARY_JS)
        except Exception:
            summary = {"items": [], "title": ""}

        title = (summary.get("title") if isinstance(summary, dict) else "") or ""
        items = (summary.get("items") if isinstance(summary, dict) else []) or []

        # Pick a few interactive items — links to follow + key
        # affordances to record.
        link_targets: list[str] = []
        primary_cta: list[str] = []
        for it in items[:80]:
            role = (it.get("role") or "").lower()
            name = (it.get("name") or "").strip()
            href = it.get("href") or ""
            if not name:
                continue
            if role == "link" and href:
                # Same-host links only; absolute-ize relative URLs.
                full = urljoin(cur_url, href)
                if (
                    _is_same_host(full, target_url)
                    and full not in visited
                ):
                    if any(
                        k in name.lower() for k in _NAV_KEYWORDS
                    ):
                        link_targets.append(full)
            if role in ("button", "link"):
                if (
                    "search" in name.lower()
                    or "sign" in name.lower()
                    or "cart" in name.lower()
                    or "menu" in name.lower()
                ):
                    primary_cta.append(f"{role}: {name[:60]}")

        # Optional vision-call summary on the homepage (turn 1 only)
        # to enrich AKB with "what is this app".
        screenshot_summary: str = ""
        if (
            provider is not None
            and out.pages_visited == 0
            and getattr(provider, "supports_vision", False)
        ):
            try:
                shot = capture_screenshot_for_vision(page)
                from app.llm.base import ChatMessage  # noqa: PLC0415
                from app.llm.router import (  # noqa: PLC0415
                    LLMRole, call_for_role,
                )

                _SUMMARY_SCHEMA = {
                    "type": "object",
                    "properties": {
                        "app_kind": {"type": "string"},
                        "primary_actions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "gotchas": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "summary": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "app_kind", "primary_actions",
                        "gotchas", "summary", "confidence",
                    ],
                    "additionalProperties": False,
                }
                tiered = call_for_role(
                    strong=provider,
                    cheap=cheap_provider,
                    role=LLMRole.SCREEN_CLASSIFIER,
                    messages=[
                        ChatMessage(
                            role="system",
                            content=(
                                "You are a senior QA scout. Look at "
                                "the homepage screenshot and return a "
                                "structured summary of the application. "
                                "Be concrete; surface common gotchas a "
                                "first-time tester would benefit from "
                                "knowing (e.g. 'this app uses a sticky "
                                "header that intercepts clicks at the "
                                "top of the viewport')."
                            ),
                        ),
                        ChatMessage(
                            role="user",
                            content=(
                                f"PAGE URL: {cur_url}\n"
                                f"PAGE TITLE: {title}\n"
                                "Summarise this application + its "
                                "primary actions + any visible gotchas."
                            ),
                            image=shot,
                        ),
                    ],
                    schema=_SUMMARY_SCHEMA,
                    schema_name="recon_homepage",
                    temperature=0.2,
                    max_output_tokens=512,
                )
                parsed = tiered.chat.parsed or {}
                if isinstance(parsed, dict):
                    screenshot_summary = str(
                        parsed.get("summary") or "",
                    )
                    out.primary_nav_items = list(
                        parsed.get("primary_actions") or [],
                    )[:10]
                    if parsed.get("gotchas"):
                        for g in parsed["gotchas"][:5]:
                            if isinstance(g, str) and g.strip():
                                out.notes.append(g.strip())
                if tiered.chat.input_tokens:
                    out.input_tokens += tiered.chat.input_tokens
                if tiered.chat.output_tokens:
                    out.output_tokens += tiered.chat.output_tokens
                out.vision_calls += 1
            except Exception as e:
                logger.debug("recon vision summary skipped: %s", e)

        # Persist the page note to AKB.
        page_note_parts: list[str] = [
            f"Page: {title or '(untitled)'} ({cur_url})",
        ]
        if primary_cta:
            page_note_parts.append(
                f"Primary CTAs visible: {', '.join(primary_cta[:6])}",
            )
        if screenshot_summary:
            page_note_parts.append(f"Summary: {screenshot_summary}")
        page_note = "\n".join(page_note_parts)
        try:
            write_chunk(
                db,
                target_url_pattern=home_host,
                kind="recon_note",
                content=page_note[:1500],
                tags=["recon", "page"],
            )
        except Exception as e:
            logger.debug("AKB write_chunk failed during recon: %s", e)

        out.pages.append({
            "url": cur_url,
            "title": title[:120],
            "primary_cta": primary_cta[:6],
        })
        out.pages_visited += 1
        _emit("recon_page_visited", {
            "url": cur_url,
            "title": title[:120],
            "visited": out.pages_visited,
            "max_pages": max_pages,
        })

        # Enqueue at most 3 same-host links per page so we breadth-first
        # cover the app rather than tunneling deep.
        for cand in link_targets[:3]:
            if cand not in visited and cand not in queue:
                queue.append(cand)

        # Be polite to the target.
        try:
            time.sleep(0.5)
        except Exception:
            pass

    # Write the App Profile as a single high-confidence chunk.
    if out.pages_visited > 0:
        profile_lines = [
            f"App Profile for {home_host}",
            f"  homepage: {target_url}",
            f"  pages scouted: {out.pages_visited}",
        ]
        if out.auth_surface:
            profile_lines.append(f"  auth surface: {out.auth_surface}")
        if out.primary_nav_items:
            profile_lines.append(
                "  primary actions: "
                + ", ".join(out.primary_nav_items[:6]),
            )
        if out.notes:
            profile_lines.append("  notes:")
            for n in out.notes[:6]:
                profile_lines.append(f"    - {n}")
        try:
            from app.services.akb import write_chunk as _write  # noqa: PLC0415

            _write(
                db,
                target_url_pattern=home_host,
                kind="recon_note",
                content="\n".join(profile_lines)[:1800],
                tags=["recon", "profile"],
                confidence=0.85,
            )
        except Exception as e:
            logger.debug("AKB profile write failed: %s", e)

    _emit("recon_completed", {
        "target_url": target_url,
        "pages_visited": out.pages_visited,
        "auth_surface": out.auth_surface,
    })
    return out
