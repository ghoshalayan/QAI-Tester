"""Pattern pack registry + autoload dispatcher.

Detection precedence (first match wins):
1. Exact URL host match (e.g. ``salesforce.com``)
2. URL substring match (e.g. ``.lightning.force.com``)
3. DOM signature on the live page (e.g. presence of
   ``ui5-shellbar`` element → SAP Fiori)
4. Fall back to family heuristics:
   - ``/admin``, ``/wp-admin`` → generic admin panel
   - ``/cart`` / product listings → generic e-commerce
   - else → generic CMS

The detector is best-effort; pattern packs are augmentative
knowledge, not the agent's primary plan. Missing detection just
means "no pack-level boost"; the agent still has BRD chunks +
recon notes + disputes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class PatternPack:
    """One UX family's knowledge primer."""

    id: str  # short stable id, e.g. "sap_fiori"
    label: str  # human-readable, e.g. "SAP Fiori / UI5"
    # URL substrings that trigger this pack (any-of match).
    url_hints: list[str] = field(default_factory=list)
    # CSS selectors / element queries that, if present on the page,
    # imply this pack. Any-of match.
    dom_signatures: list[str] = field(default_factory=list)
    # Rules pushed into AKB. Each is short + actionable.
    rules: list[str] = field(default_factory=list)
    # Suggested tags applied to AKB chunks from this pack.
    tags: list[str] = field(default_factory=list)


PATTERN_PACKS: list[PatternPack] = [
    PatternPack(
        id="sap_fiori",
        label="SAP Fiori / UI5",
        url_hints=[
            "fiori", "/sap/bc/", "ui5", "hana.ondemand", "s4hana",
        ],
        dom_signatures=[
            "ui5-shellbar", "[data-sap-ui]", "div.sapMShell",
        ],
        tags=["sap", "fiori", "ui5"],
        rules=[
            "SAP Fiori uses Web Components heavily — many click "
            "targets live inside sealed shadow DOM. When DOM "
            "resolution misses, prefer vision_only mode or "
            "coord-click directly.",
            "Fiori shellbar (``ui5-shellbar``) wraps user menu, "
            "search, and notifications. Click the avatar to open "
            "personalisation; click the bell for notifications.",
            "Fiori inputs are ``ui5-input``; the actual native "
            "<input> is inside shadow DOM. Type via coord-click on "
            "the field, then page.keyboard.type — not via Playwright "
            "fill().",
            "Fiori tables (``ui5-table``) row-click toggles "
            "selection; double-click opens the detail. Selection is "
            "via the leading checkbox column.",
            "SAP busy indicators show as a translucent overlay with "
            "a circular spinner. Wait for ``.sapMBusyIndicator`` to "
            "disappear before next action.",
            "ALWAYS use go_back over navigate() on Fiori — back-"
            "stack is shell-managed and direct navigate often "
            "loses state.",
        ],
    ),
    PatternPack(
        id="salesforce_lightning",
        label="Salesforce Lightning Experience",
        url_hints=[
            "salesforce.com", "force.com", "lightning.com",
            "my.salesforce", ".cloudforce.com",
        ],
        dom_signatures=[
            "div.slds-scope", "lightning-button", "[data-aura-class]",
        ],
        tags=["salesforce", "lightning", "lwc"],
        rules=[
            "Salesforce Lightning uses LWC + Aura — many components "
            "render shadow DOM. Tab labels and inline edits often "
            "fail DOM resolution; vision-search rescue works better.",
            "Save / Cancel in Lightning record edit dialogs are "
            "inside a footer with role='dialog' descendant. The "
            "Save button text is exactly 'Save' — use literal text.",
            "Inline-edit a record field by double-clicking the "
            "value display, NOT the field label.",
            "Lightning's global search is in the top-bar; Cmd/Ctrl-K "
            "opens it from anywhere.",
            "Toast notifications (record saved / error) appear in "
            "the top-right and auto-dismiss in ~5s; if you need "
            "to verify a save, snapshot quickly.",
            "Salesforce often takes 2-4s to navigate between record "
            "pages; use wait_for_load_state('networkidle') with a "
            "10s timeout.",
        ],
    ),
    PatternPack(
        id="servicenow",
        label="ServiceNow",
        url_hints=[
            "service-now.com", ".servicenow.com",
        ],
        dom_signatures=[
            "[data-snm]", "iframe#gsft_main", "div.glide-list",
        ],
        tags=["servicenow"],
        rules=[
            "ServiceNow renders most of its content inside the "
            "``gsft_main`` iframe. The selectors.py iframe "
            "penetration handles this; vision-only also works.",
            "List rows have a ``magnifying-glass`` icon for preview, "
            "row-text-click opens the record. Use the icon if the "
            "test wants 'preview', the row if it wants 'open'.",
            "Save is via the form's hamburger menu OR keyboard "
            "shortcut (right-click the form header → Save). The "
            "primary visual button at top-right says 'Update' on "
            "edit and 'Submit' on new record.",
            "ServiceNow uses Catalog Items for service requests. "
            "Order Now → Order Status are the canonical end-states.",
        ],
    ),
    PatternPack(
        id="workday",
        label="Workday",
        url_hints=[
            "workday.com", "myworkday", "wd5.myworkday",
        ],
        dom_signatures=[
            "[data-automation-id]", "div.WMM",
        ],
        tags=["workday"],
        rules=[
            "Workday wraps every actionable element with "
            "``data-automation-id``; prefer it as a selector.",
            "Workday's nav uses a side rail with icon-only buttons; "
            "hover to reveal labels, or use the menu shortcut at "
            "the top-right.",
            "Inkblot / hexagon icons in Workday are 'Recent' / "
            "'Favorites' / 'Inbox' — vision-only handles these "
            "better than DOM.",
            "Workday's 'Submit' button is at the bottom of long "
            "forms — scroll all the way down before searching.",
        ],
    ),
    PatternPack(
        id="generic_ecommerce",
        label="Generic e-commerce",
        url_hints=[
            "/cart", "/checkout", "/product/", "shop.",
        ],
        dom_signatures=[
            "[data-testid='cart']", "form[action*='cart']",
        ],
        tags=["ecommerce", "generic"],
        rules=[
            "Most e-commerce search bars submit on Enter — use "
            "type(submit=true) rather than hunting for a Search "
            "button.",
            "Sponsored / ad placements at the top of search results "
            "rarely match the test's intent — smart-pick should "
            "skip them.",
            "Cart pages typically hide 'Proceed to checkout' when "
            "the cart is empty. If the agent can't find it AND "
            "WorldState shows cart_count=0, flag "
            "precondition_failed instead of looping.",
            "Add-to-cart confirmations may be a toast (top right) "
            "OR a side drawer OR a navigation to /cart — verify "
            "via cart count badge change rather than the toast.",
            "Quantity selectors are often <select> dropdowns OR "
            "+/- buttons next to a numeric display. Prefer the "
            "buttons; fewer surprises.",
        ],
    ),
    PatternPack(
        id="generic_admin",
        label="Generic admin panel",
        url_hints=[
            "/admin", "/wp-admin", "/dashboard", "/console",
        ],
        dom_signatures=[
            "body.wp-admin", "[data-admin]",
        ],
        tags=["admin", "cms", "generic"],
        rules=[
            "Admin tables usually have a 'Bulk actions' selector "
            "+ checkbox on each row. Select All checkbox is in the "
            "header.",
            "Save / Update buttons are typically at the bottom "
            "of forms; on long forms, scroll past sections to "
            "find them.",
            "Confirmation dialogs for destructive actions (delete, "
            "deactivate) appear as modals — they're required_step "
            "popups, ENGAGE.",
            "Filters / search bars at the top of list views; "
            "filter chips below show active filters — clearing "
            "is via the chip's 'x'.",
        ],
    ),
    PatternPack(
        id="generic_cms",
        label="Generic CMS / WordPress / Drupal",
        url_hints=[
            "wp-admin", "/edit.php", "/wp-login", ".wpengine.",
        ],
        dom_signatures=[
            "body.wp-admin", "div#wpadminbar",
        ],
        tags=["cms", "wordpress"],
        rules=[
            "WordPress admin uses iframe for the visual editor "
            "(content area); content links live inside it.",
            "Block editor (Gutenberg) blocks are added via the "
            "'+' button at the top-left or inline within content; "
            "search blocks by name.",
            "Publish in the block editor is at the top-right — "
            "two-step (Publish → confirm).",
            "Permalink editor is below the title; click 'Edit' "
            "next to the slug.",
        ],
    ),
]


def detect_pack(
    *, target_url: str, page: "Page | None" = None,
) -> PatternPack | None:
    """Detect which pattern pack applies to a target. ``page`` is
    optional; when supplied the DOM signatures are also checked.

    Returns ``None`` when no pack matches (agent runs without pack-
    level boost — BRD chunks + recon notes + disputes still apply).
    """
    url = (target_url or "").lower()
    # 1 + 2: URL hints, longest match first.
    by_url: list[tuple[int, PatternPack]] = []
    for pack in PATTERN_PACKS:
        for hint in pack.url_hints:
            if hint and hint.lower() in url:
                by_url.append((len(hint), pack))
                break
    if by_url:
        by_url.sort(key=lambda t: t[0], reverse=True)
        return by_url[0][1]

    # 3: DOM signature when page is available.
    if page is not None:
        for pack in PATTERN_PACKS:
            for sig in pack.dom_signatures:
                try:
                    if page.locator(sig).count() > 0:
                        return pack
                except Exception:
                    continue
    return None


def autoload_pack(
    db: "Session",
    *,
    target_url: str,
    page: "Page | None" = None,
) -> PatternPack | None:
    """Detect + write the pack's rules to AKB. Idempotent — the AKB
    write_chunk helper deduplicates so re-running is cheap. Returns
    the pack that was loaded (or None when no match).
    """
    pack = detect_pack(target_url=target_url, page=page)
    if pack is None:
        return None
    from app.services.akb import write_chunk  # noqa: PLC0415

    for rule in pack.rules:
        try:
            write_chunk(
                db,
                target_url_pattern=target_url,
                kind="pattern_rule",
                content=rule,
                tags=list(pack.tags),
                confidence=0.95,
            )
        except Exception as e:
            logger.warning(
                "AKB pack write failed for %s: %s", pack.id, e,
            )
            break
    logger.info(
        "Loaded pattern pack %s for target %r (%d rules)",
        pack.id, target_url, len(pack.rules),
    )
    return pack
