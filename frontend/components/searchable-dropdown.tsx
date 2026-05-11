"use client";

/**
 * Lightweight searchable dropdown — a button that opens a popover
 * with a filter input + scrollable list. Used by the Cost Logs
 * drill-in for picking a project, then a run within that project.
 *
 * Why custom rather than shadcn's Combobox: this is one-off enough
 * that pulling in cmdk + popover + the recipe boilerplate adds more
 * code than the 80-line component below. No keyboard-arrow
 * navigation in v1 — pointer + filter is enough for the run-count
 * scale we expect (~hundreds, not thousands).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Search, X } from "lucide-react";

import { cn } from "@/lib/utils";

export interface DropdownOption {
  value: string;
  label: string;
  /** Optional second line shown smaller (e.g. timestamp / model). */
  hint?: string;
}

interface Props {
  label: string;
  placeholder?: string;
  options: DropdownOption[];
  value: string | null;
  onChange: (value: string | null) => void;
  disabled?: boolean;
  /** Width override. Defaults to a sensible fixed width. */
  className?: string;
}

export function SearchableDropdown({
  label,
  placeholder,
  options,
  value,
  onChange,
  disabled,
  className,
}: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const wrapRef = useRef<HTMLDivElement | null>(null);

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (!wrapRef.current) return;
      if (!wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  // Reset filter when closed.
  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter(
      (o) =>
        o.label.toLowerCase().includes(q) ||
        (o.hint?.toLowerCase().includes(q) ?? false) ||
        o.value.toLowerCase().includes(q),
    );
  }, [options, query]);

  const selected = options.find((o) => o.value === value) ?? null;

  return (
    <div className={cn("space-y-1", className)} ref={wrapRef}>
      <label className="block text-xs font-medium text-muted-foreground">
        {label}
      </label>
      <div className="relative">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          disabled={disabled}
          className={cn(
            "flex w-full items-center justify-between gap-2 rounded-md border px-3 py-2 text-left text-sm transition-colors",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            disabled
              ? "cursor-not-allowed opacity-50"
              : "hover:border-input",
            open && "border-primary",
          )}
        >
          <span
            className={cn(
              "min-w-0 flex-1 truncate",
              !selected && "text-muted-foreground",
            )}
          >
            {selected
              ? selected.label
              : placeholder ?? "Select…"}
          </span>
          {selected ? (
            <X
              role="button"
              className="size-4 shrink-0 text-muted-foreground hover:text-foreground"
              onClick={(e) => {
                e.stopPropagation();
                onChange(null);
              }}
            />
          ) : (
            <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
          )}
        </button>

        {open && (
          <div
            className="absolute left-0 right-0 z-50 mt-1 max-h-80 overflow-hidden rounded-md border bg-popover shadow-md"
            // Inline shadow color so it works in light + dark
            style={{ minWidth: "100%" }}
          >
            <div className="flex items-center gap-2 border-b px-2 py-1.5">
              <Search className="size-4 shrink-0 text-muted-foreground" />
              <input
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Type to filter…"
                className="w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground"
              />
            </div>
            <ul className="max-h-64 overflow-y-auto py-1">
              {filtered.length === 0 ? (
                <li className="px-3 py-4 text-center text-xs text-muted-foreground">
                  No matches
                </li>
              ) : (
                filtered.map((opt) => (
                  <li key={opt.value}>
                    <button
                      type="button"
                      onClick={() => {
                        onChange(opt.value);
                        setOpen(false);
                      }}
                      className={cn(
                        "w-full px-3 py-2 text-left text-sm hover:bg-accent",
                        opt.value === value && "bg-accent/60",
                      )}
                    >
                      <div className="truncate">{opt.label}</div>
                      {opt.hint && (
                        <div className="truncate text-[10px] text-muted-foreground">
                          {opt.hint}
                        </div>
                      )}
                    </button>
                  </li>
                ))
              )}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
