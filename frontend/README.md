# QAI Tester v2 — Frontend

Next.js 16 + React 19 + Tailwind v4 + shadcn/ui (New York / neutral, OKLCH).

See [../README.md](../README.md) for the full Phase 1 demo flow.

## Run

```bash
npm install
npm run dev
```

App: <http://localhost:3000>

The backend must be running at <http://localhost:8000> — set
`NEXT_PUBLIC_API_URL` in `.env.local` to point elsewhere.

## Module layout

```
app/
├── layout.tsx                # Providers → FirstRunGate → Sidebar + main
├── providers.tsx             # next-themes + TanStack Query + sonner Toaster
├── globals.css               # Tailwind v4 (CSS-first config) + shadcn New York palette
├── icon.svg                  # Favicon
├── page.tsx                  # /  → Projects list + create dialog
├── settings/page.tsx         # /settings → LLM provider config form
└── projects/[id]/page.tsx    # /projects/{id} → detail shell

components/
├── sidebar.tsx               # Brand + nav + theme toggle
├── theme-toggle.tsx          # Hydration-safe dark/light/system
├── first-run-gate.tsx        # Welcome screen until LLM is configured
├── project-form-dialog.tsx   # Create OR Edit (driven by `project` prop)
├── delete-project-dialog.tsx # Confirm dialog with FAISS-cleanup warning
└── ui/
    ├── button.tsx            # shadcn New York Button
    ├── card.tsx
    ├── dialog.tsx            # Radix Dialog wrapper
    ├── input.tsx
    ├── label.tsx
    ├── skeleton.tsx
    └── textarea.tsx

lib/
├── api.ts                    # apiFetch + DTO types + api.* endpoint helpers
└── utils.ts                  # cn()
```

## Adding more shadcn components later

Either copy them by hand from the shadcn site, or use the CLI:

```bash
npx shadcn@latest add select dropdown-menu tabs alert-dialog tooltip ...
```

`components.json` is already wired, so the CLI knows where things go.

## Conventions

- **Server state** lives in TanStack Query. Hooks call `api.*` helpers and key on
  `["projects"]`, `["project", id]`, `["settings"]`.
- **Mutations** invalidate the affected query keys in `onSuccess`.
- **Toasts** via `sonner`'s `toast.success` / `toast.error`.
- **Theme tokens**: use `bg-background`, `text-foreground`, `bg-card`,
  `bg-sidebar`, `text-muted-foreground`, etc. — don't hard-code colors.
- **Icons**: lucide-react, default size `size-4` inside buttons.
