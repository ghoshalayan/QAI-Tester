"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [mounted, setMounted] = useState(false);
  const { resolvedTheme, setTheme } = useTheme();

  // Avoid hydration mismatch — render an empty placeholder until mounted.
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="h-9" />;

  const next = resolvedTheme === "dark" ? "light" : "dark";
  const Icon = resolvedTheme === "dark" ? Sun : Moon;

  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors hover:bg-sidebar-accent/60"
    >
      <Icon className="size-4" />
      {resolvedTheme === "dark" ? "Light mode" : "Dark mode"}
    </button>
  );
}
