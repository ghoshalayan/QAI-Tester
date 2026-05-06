"use client";

import { type FormEvent, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Plus, Sparkles, X } from "lucide-react";

import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface Props {
  projectId: number;
  /** Currently-linked document ids. Suggestions are pulled from these docs' headings. */
  linkedDocumentIds: number[];
  scope: string[];
  onChange: (next: string[]) => void;
}

export function ScopeEditor({
  projectId,
  linkedDocumentIds,
  scope,
  onChange,
}: Props) {
  const [custom, setCustom] = useState("");

  // Heading suggestions track current linked-doc set — refetch when it changes.
  const { data: suggestions, isFetching: isLoadingSuggestions } = useQuery({
    queryKey: ["heading-suggestions", projectId, linkedDocumentIds],
    queryFn: () => api.getHeadingSuggestions(projectId, linkedDocumentIds),
    enabled: linkedDocumentIds.length > 0,
    staleTime: 60_000,
  });

  const add = (value: string) => {
    const v = value.trim();
    if (!v) return;
    if (scope.includes(v)) return;
    onChange([...scope, v]);
  };

  const remove = (value: string) => {
    onChange(scope.filter((s) => s !== value));
  };

  const handleSubmitCustom = (e: FormEvent) => {
    e.preventDefault();
    add(custom);
    setCustom("");
  };

  const availableSuggestions = (suggestions?.suggestions ?? []).filter(
    (s) => !scope.includes(s),
  );

  return (
    <div className="space-y-3">
      {/* Selected chips */}
      <div className="flex flex-wrap gap-2">
        {scope.length === 0 ? (
          <span className="text-sm italic text-muted-foreground">
            No scope yet. The agent will treat the whole site as in-scope unless
            you narrow it.
          </span>
        ) : (
          scope.map((s) => (
            <Badge
              key={s}
              variant="default"
              className="gap-1.5 pr-1"
            >
              {s}
              <button
                type="button"
                onClick={() => remove(s)}
                className="rounded-full p-0.5 transition-colors hover:bg-primary-foreground/20"
                aria-label={`Remove ${s}`}
              >
                <X className="size-3" />
              </button>
            </Badge>
          ))
        )}
      </div>

      {/* Suggestions from linked docs */}
      {linkedDocumentIds.length > 0 && (
        <div className="rounded-md border bg-muted/30 p-3">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
            <Sparkles className="size-3" />
            <span>
              Suggested modules from{" "}
              {linkedDocumentIds.length === 1
                ? "the linked doc"
                : `${linkedDocumentIds.length} linked docs`}
              {isLoadingSuggestions ? "…" : null}
            </span>
          </div>
          {availableSuggestions.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {availableSuggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => add(s)}
                  className="inline-flex items-center gap-1 rounded-md border border-input bg-background px-2 py-0.5 text-xs hover:border-primary/50 hover:bg-accent"
                >
                  <Plus className="size-3" />
                  {s}
                </button>
              ))}
            </div>
          ) : (
            <p className="text-xs italic text-muted-foreground">
              {suggestions?.chunk_count
                ? "All suggestions already added."
                : "Linked docs have no chunks yet — finish ingest to see suggestions."}
            </p>
          )}
        </div>
      )}

      {/* Custom input */}
      <form onSubmit={handleSubmitCustom} className="flex gap-2">
        <Input
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          placeholder="Type a module name (e.g., Reports, Settings)…"
          maxLength={120}
        />
        <Button
          type="submit"
          size="sm"
          variant="outline"
          disabled={!custom.trim()}
          className={cn(!custom.trim() && "opacity-50")}
        >
          <Plus className="size-4" /> Add
        </Button>
      </form>
    </div>
  );
}
