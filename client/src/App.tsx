import { useCallback, useEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";
import axios, { type AxiosProgressEvent } from "axios";
import "./App.css";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Role = "user" | "assistant";

interface ToolCallStep {
    type: "tool_call";
    name: string;
}

interface ToolResultStep {
    type: "tool_result";
    name: string;
    result: unknown;
}

interface ToolErrorStep {
    type: "tool_error";
    name: string;
}

type TraceStep = ToolCallStep | ToolResultStep | ToolErrorStep;

interface ChatMessage {
    id: string;
    role: Role;
    content: string;
    trace?: TraceStep[];
    streaming?: boolean;
    error?: boolean;
}

interface HealthResponse {
    model?: string;
    keyConfigured?: boolean;
}

interface StatusState {
    ok: boolean | null;
    model: string;
    keyConfigured: boolean;
}

interface SSEBlock {
    event: string;
    data: string;
}

interface TokenEvent {
    content: string;
}

interface DoneEvent {
    reply: string;
    trace: TraceStep[];
}

interface ErrorEvent {
    error: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EXAMPLE_PROMPTS: { text: string; icon: string }[] = [
    { text: "What's the weather in Bengaluru right now?", icon: "☀️" },
    { text: "Compare the weather in Tokyo and Reykjavik today.", icon: "🌍" },
    { text: "How many words are in the sentence: 'Relay reasons, then it acts.'?", icon: "🔤" },
];

const TOOL_LABEL: Record<string, string> = {
    get_weather: "weather",
    get_current_time: "clock",
    word_stats: "text analyzer",
};

/** Maps a weather condition string to a color + icon so the UI can react to it. */
interface WeatherTheme {
    color: string;
    icon: string;
}

const WEATHER_THEME: Record<string, WeatherTheme> = {
    clear: { color: "#F5A623", icon: "☀️" },
    sunny: { color: "#F5A623", icon: "☀️" },
    cloud: { color: "#94A3B8", icon: "⛅" },
    overcast: { color: "#94A3B8", icon: "☁️" },
    drizzle: { color: "#38BDF8", icon: "🌦️" },
    rain: { color: "#38BDF8", icon: "🌧️" },
    thunder: { color: "#A78BFA", icon: "⛈️" },
    storm: { color: "#A78BFA", icon: "⛈️" },
    snow: { color: "#BAE6FD", icon: "❄️" },
    sleet: { color: "#BAE6FD", icon: "🌨️" },
    fog: { color: "#9CA3AF", icon: "🌫️" },
    mist: { color: "#9CA3AF", icon: "🌫️" },
    haze: { color: "#9CA3AF", icon: "🌫️" },
    wind: { color: "#5EEAD4", icon: "🌬️" },
};

const DEFAULT_WEATHER_THEME: WeatherTheme = { color: "#38BDF8", icon: "🌤️" };

/** Resolves a raw condition string (e.g. "light rain") to a theme by keyword match. */
function resolveWeatherTheme(condition?: string): WeatherTheme {
    if (!condition) return DEFAULT_WEATHER_THEME;
    const key = condition.toLowerCase();
    const match = Object.keys(WEATHER_THEME).find((k) => key.includes(k));
    return match ? WEATHER_THEME[match] : DEFAULT_WEATHER_THEME;
}

/** Pulls a weather theme out of a get_weather tool_result, if present in the trace. */
function weatherThemeFromTrace(trace?: TraceStep[]): WeatherTheme | null {
    if (!trace) return null;
    for (let i = trace.length - 1; i >= 0; i--) {
        const step = trace[i];
        if (step.type === "tool_result" && step.name === "get_weather") {
            const r = step.result as Record<string, unknown> | undefined;
            const condition = typeof r?.condition === "string" ? r.condition : undefined;
            return resolveWeatherTheme(condition);
        }
    }
    return null;
}

// XHR adapter is required (not the fetch adapter) so onDownloadProgress can
// read the partial response body via xhr.responseText while streaming.
const api = axios.create({ baseURL: import.meta.env.VITE_API_URL ?? "/api", adapter: "xhr" });

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Parses one SSE "event: X\ndata: Y\n\n" block into { event, data }. */
function parseSSEBlock(block: string): SSEBlock {
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
    }
    return { event, data };
}

/** Applies one parsed SSE event to an assistant message, returning the updated message. */
function applySSEEvent(message: ChatMessage, event: string, parsed: TokenEvent | TraceStep | DoneEvent | ErrorEvent): ChatMessage {
    switch (event) {
        case "token":
            return { ...message, content: message.content + (parsed as TokenEvent).content };
        case "trace": {
            const incoming = parsed as TraceStep;
            const trace = message.trace ?? [];

            const exists = trace.some(
                (t) => JSON.stringify(t) === JSON.stringify(incoming)
            );

            if (exists) {
                return message;
            }

            return {
                ...message,
                trace: [...trace, incoming],
            };
        }
        case "done": {
            const done = parsed as DoneEvent;
            return { ...message, content: done.reply, trace: done.trace, streaming: false };
        }
        case "error":
            return { ...message, content: (parsed as ErrorEvent).error, error: true, streaming: false };
        default:
            return message;
    }
}

// ---------------------------------------------------------------------------
// Trace strip
// ---------------------------------------------------------------------------

/** A tool_call paired with its (possibly not-yet-arrived) outcome. */
interface TraceGroup {
    call: ToolCallStep;
    outcome?: ToolResultStep | ToolErrorStep;
}

/** Groups a flat trace array into call+outcome pairs, in call order. */
function groupTrace(trace: TraceStep[]): TraceGroup[] {
    const groups: TraceGroup[] = [];
    for (const step of trace) {
        if (step.type === "tool_call") {
            groups.push({ call: step });
        } else {
            const openGroup = [...groups].reverse().find((g) => g.call.name === step.name && !g.outcome);
            if (openGroup) openGroup.outcome = step;
            else groups.push({ call: { type: "tool_call", name: step.name }, outcome: step });
        }
    }
    return groups;
}

const TOOL_ICON: Record<string, string> = {
    get_weather: "☀️",
    get_current_time: "🕐",
    word_stats: "🔤",
};

/** Turns a raw tool result into a short, human-readable summary. */
function summarizeResult(name: string, result: unknown): string {
    if (result && typeof result === "object") {
        const r = result as Record<string, unknown>;
        if (name === "get_weather") {
            const parts: string[] = [];
            if (typeof r.location === "string") parts.push(r.location);
            if (typeof r.temperature === "number") parts.push(`${r.temperature}°C`);
            if (typeof r.condition === "string") parts.push(String(r.condition));
            if (parts.length) return parts.join(" — ");
        }
        if (name === "get_current_time") {
            if (typeof r.iso === "string") {
                const d = new Date(r.iso);
                if (!Number.isNaN(d.getTime())) {
                    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                }
            }
        }
        if (name === "word_stats" && typeof r.words === "number") {
            return `${r.words} word${r.words === 1 ? "" : "s"}`;
        }
    }
    const text = typeof result === "object" ? JSON.stringify(result) : String(result);
    return text.length > 60 ? text.slice(0, 60) + "…" : text;
}

function TraceStepRow({ group }: { group: TraceGroup }) {
    const [open, setOpen] = useState(false);
    const { call, outcome } = group;
    const label = TOOL_LABEL[call.name] || call.name;

    const isError = outcome?.type === "tool_error";
    const isPending = !outcome;
    const raw = outcome?.type === "tool_result" ? outcome.result : undefined;

    const weather =
        call.name === "get_weather" && raw && typeof raw === "object"
            ? resolveWeatherTheme((raw as Record<string, unknown>).condition as string | undefined)
            : null;
    const icon = weather ? weather.icon : TOOL_ICON[call.name] || "🛠️";
    const accent = weather?.color;

    return (
        <li
            className={`trace-row ${isError ? "trace-row-error" : ""} ${isPending ? "trace-row-pending" : ""}`}
            style={accent ? ({ "--row-accent": accent } as CSSProperties) : undefined}
        >
            <button
                type="button"
                className={`trace-row-main ${accent ? "trace-row-tinted" : ""}`}
                onClick={() => raw !== undefined && setOpen((o) => !o)}
                aria-expanded={open}
            >
                <span className="trace-icon" aria-hidden="true">{isPending ? <span className="trace-spinner" /> : icon}</span>
                <span className="trace-text">
                    <span className="trace-label">{label}</span>
                    <span className="trace-summary">
                        {isPending
                            ? "working…"
                            : isError
                                ? "failed"
                                : summarizeResult(call.name, raw)}
                    </span>
                </span>
                {raw !== undefined && <span className={`trace-chevron ${open ? "trace-chevron-open" : ""}`}>›</span>}
            </button>
            {open && raw !== undefined && (
                <pre className="trace-raw">{JSON.stringify(raw, null, 2)}</pre>
            )}
        </li>
    );
}

function TraceStrip({ trace }: { trace?: TraceStep[] }) {
    const [expanded, setExpanded] = useState(false);

    if (!trace || trace.length === 0) return null;

    const groups = groupTrace(trace);
    const anyPending = groups.some((g) => !g.outcome);
    const doneCount = groups.filter((g) => g.outcome).length;

    return (
        <div className="trace">
            <button type="button" className="trace-toggle" onClick={() => setExpanded((e) => !e)} aria-expanded={expanded}>
                <span className={`trace-toggle-icon ${anyPending ? "trace-toggle-icon-pending" : ""}`} aria-hidden="true">
                    {anyPending ? <span className="trace-spinner" /> : "✓"}
                </span>
                {anyPending
                    ? `Using ${groups.length} tool${groups.length === 1 ? "" : "s"}…`
                    : `Used ${doneCount} tool${doneCount === 1 ? "" : "s"}`}
                <span className={`trace-chevron ${expanded ? "trace-chevron-open" : ""}`}>›</span>
            </button>
            {expanded && (
                <ul className="trace-list">
                    {groups.map((group, i) => (
                        <TraceStepRow key={i} group={group} />
                    ))}
                </ul>
            )}
        </div>
    );
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState<StatusState>({ ok: null, model: "", keyConfigured: true });
    const listRef = useRef<HTMLElement>(null);

    const fetchHealth = useCallback(async () => {
        try {
            const { data } = await api.get<HealthResponse>("/health");
            setStatus({ ok: true, model: data.model ?? "", keyConfigured: data.keyConfigured ?? true });
        } catch {
            setStatus({ ok: false, model: "", keyConfigured: true });
        }
    }, []);

    useEffect(() => {
        fetchHealth();
    }, [fetchHealth]);

    useEffect(() => {
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
    }, [messages, loading]);

    // Refs mirror the latest state so `send` can read current values without
    // needing `messages`/`loading` in its dependency array — keeping it (and
    // anything that depends on it, like onKeyDown) referentially stable.
    const messagesRef = useRef<ChatMessage[]>(messages);
    const loadingRef = useRef(loading);

    useEffect(() => {
        messagesRef.current = messages;
    }, [messages]);

    useEffect(() => {
        loadingRef.current = loading;
    }, [loading]);

    const send = useCallback(async (text: string) => {
            const trimmed = text.trim();
            if (!trimmed || loadingRef.current) return;

            const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: trimmed };
            const nextMessages = [...messagesRef.current, userMsg];
            const assistantId = crypto.randomUUID();

            setMessages([...nextMessages,{ id: assistantId, role: "assistant", content: "", trace: [], streaming: true }]);
            setInput("");
            setLoading(true);

            const patch = (fn: (m: ChatMessage) => ChatMessage) =>
            setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

            // axios hands onDownloadProgress the *entire* buffer received so far,
            // so we track how much of it we've already parsed into SSE blocks.
            let processedLength = 0;
            let pending = "";

            const handleChunk = (progressEvent: AxiosProgressEvent) => {
                const xhr = progressEvent.event?.target as XMLHttpRequest;
                const full = xhr.responseText;

                const chunk = full.slice(processedLength);
                processedLength = full.length;

                pending += chunk;

                const blocks = pending.split("\n\n");

                // keep incomplete block
                pending = blocks.pop() ?? "";

                for (const block of blocks) {
                    if (!block.trim() || block.startsWith(":")) continue;

                    const { event, data } = parseSSEBlock(block);

                    if (!data) continue;

                    try {
                        patch((m) => applySSEEvent(m, event, JSON.parse(data)));
                    } catch {
                        // ignore malformed partial json
                    }
                }
            };

            try {
                await api.post("/chat/stream",{ messages: nextMessages.map(({ role, content }) => ({ role, content })) },
                    {
                        responseType: "text",
                        headers: { "Content-Type": "application/json" },
                        onDownloadProgress: handleChunk,
                    }
                );
            } catch (err) {
                const message = axios.isAxiosError(err) ? err.response?.data?.error ?? err.message : err instanceof Error ? err.message : "Something went wrong";
                patch((m) => ({ ...m, content: message, error: true, streaming: false }));
            } finally {
                setLoading(false);
            }
    }, []); // stable — reads current state via refs above

    const onKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send(input);
        }
    },[input, send]);

    return (
        <div className="app">
            <header className="header">
                <div className="brand">
                    <img className="brand-mark" src="/vayuniq.png" alt="Vayuniq" />
                    <span className="brand-name">VAYUNIQ</span>
                </div>
                <div className="header-meta">
                    {status.keyConfigured === false && <span className="badge badge-danger">no API key set</span>}
                    {status.model && <span className="badge">{status.model}</span>}
                    <span className={`status-dot ${status.ok ? "status-on" : "status-off"}`} />
                </div>
            </header>

            <main className="chat" ref={listRef}>
                {messages.length === 0 && (
                    <div className="empty">
                        <p className="empty-title">Ask something a tool can help with.</p>
                        <p className="empty-sub">Vayuniq decides when to check live weather, the clock, or the text analyzer — you'll see each step it takes before it answers.</p>
                        <div className="examples">
                            {EXAMPLE_PROMPTS.map(({ text, icon }) => (
                                <button key={text} className="example-chip" onClick={() => send(text)}>
                                    <span className="example-chip-icon" aria-hidden="true">{icon}</span>
                                    {text}
                                </button>
                            ))}
                        </div>
                    </div>
                )}
                {messages.map((m) => {
                    const weather = m.role === "assistant" ? weatherThemeFromTrace(m.trace) : null;
                    return (
                        <div key={m.id} className={`row row-${m.role}`}>
                            <div
                                className={`bubble bubble-${m.role} ${m.error ? "bubble-error" : ""} ${weather ? "bubble-weather" : ""}`}
                                style={weather ? ({ "--msg-accent": weather.color } as CSSProperties) : undefined}
                            >
                                {m.role === "assistant" && <TraceStrip trace={m.trace} />}
                                {m.role === "assistant" && m.streaming && !m.content && (m.trace?.length ?? 0) === 0 ? (
                                    <span className="typing-dots">
                                        <span className="typing-dot" />
                                        <span className="typing-dot" />
                                        <span className="typing-dot" />
                                    </span>
                                ) : (
                                    <p>{m.content}{m.streaming && <span className="cursor" />}</p>
                                )}
                            </div>
                        </div>
                    );
                })}
            </main>

            <footer className="composer">
                <textarea value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={onKeyDown} placeholder="Message Vayuniq… (Enter to send, Shift+Enter for a new line)" rows={1}/>
                <button className="send-btn" onClick={() => send(input)} disabled={loading || !input.trim()}>Send</button>
            </footer>
        </div>
    );
}