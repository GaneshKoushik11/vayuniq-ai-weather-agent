import "dotenv/config";
import express, { type Request, type Response } from "express";
import cors from "cors";
import Groq from "groq-sdk";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Role = "system" | "user" | "assistant" | "tool";

interface ChatMessage {
    role: Role;
    content: string | null;
    tool_call_id?: string;
    tool_calls?: GroqToolCall[];
}

interface GroqToolCall {
    id: string;
    type: "function";
    function: { name: string; arguments: string };
}

interface ToolCallStep {
    type: "tool_call";
    name: string;
    args: Record<string, unknown>;
}

interface ToolResultStep {
    type: "tool_result";
    name: string;
    result: unknown;
}

interface ToolErrorStep {
    type: "tool_error";
    name: string;
    error: string;
}

type TraceStep = ToolCallStep | ToolResultStep | ToolErrorStep;

interface AgentResult {
    reply: string;
    trace: TraceStep[];
    model: string;
}

interface WeatherResult {
    location: string;
    temperature_c: number | null;
    feels_like_c: number | null;
    humidity_percent: number | null;
    wind_kph: number | null;
    condition: string;
    local_time: string;
}

interface WordStatsResult {
    words: number;
    characters: number;
    sentences: number;
}

type Emit = (event: string, data: unknown) => void;

// Minimal shape of an in-progress streamed tool call as it's assembled
// across delta chunks.
interface AccumulatingToolCall {
    id: string;
    name: string;
    arguments: string;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const PORT = Number(process.env.PORT) || 8787;
const MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
const MAX_STEPS = 10; // safety cap on tool-call loops

if (!process.env.GROQ_API_KEY) {
    console.warn("\n⚠️  GROQ_API_KEY is not set. Copy server/.env.example to server/.env and add your key.\n");
}

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// ---------------------------------------------------------------------------
// Tools the agent is allowed to call. Each has a JSON-schema description
// (sent to the model) and a matching implementation (run locally).
// ---------------------------------------------------------------------------

const TOOL_DEFINITIONS = [
    {
        type: "function",
        function: {
            name: "get_weather",
            description: "Get the current real-time weather for a city or place name (temperature, conditions, wind, humidity).",
            parameters: {
                type: "object",
                properties: {
                    location: {
                        type: "string",
                        description: "A city or place name, optionally with country, e.g. 'Bengaluru' or 'Paris, France'",
                    },
                },
                required: ["location"],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_current_time",
            description: "Get the current date and time on the server, in ISO format and UTC.",
            parameters: {
                type: "object",
                properties: {},
            },
        },
    },
    {
        type: "function",
        function: {
            name: "word_stats",
            description:"Count words, characters, and sentences in a piece of text. Use this instead of counting manually.",
            parameters: {
                type: "object",
                properties: {
                    text: { type: "string", description: "The text to analyze" },
                },
                required: ["text"],
            },
        },
    },
] as const;

// WMO weather codes -> plain-language conditions (used by Open-Meteo).
const WEATHER_CODES: Record<number, string> = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
};

async function getWeather(location: string): Promise<WeatherResult> {
    if (!location || typeof location !== "string") {
        throw new Error("A location is required.");
    }

    // 1. Turn the place name into coordinates (free, no API key).
    const geoUrl = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(location)}&count=1&language=en&format=json`;
    const geoRes = await fetch(geoUrl);
    if (!geoRes.ok) throw new Error("Location lookup failed.");
    const geo = await geoRes.json() as {
        results?: Array<{
            name: string;
            admin1?: string;
            country?: string;
            latitude: number;
            longitude: number;
        }>;
    };
    const place = geo.results?.[0];
    if (!place) throw new Error(`Couldn't find a place called "${location}".`);

    // 2. Fetch current conditions for those coordinates.
    const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${place.latitude}&longitude=${place.longitude}` + `&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code` + `&timezone=auto`;
    const weatherRes = await fetch(weatherUrl);
    if (!weatherRes.ok) throw new Error("Weather lookup failed.");
    const weather = await weatherRes.json() as {
        current?: {
            temperature_2m: number | null;
            apparent_temperature: number | null;
            relative_humidity_2m: number | null;
            wind_speed_10m: number | null;
            weather_code: number;
            time: string;
        };
    };
    const current = weather.current;
    if (!current) throw new Error("No current weather data available for that location.");

    return {
        location: [place.name, place.admin1, place.country].filter(Boolean).join(", "),
        temperature_c: current.temperature_2m,
        feels_like_c: current.apparent_temperature,
        humidity_percent: current.relative_humidity_2m,
        wind_kph: current.wind_speed_10m,
        condition: WEATHER_CODES[current.weather_code] || "unknown conditions",
        local_time: current.time,
    };
}

function wordStats(text: string): WordStatsResult {
    const trimmed = (text || "").trim();
    const words = trimmed.length ? trimmed.split(/\s+/).length : 0;
    const characters = text.length;
    const sentences = (trimmed.match(/[.!?]+(\s|$)/g) || []).length || (trimmed ? 1 : 0);
    return { words, characters, sentences };
}

async function executeTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    switch (name) {
        case "get_weather":
            return await getWeather(args.location as string);
        case "get_current_time":
            return { iso: new Date().toISOString(), utc: new Date().toUTCString() };
        case "word_stats":
            return wordStats(args.text as string);
        default:
            throw new Error(`Unknown tool: ${name}`);
    }
}

function parseToolArgs(raw: string | undefined): Record<string, unknown> {
    try {
        return JSON.parse(raw || "{}");
    } catch {
        return {};
    }
}

async function callTool(name: string,argsRaw: string | undefined): Promise<{ callStep: ToolCallStep; resultStep: ToolResultStep | ToolErrorStep; result: unknown }> {
    const args = parseToolArgs(argsRaw);
    const callStep: ToolCallStep = { type: "tool_call", name, args };

    try {
        const result = await executeTool(name, args);
        const resultStep: ToolResultStep = { type: "tool_result", name, result };
        return { callStep, resultStep, result };
    } catch (err) {
        const message = err instanceof Error ? err.message : "Tool execution failed.";
        const result = { error: message };
        const resultStep: ToolErrorStep = { type: "tool_error", name, error: message };
        return { callStep, resultStep, result };
    }
}

// ---------------------------------------------------------------------------
// Agent loop: call the model, and if it asks for tools, run them and feed
// the results back in, repeating until it produces a final answer.
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are Vayuniq, a concise, capable assistant with access to tools.
- Use a tool whenever it would make your answer more accurate (live weather, current time, text stats).
- Think step by step, but only show your final, direct answer to the user — do not narrate your reasoning process.
- If a tool result is surprising, double-check by reasoning about whether it makes sense before answering.
- Keep answers short and to the point unless the user asks for detail.`;

const STEP_LIMIT_MESSAGE = "I wasn't able to finish reasoning about that within my step limit — try breaking the question down.";

async function runAgent(conversation: ChatMessage[]): Promise<AgentResult> {
    const messages: ChatMessage[] = [{ role: "system", content: SYSTEM_PROMPT }, ...conversation];
    const trace: TraceStep[] = [];

    for (let step = 0; step < MAX_STEPS; step++) {
        const completion = await groq.chat.completions.create({
            model: MODEL,
            messages: messages as any,
            tools: TOOL_DEFINITIONS as any,
            tool_choice: "auto",
            temperature: 0.3,
        });

        const message = completion.choices[0].message;
        messages.push(message as unknown as ChatMessage);

        const toolCalls = message.tool_calls || [];
        if (toolCalls.length === 0) {
            return { reply: message.content ?? "", trace, model: MODEL };
        }

        for (const call of toolCalls) {
            const { callStep, resultStep, result } = await callTool(call.function.name, call.function.arguments);
            trace.push(callStep, resultStep);
            messages.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(result) });
        }
    }

    return { reply: STEP_LIMIT_MESSAGE, trace, model: MODEL };
}

// ---------------------------------------------------------------------------
// Streaming version of the same loop. Emits three kinds of events via
// `emit(event, data)`:
//   - "token": a chunk of the model's answer text, as it's generated
//   - "trace": a tool_call / tool_result / tool_error step, live
//   - "done":  the final reply + full trace
// This is what makes the agent feel real-time: the browser doesn't wait for
// the whole answer, and it sees each tool call the moment it happens.
// ---------------------------------------------------------------------------

async function runAgentStream(conversation: ChatMessage[], emit: Emit): Promise<void> {
    const messages: ChatMessage[] = [{ role: "system", content: SYSTEM_PROMPT }, ...conversation];
    const trace: TraceStep[] = [];

    for (let step = 0; step < MAX_STEPS; step++) {
        const stream = await groq.chat.completions.create({
            model: MODEL,
            messages: messages as any,
            tools: TOOL_DEFINITIONS as any,
            tool_choice: "auto",
            temperature: 0.3,
            stream: true,
        });

        let content = "";
        const toolCallsAcc: Record<number, AccumulatingToolCall> = {};

        for await (const chunk of stream) {
            const delta = chunk.choices?.[0]?.delta;
            if (!delta) continue;

            if (delta.content) {
                content += delta.content;
                emit("token", { content: delta.content });
            }

            if (delta.tool_calls) {
                for (const tc of delta.tool_calls) {
                    const idx = tc.index ?? 0;
                    if (!toolCallsAcc[idx]) toolCallsAcc[idx] = { id: "", name: "", arguments: "" };
                    if (tc.id) toolCallsAcc[idx].id = tc.id;
                    if (tc.function?.name) toolCallsAcc[idx].name += tc.function.name;
                    if (tc.function?.arguments) toolCallsAcc[idx].arguments += tc.function.arguments;
                }
            }
        }

        const toolCalls = Object.values(toolCallsAcc);

        const assistantMessage: ChatMessage = { role: "assistant", content: content || null };
        if (toolCalls.length > 0) {
            assistantMessage.tool_calls = toolCalls.map((tc) => ({
                id: tc.id,
                type: "function",
                function: { name: tc.name, arguments: tc.arguments },
            }));
        }
        messages.push(assistantMessage);

        if (toolCalls.length === 0) {
            emit("done", { reply: content, trace, model: MODEL });
            return;
        }

        for (const tc of toolCalls) {
            const { callStep, resultStep, result } = await callTool(tc.name, tc.arguments);
            trace.push(callStep);
            emit("trace", callStep);
            trace.push(resultStep);
            emit("trace", resultStep);

            messages.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify(result) });
        }
    }

    emit("done", { reply: STEP_LIMIT_MESSAGE, trace, model: MODEL });
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

function isValidMessages(body: unknown): body is { messages: ChatMessage[] } {
    return (
        typeof body === "object" &&
        body !== null &&
        Array.isArray((body as { messages?: unknown }).messages) &&
        (body as { messages: unknown[] }).messages.length > 0
    );
}

app.get("/api/health", (req: Request, res: Response) => {
    res.json({ ok: true, model: MODEL, keyConfigured: Boolean(process.env.GROQ_API_KEY) });
});

app.post("/api/chat", async (req: Request, res: Response) => {
    if (!isValidMessages(req.body)) {
        return res.status(400).json({ error: "Request body must include a non-empty 'messages' array." });
    }

    try {
        const result = await runAgent(req.body.messages);
        res.json(result);
    } catch (err) {
        console.error("Agent error:", err);
        const message = err instanceof Error ? err.message : "The agent failed to respond.";
        res.status(500).json({ error: message });
    }
});

// Real-time streaming endpoint: Server-Sent Events over a POST request.
// The browser reads the response body as a stream instead of polling, so
// tokens and tool-call events show up the instant the server produces them.
app.post("/api/chat/stream", async (req: Request, res: Response) => {
    if (!isValidMessages(req.body)) {
        return res.status(400).json({ error: "Request body must include a non-empty 'messages' array." });
    }

    res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no", // disable proxy buffering (e.g. nginx) so events flush immediately
    });
    res.flushHeaders?.();

    const emit: Emit = (event, data) => {
        res.write(`event: ${event}\n`);
        res.write(`data: ${JSON.stringify(data)}\n\n`);

        if (typeof (res as any).flush === "function") {
            (res as any).flush();
        }
    };

    // Keep the connection alive through slow tool calls / proxies.
    const heartbeat = setInterval(() => res.write(": ping\n\n"), 15000);

    try {
        await runAgentStream(req.body.messages, emit);
    } catch (err) {
        console.error("Agent stream error:", err);
        const message = err instanceof Error ? err.message : "The agent failed to respond.";
        emit("error", { error: message });
    } finally {
        clearInterval(heartbeat);
        res.end();
    }
});

app.listen(PORT, () => {
    console.log(`Simple Agent server running at http://localhost:${PORT}`);
});