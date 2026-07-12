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

interface ForecastDay {
    date: string;
    condition: string;
    temp_max_c: number | null;
    temp_min_c: number | null;
    precipitation_probability_percent: number | null;
    wind_kph: number | null;
}

interface ForecastResult {
    location: string;
    days_returned: number;
    days_requested: number;
    note: string | null;
    forecast: ForecastDay[];
}

// Coordinates the client's browser supplied via navigator.geolocation, if
// the person allowed it. Never guessed or defaulted server-side.
interface RequestLocation {
    latitude: number;
    longitude: number;
}

type Emit = (event: string, data: unknown) => void;

interface AccumulatingToolCall {
    id: string;
    name: string;
    arguments: string;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const PORT = Number(process.env.PORT) || 8787;
const MODEL = process.env.GROQ_MODEL || "llama-3.1-8b-instant";
const MAX_STEPS = 10;

if (!process.env.GROQ_API_KEY) {
    console.warn("\n⚠️  GROQ_API_KEY is not set. Copy server/.env.example to server/.env and add your key.\n");
}

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

const TOOL_DEFINITIONS = [
    {
        type: "function",
        function: {
            name: "get_weather",
            description:
                "Get the current real-time weather for a place (temperature, conditions, wind, humidity). " +
                "If the user is asking about their own current location ('here', 'my location', 'near me', " +
                "'where I am'), set use_current_location=true and omit 'location' — never guess or invent a " +
                "city name for them.",
            parameters: {
                type: "object",
                properties: {
                    location: {
                        type: "string",
                        description: "A city or place name, optionally with country, e.g. 'Bengaluru' or 'Paris, France'. Omit this if use_current_location is true.",
                    },
                    use_current_location: {
                        type: "boolean",
                        description: "Set true when the user means their own current location rather than a named place.",
                    },
                },
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_forecast",
            description:
                "Get a daily weather forecast for a place (max/min temperature, condition, precipitation " +
                "chance, wind). Ask for as many days as the user needs, even if that's weeks or a month out — " +
                "the tool will return as many days as the weather provider can actually forecast and tell you " +
                "if it had to return fewer than requested, along with why. Don't refuse a request just because " +
                "it's far in the future; call the tool and let it report its real limits. If the user means " +
                "their own current location ('here', 'my location', 'near me'), set use_current_location=true " +
                "and omit 'location' — never guess a city name for them.",
            parameters: {
                type: "object",
                properties: {
                    location: {
                        type: "string",
                        description: "A city or place name, optionally with country, e.g. 'Bengaluru' or 'Paris, France'. Omit this if use_current_location is true.",
                    },
                    use_current_location: {
                        type: "boolean",
                        description: "Set true when the user means their own current location rather than a named place.",
                    },
                    days: {
                        type: "integer",
                        description: "Number of days to forecast ahead. Defaults to 7 if omitted. Fine to ask for more than any known limit — the tool clamps and explains.",
                        minimum: 1,
                    },
                },
            },
        },
    },
] as const;

const WEATHER_CODES: Record<number, string> = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "moderate drizzle",
    55: "dense drizzle", 61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with light hail", 99: "thunderstorm with heavy hail",
};

async function geocode(location: string): Promise<{ name: string; admin1?: string; country?: string; latitude: number; longitude: number }> {
    const geoUrl = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(location)}&count=1&language=en&format=json`;
    const geoRes = await fetch(geoUrl);
    if (!geoRes.ok) throw new Error("Location lookup failed.");
    const geo = await geoRes.json() as {
        results?: Array<{ name: string; admin1?: string; country?: string; latitude: number; longitude: number }>;
    };
    const place = geo.results?.[0];
    if (!place) throw new Error(`Couldn't find a place called "${location}".`);
    return place;
}

function formatCoordsLabel(lat: number, lon: number): string {
    const ns = lat >= 0 ? "N" : "S";
    const ew = lon >= 0 ? "E" : "W";
    return `${Math.abs(lat).toFixed(2)}°${ns}, ${Math.abs(lon).toFixed(2)}°${ew}`;
}

async function reverseGeocodeViaBigDataCloud(lat: number, lon: number): Promise<string | null> {
    const res = await fetch(
        `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`
    );
    if (!res.ok) return null;
    const data = await res.json() as { city?: string; locality?: string; principalSubdivision?: string; countryName?: string };
    const label = [data.city || data.locality, data.principalSubdivision, data.countryName].filter(Boolean).join(", ");
    return label || null;
}

async function reverseGeocodeViaNominatim(lat: number, lon: number): Promise<string | null> {
    const res = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=10`,
        { headers: { "User-Agent": "VayuniqWeatherAgent/1.0 (weather lookup)" } }
    );
    if (!res.ok) return null;
    const data = await res.json() as { address?: Record<string, string> };
    const addr = data.address || {};
    const label = [addr.city || addr.town || addr.village || addr.county, addr.state, addr.country].filter(Boolean).join(", ");
    return label || null;
}

// Best-effort label for a set of coordinates. Tries a couple of free
// reverse-geocoding providers (in case one is rate-limited or blocked for
// this server's IP), and if all of them fail, falls back to the actual
// coordinates rather than a vague placeholder — so the reply always names
// something specific instead of silently saying "your location".
async function reverseGeocodeLabel(lat: number, lon: number): Promise<string> {
    for (const provider of [reverseGeocodeViaBigDataCloud, reverseGeocodeViaNominatim]) {
        try {
            const label = await provider(lat, lon);
            if (label) return label;
        } catch {
            // try the next provider
        }
    }
    return formatCoordsLabel(lat, lon);
}

function requireLocation(location: RequestLocation | undefined): RequestLocation {
    if (!location) {
        throw new Error(
            "The user's current location hasn't been shared by their browser/device yet, so it can't be looked " +
            "up. Don't guess or invent a city — ask them to allow location access, or ask which place they mean."
        );
    }
    return location;
}

async function getWeatherByCoords(location: RequestLocation): Promise<WeatherResult> {
    const { latitude, longitude } = location;
    const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}` + `&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code` + `&timezone=auto`;
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
    if (!current) throw new Error("No current weather data available for your location.");

    const label = await reverseGeocodeLabel(latitude, longitude);

    return {
        location: label,
        temperature_c: current.temperature_2m,
        feels_like_c: current.apparent_temperature,
        humidity_percent: current.relative_humidity_2m,
        wind_kph: current.wind_speed_10m,
        condition: WEATHER_CODES[current.weather_code] || "unknown conditions",
        local_time: current.time,
    };
}

async function getWeather(location: string): Promise<WeatherResult> {
    if (!location || typeof location !== "string") {
        throw new Error("A location is required.");
    }

    const place = await geocode(location);

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

async function fetchDailyForecast(lat: number, lon: number, days: number): Promise<{
    time: string[];
    weather_code: number[];
    temperature_2m_max: number[];
    temperature_2m_min: number[];
    precipitation_probability_max: number[];
    wind_speed_10m_max: number[];
}> {
    const url =
        `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
        `&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max` +
        `&forecast_days=${days}&timezone=auto`;
    const res = await fetch(url);

    if (!res.ok) {
        const body = await res.json().catch(() => null) as { reason?: string } | null;
        const err = new Error(body?.reason || "Forecast lookup failed.");
        (err as Error & { providerReason?: string }).providerReason = body?.reason;
        throw err;
    }

    const data = await res.json() as { daily?: ForecastDaily };
    if (!data.daily) throw new Error("No forecast data available for that location.");
    return data.daily;
}

type ForecastDaily = {
    time: string[];
    weather_code: number[];
    temperature_2m_max: number[];
    temperature_2m_min: number[];
    precipitation_probability_max: number[];
    wind_speed_10m_max: number[];
};

// The provider caps how many days out it can forecast, but that cap isn't
// something we hardcode — we ask for what the user actually wants, and if the
// provider rejects it, its error message tells us the real current limit. We
// retry once at that corrected value rather than assuming a fixed number, so
// this keeps working correctly even if the provider's limit changes.
async function getDailyForecastClamped(lat: number, lon: number, daysRequested: number): Promise<{ daily: ForecastDaily; daysServed: number }> {
    try {
        const daily = await fetchDailyForecast(lat, lon, daysRequested);
        return { daily, daysServed: daysRequested };
    } catch (err) {
        const reason = (err as Error & { providerReason?: string }).providerReason || (err as Error).message || "";
        const match = reason.match(/\b(\d+)\b(?!.*\b\d+\b)/); // last number mentioned, typically the max allowed
        const discoveredLimit = match ? Number(match[1]) : null;

        if (discoveredLimit && discoveredLimit < daysRequested) {
            const daily = await fetchDailyForecast(lat, lon, discoveredLimit);
            return { daily, daysServed: discoveredLimit };
        }
        throw err;
    }
}

function buildForecastResult(
    daily: ForecastDaily,
    daysReturned: number,
    daysRequested: number,
    locationLabel: string
): ForecastResult {
    const forecast: ForecastDay[] = daily.time.map((date, i) => ({
        date,
        condition: WEATHER_CODES[daily.weather_code[i]] || "unknown conditions",
        temp_max_c: daily.temperature_2m_max?.[i] ?? null,
        temp_min_c: daily.temperature_2m_min?.[i] ?? null,
        precipitation_probability_percent: daily.precipitation_probability_max?.[i] ?? null,
        wind_kph: daily.wind_speed_10m_max?.[i] ?? null,
    }));

    const note =
        daysReturned < daysRequested
            ? `Only ${daysReturned} day(s) of forecast could be retrieved even though ${daysRequested} were requested. ` +
              `This is because weather models lose reliable day-by-day accuracy beyond a certain horizon — not a ` +
              `limitation of this app. Mention this plainly to the user, offer the ${daysReturned}-day outlook you ` +
              `do have, and suggest typical/seasonal conditions if they wanted something further out.`
            : null;

    return { location: locationLabel, days_returned: daysReturned, days_requested: daysRequested, note, forecast };
}

async function getForecast(location: string, daysRequested: number): Promise<ForecastResult> {
    if (!location || typeof location !== "string") {
        throw new Error("A location is required.");
    }

    const requested = Math.max(1, daysRequested || 7);
    const place = await geocode(location);
    const { daily, daysServed } = await getDailyForecastClamped(place.latitude, place.longitude, requested);
    const label = [place.name, place.admin1, place.country].filter(Boolean).join(", ");

    return buildForecastResult(daily, daysServed, requested, label);
}

async function getForecastByCoords(location: RequestLocation, daysRequested: number): Promise<ForecastResult> {
    const requested = Math.max(1, daysRequested || 7);
    const { daily, daysServed } = await getDailyForecastClamped(location.latitude, location.longitude, requested);
    const label = await reverseGeocodeLabel(location.latitude, location.longitude);

    return buildForecastResult(daily, daysServed, requested, label);
}

async function executeTool(name: string, args: Record<string, unknown>, requestLocation: RequestLocation | undefined): Promise<unknown> {
    switch (name) {
        case "get_weather":
            if (args.use_current_location) {
                return await getWeatherByCoords(requireLocation(requestLocation));
            }
            return await getWeather(args.location as string);
        case "get_forecast":
            if (args.use_current_location) {
                return await getForecastByCoords(requireLocation(requestLocation), Number(args.days) || 7);
            }
            return await getForecast(args.location as string, Number(args.days) || 7);
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

async function callTool(name: string, argsRaw: string | undefined, requestLocation: RequestLocation | undefined): Promise<{ callStep: ToolCallStep; resultStep: ToolResultStep | ToolErrorStep; result: unknown }> {
    const args = parseToolArgs(argsRaw);
    const callStep: ToolCallStep = { type: "tool_call", name, args };

    try {
        const result = await executeTool(name, args, requestLocation);
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
// Updated Dynamic System Prompt for Dynamic Tables & Data Charts
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are Vayuniq, a rich weather assistant. You ONLY help with weather-related questions (current conditions, forecasts, comparisons, metrics).
- Use the get_weather tool for current/real-time conditions.
- Use the get_forecast tool for anything about upcoming days (tomorrow, this weekend, next week, next month, etc.). Pass however many days out the user is asking about — don't refuse just because it sounds far away; the tool itself will tell you how many days it could actually retrieve and why.
- If the tool returns fewer days than requested (see its "note" field), that's because weather models lose reliable accuracy beyond a certain horizon — a physical limit of forecasting, not a limitation of this app. Explain that plainly, share the outlook you do have, and offer typical/seasonal conditions as a substitute for anything further out. Never just say "I can't predict future weather" with no explanation.
- If the user asks about their own current location ("here", "my location", "near me", "where I am"), call get_weather or get_forecast with use_current_location=true and no "location" argument. NEVER invent, assume, or default to any specific city (not New York, not anywhere) for a "current location" request — you have no way to know where someone is unless the tool tells you.
- If the current-location tool call fails because no location was shared, tell the user plainly that their location hasn't been shared yet (e.g. the browser didn't grant permission) and ask them to allow location access or just name the city — never substitute a guessed city instead.
- CRITICAL: after any successful tool call, your reply MUST report the actual values the tool returned — the real temperature, condition, humidity, wind, or forecast values — in plain sentences. A reply that only comments on the mechanics of the call itself (e.g. "the function call was successful", "it seems your location has been shared", "I was able to retrieve the data") without stating what that data actually is is USELESS and NEVER acceptable, even if it's your one and only reply. "Don't narrate your reasoning process" means don't describe your internal steps or which tool you're about to use — it does NOT mean withholding the tool's actual results from the user.
- Every tool result includes a "location" field with the actual resolved place name (e.g. "Bengaluru, Karnataka, India"). ALWAYS name that place explicitly in your reply — say "The weather in Bengaluru is..." not "The weather at your location is...". This matters most for current-location requests: the "location" field is the only way the user finds out where the data is even for, so silently saying "your location" instead of naming it defeats the purpose of asking.
- If the user asks anything that is not about weather, politely decline.
- Think step by step, but only show your final answer to the user — do not narrate your reasoning process.

FORMATTING OUTPUT RULES:
1. TABLES: Whenever comparing locations, displaying timelines, or breakdown weather states, use Markdown Tables.
2. CHARTS: When asked to chart data, map out structural metric lines, or show comparative graphics, append a JSON chart code block at the absolute end of your response inside this token wrapper syntax exactly:
[CHART_DATA]
[
  {"name": "Location/Day A", "temperature": 22, "humidity": 60, "wind": 15},
  {"name": "Location/Day B", "temperature": 18, "humidity": 75, "wind": 25}
]
[/CHART_DATA]
Ensure the keys inside the JSON list align to: "name", "temperature", "humidity", or "wind". Keep conversational summary text brief outside the token block.`;

const OFF_TOPIC_REPLY = "I can only help with weather questions — try asking about the current conditions or forecast for a place.";

const WEATHER_KEYWORDS = [
    "weather", "temperature", "temp", "forecast", "rain", "rainy", "snow", "snowy",
    "wind", "windy", "humidity", "humid", "sunny", "sun", "cloud", "cloudy", "storm",
    "thunderstorm", "hail", "fog", "foggy", "climate", "hot", "cold", "warm", "cool",
    "degrees", "celsius", "fahrenheit", "umbrella", "jacket", "drizzle", "overcast",
    "clear sky", "feels like", "chart", "graph", "plot"
];

function isWeatherRelated(text: string): boolean {
    const lower = text.toLowerCase();
    return WEATHER_KEYWORDS.some((kw) => lower.includes(kw));
}

function latestUserMessage(conversation: ChatMessage[]): string {
    for (let i = conversation.length - 1; i >= 0; i--) {
        if (conversation[i].role === "user" && typeof conversation[i].content === "string") {
            return conversation[i].content as string;
        }
    }
    return "";
}

// Looks at the whole conversation so far, not just the newest message. This
// lets contextual follow-ups like "what about next month" or "and tomorrow?"
// pass the gate once weather context is established, instead of being
// rejected just because that one sentence lacks a keyword. Scanning the full
// conversation (rather than an arbitrary recent-N window) also avoids a
// long-running chat sliding past its own established context and getting
// incorrectly gated again later on.
function isConversationWeatherRelated(conversation: ChatMessage[]): boolean {
    const allText = conversation
        .filter((m) => typeof m.content === "string")
        .map((m) => m.content as string)
        .join(" ");

    return isWeatherRelated(allText);
}

const STEP_LIMIT_MESSAGE = "I wasn't able to finish reasoning about that within my step limit — try breaking the question down.";

async function runAgent(conversation: ChatMessage[], location: RequestLocation | undefined): Promise<AgentResult> {
    if (!isConversationWeatherRelated(conversation)) {
        return { reply: OFF_TOPIC_REPLY, trace: [], model: MODEL };
    }

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
            const { callStep, resultStep, result } = await callTool(call.function.name, call.function.arguments, location);
            trace.push(callStep, resultStep);
            messages.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(result) });
        }
    }

    return { reply: STEP_LIMIT_MESSAGE, trace, model: MODEL };
}

async function runAgentStream(conversation: ChatMessage[], emit: Emit, location: RequestLocation | undefined): Promise<void> {
    if (!isConversationWeatherRelated(conversation)) {
        emit("token", { content: OFF_TOPIC_REPLY });
        emit("done", { reply: OFF_TOPIC_REPLY, trace: [], model: MODEL });
        return;
    }

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
            const { callStep, resultStep, result } = await callTool(tc.name, tc.arguments, location);
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
// Express Routes Setup
// ---------------------------------------------------------------------------

function isValidMessages(body: unknown): body is { messages: ChatMessage[] } {
    return (
        typeof body === "object" &&
        body !== null &&
        Array.isArray((body as { messages?: unknown }).messages) &&
        (body as { messages: unknown[] }).messages.length > 0
    );
}

// The client sends this only if the browser's Geolocation API succeeded and
// the user granted permission. Absent/malformed input is just treated as
// "no location available" — the model is instructed never to guess instead.
function extractLocation(body: unknown): RequestLocation | undefined {
    const raw = (body as { location?: unknown } | null)?.location as { latitude?: unknown; longitude?: unknown } | undefined;
    if (!raw || typeof raw.latitude !== "number" || typeof raw.longitude !== "number") return undefined;
    if (Number.isNaN(raw.latitude) || Number.isNaN(raw.longitude)) return undefined;
    return { latitude: raw.latitude, longitude: raw.longitude };
}

app.get("/api/health", (req: Request, res: Response) => {
    res.json({ ok: true, model: MODEL, keyConfigured: Boolean(process.env.GROQ_API_KEY) });
});

app.post("/api/chat", async (req: Request, res: Response) => {
    if (!isValidMessages(req.body)) {
        return res.status(400).json({ error: "Request body must include a non-empty 'messages' array." });
    }

    try {
        const result = await runAgent(req.body.messages, extractLocation(req.body));
        res.json(result);
    } catch (err) {
        console.error("Agent error:", err);
        const message = err instanceof Error ? err.message : "The agent failed to respond.";
        res.status(500).json({ error: message });
    }
});

app.post("/api/chat/stream", async (req: Request, res: Response) => {
    if (!isValidMessages(req.body)) {
        return res.status(400).json({ error: "Request body must include a non-empty 'messages' array." });
    }

    res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
    });
    res.flushHeaders?.();

    const emit: Emit = (event, data) => {
        res.write(`event: ${event}\n`);
        res.write(`data: ${JSON.stringify(data)}\n\n`);

        if (typeof (res as any).flush === "function") {
            (res as any).flush();
        }
    };

    const heartbeat = setInterval(() => res.write(": ping\n\n"), 15000);

    try {
        await runAgentStream(req.body.messages, emit, extractLocation(req.body));
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
    console.log(`Vayuniq Weather Agent server running at http://localhost:${PORT}`);
});