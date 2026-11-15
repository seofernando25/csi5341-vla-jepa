/* Read-only API client. Demo mode swaps in js/demo.js; callers never branch on it. */

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function qs(params) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value != null) search.set(key, String(value));
  }
  return search.toString();
}

async function getJson(path) {
  let response;
  try {
    response = await fetch(path, { cache: "no-store" });
  } catch (error) {
    throw new ApiError(`network error: ${error.message}`, 0);
  }
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const message = body && body.error ? body.error : `HTTP ${response.status}`;
    throw new ApiError(message, response.status);
  }
  if (body == null) throw new ApiError("malformed JSON response", response.status);
  return body;
}

export function createApi({ demo = false } = {}) {
  if (demo) return createDemoApi();
  return {
    demo: false,
    state: () => getJson("/api/state"),
    events: (source, after = 0) => getJson(`/api/events?${qs({ source, after })}`),
    artifact: (source, name) => getJson(`/api/artifact?${qs({ source, name })}`),
    context: (source) => getJson(`/api/context?${qs({ source })}`),
    dream: (cycle, world) => getJson(`/api/dream?${qs({ cycle, world })}`),
  };
}

/* Loaded lazily so the live page never downloads the fixtures. */
async function createDemoApi() {
  const { createDemo } = await import("./demo.js");
  const demo = await createDemo();
  return { demo: true, ...demo };
}
