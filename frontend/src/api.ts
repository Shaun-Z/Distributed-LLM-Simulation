export class ApiError extends Error {
  constructor(public detail: unknown) {
    super(
      typeof detail === "string" ? detail : JSON.stringify(detail, null, 2),
    );
  }
}
export async function api<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch("/api" + path, {
    method: body === undefined ? "GET" : "POST",
    headers:
      body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    let data;
    try {
      data = await response.json();
    } catch {
      throw new ApiError(response.statusText);
    }
    throw new ApiError(data.detail ?? data);
  }
  return response.json();
}
export function download(name: string, text: string) {
  const url = URL.createObjectURL(
    new Blob([text], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
