export interface ApiConfig {
  baseUrl: string;
  token: string;
}

export function readApiConfig(): ApiConfig {
  return {
    baseUrl: (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, ""),
    token: import.meta.env.VITE_DEMO_TOKEN ?? "",
  };
}
