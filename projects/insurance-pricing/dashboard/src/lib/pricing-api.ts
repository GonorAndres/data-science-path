const BASE_URL = "/api/pricing";

export class PricingAPIError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "PricingAPIError";
    this.status = status;
  }
}

export async function pricingFetcher<T = unknown>(
  endpoint: string
): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const res = await fetch(url);

  if (!res.ok) {
    const body = await res.text().catch(() => "Unknown error");
    throw new PricingAPIError(
      `API error ${res.status}: ${body}`,
      res.status
    );
  }

  return res.json() as Promise<T>;
}
