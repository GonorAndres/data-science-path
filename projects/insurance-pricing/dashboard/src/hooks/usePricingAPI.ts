import useSWR, { SWRConfiguration } from "swr";
import { pricingFetcher } from "@/lib/pricing-api";

/* ------------------------------------------------------------------ */
/*  Shared SWR config: conservative revalidation, generous retries    */
/* ------------------------------------------------------------------ */

const defaultConfig: SWRConfiguration = {
  revalidateOnFocus: false,
  dedupingInterval: 60_000,
  errorRetryCount: 10,
  errorRetryInterval: 3_000,
};

function usePricing<T = unknown>(
  endpoint: string | null,
  config?: SWRConfiguration
) {
  return useSWR<T>(endpoint, pricingFetcher, {
    ...defaultConfig,
    ...config,
  });
}

/* ------------------------------------------------------------------ */
/*  Overview                                                          */
/* ------------------------------------------------------------------ */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyData = Record<string, any>

export function useOverview() {
  return usePricing<AnyData>("/api/v1/overview");
}

/* ------------------------------------------------------------------ */
/*  EDA endpoints                                                     */
/* ------------------------------------------------------------------ */

export function useDistributions(feature?: string) {
  const params = feature ? `?feature=${encodeURIComponent(feature)}` : "";
  return usePricing<AnyData>(`/api/v1/eda/distributions${params}`);
}

export function useCorrelations() {
  return usePricing<AnyData>("/api/v1/eda/correlations");
}

export function useExposure() {
  return usePricing<AnyData>("/api/v1/eda/exposure");
}

export function useClaimFrequency() {
  return usePricing<AnyData>("/api/v1/eda/claim-frequency");
}

/* ------------------------------------------------------------------ */
/*  Model comparison & diagnostics                                    */
/* ------------------------------------------------------------------ */

export function useModelComparison() {
  return usePricing<AnyData>("/api/v1/models/comparison");
}

export function useLiftChart() {
  return usePricing<AnyData>("/api/v1/models/lift-chart");
}

export function useCalibration() {
  return usePricing<AnyData>("/api/v1/models/calibration");
}

export function useLorenzCurves() {
  return usePricing<AnyData>("/api/v1/models/lorenz-curves");
}

export function useCoefficients() {
  return usePricing<AnyData>("/api/v1/models/coefficients");
}

/* ------------------------------------------------------------------ */
/*  SHAP & interpretability                                           */
/* ------------------------------------------------------------------ */

export function useShapSummary() {
  return usePricing<AnyData>("/api/v1/models/shap-summary");
}

export function useShapDependence(feature?: string, model?: string) {
  if (!feature || !model) return usePricing<AnyData>(null);
  const params = new URLSearchParams({ feature, model }).toString();
  return usePricing<AnyData>(`/api/v1/models/shap-dependence?${params}`);
}

/* ------------------------------------------------------------------ */
/*  Pricing & fairness                                                */
/* ------------------------------------------------------------------ */

export function usePurePremium() {
  return usePricing<AnyData>("/api/v1/models/pure-premium");
}

export function useDoubleLift() {
  return usePricing<AnyData>("/api/v1/models/double-lift");
}

export function useFairnessAudit() {
  return usePricing<AnyData>("/api/v1/models/fairness-audit");
}

/* ------------------------------------------------------------------ */
/*  Feature importance & MLflow                                       */
/* ------------------------------------------------------------------ */

export function useFeatureImportance() {
  return usePricing<AnyData>("/api/v1/models/feature-importance");
}

export function useMlflowSummary() {
  return usePricing<AnyData>("/api/v1/models/mlflow-summary");
}
