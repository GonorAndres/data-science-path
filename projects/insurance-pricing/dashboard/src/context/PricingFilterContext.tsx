"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                             */
/* ------------------------------------------------------------------ */

interface PricingFilterState {
  activeTab: string;
  selectedFeature: string;
  selectedModel: string;
}

interface PricingFilterActions {
  setActiveTab: (tab: string) => void;
  setSelectedFeature: (feature: string) => void;
  setSelectedModel: (model: string) => void;
}

type PricingFilterContextValue = PricingFilterState & PricingFilterActions;

/* ------------------------------------------------------------------ */
/*  Defaults                                                          */
/* ------------------------------------------------------------------ */

const DEFAULTS: PricingFilterState = {
  activeTab: "contexto",
  selectedFeature: "BonusMalus",
  selectedModel: "xgboost",
};

/* ------------------------------------------------------------------ */
/*  Context                                                           */
/* ------------------------------------------------------------------ */

const PricingFilterContext = createContext<PricingFilterContextValue | null>(
  null
);

/* ------------------------------------------------------------------ */
/*  Provider                                                          */
/* ------------------------------------------------------------------ */

export function PricingFilterProvider({ children }: { children: ReactNode }) {
  const [activeTab, setActiveTabRaw] = useState(DEFAULTS.activeTab);
  const [selectedFeature, setSelectedFeatureRaw] = useState(
    DEFAULTS.selectedFeature
  );
  const [selectedModel, setSelectedModelRaw] = useState(
    DEFAULTS.selectedModel
  );

  const setActiveTab = useCallback((tab: string) => setActiveTabRaw(tab), []);
  const setSelectedFeature = useCallback(
    (feature: string) => setSelectedFeatureRaw(feature),
    []
  );
  const setSelectedModel = useCallback(
    (model: string) => setSelectedModelRaw(model),
    []
  );

  return (
    <PricingFilterContext.Provider
      value={{
        activeTab,
        selectedFeature,
        selectedModel,
        setActiveTab,
        setSelectedFeature,
        setSelectedModel,
      }}
    >
      {children}
    </PricingFilterContext.Provider>
  );
}

/* ------------------------------------------------------------------ */
/*  Hook                                                              */
/* ------------------------------------------------------------------ */

export function usePricingFilter(): PricingFilterContextValue {
  const ctx = useContext(PricingFilterContext);
  if (!ctx) {
    throw new Error(
      "usePricingFilter debe usarse dentro de <PricingFilterProvider>"
    );
  }
  return ctx;
}
