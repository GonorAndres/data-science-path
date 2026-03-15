'use client'

import { useMemo } from 'react'
import { LoadingSpinner } from '../ui/LoadingSpinner'

interface HeatmapCell {
  veh_age: number
  driv_age: number
  frequency: number
}

interface ClaimFrequencyHeatmapProps {
  data: HeatmapCell[]
  isLoading: boolean
}

function colorScale(value: number, maxValue: number): string {
  if (maxValue === 0) return '#F0EFEB'
  const ratio = Math.min(value / maxValue, 1)
  const r = Math.round(240 + (180 - 240) * ratio)
  const g = Math.round(239 + (40 - 239) * ratio)
  const b = Math.round(235 + (40 - 235) * ratio)
  return `rgb(${r}, ${g}, ${b})`
}

export function ClaimFrequencyHeatmap({ data, isLoading }: ClaimFrequencyHeatmapProps) {
  const { vehAges, drivAges, freqMap, maxFreq } = useMemo(() => {
    if (!data || data.length === 0) {
      return { vehAges: [] as number[], drivAges: [] as number[], freqMap: new Map<string, number>(), maxFreq: 0 }
    }

    const vehSet = new Set<number>()
    const drivSet = new Set<number>()
    const fMap = new Map<string, number>()
    let max = 0

    for (const cell of data) {
      vehSet.add(cell.veh_age)
      drivSet.add(cell.driv_age)
      const key = `${cell.veh_age}-${cell.driv_age}`
      fMap.set(key, cell.frequency)
      if (cell.frequency > max) max = cell.frequency
    }

    return {
      vehAges: Array.from(vehSet).sort((a, b) => a - b),
      drivAges: Array.from(drivSet).sort((a, b) => a - b),
      freqMap: fMap,
      maxFreq: max,
    }
  }, [data])

  if (isLoading) return <LoadingSpinner />

  return (
    <div className="w-full overflow-x-auto">
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `80px repeat(${vehAges.length}, 1fr)`,
          gap: '1px',
          fontSize: '11px',
        }}
      >
        {/* Top-left corner */}
        <div
          style={{
            padding: '6px 4px',
            fontWeight: 600,
            fontSize: '10px',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: '#6b6b6b',
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'center',
          }}
        >
          Edad veh.
        </div>

        {/* Column headers: vehicle age */}
        {vehAges.map((vAge) => (
          <div
            key={`col-${vAge}`}
            style={{
              padding: '6px 4px',
              textAlign: 'center',
              fontWeight: 600,
              fontSize: '11px',
              color: '#6b6b6b',
            }}
          >
            {vAge}
          </div>
        ))}

        {/* Rows */}
        {drivAges.map((dAge) => (
          <>
            {/* Row header: driver age */}
            <div
              key={`row-${dAge}`}
              style={{
                padding: '6px 4px',
                fontWeight: 600,
                fontSize: '11px',
                color: '#6b6b6b',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                whiteSpace: 'nowrap',
              }}
            >
              {dAge} conductor
            </div>

            {/* Data cells */}
            {vehAges.map((vAge) => {
              const key = `${vAge}-${dAge}`
              const freq = freqMap.get(key) ?? 0
              const bg = colorScale(freq, maxFreq)
              const textColor = freq / maxFreq > 0.6 ? '#F0EFEB' : '#1a1a1a'

              return (
                <div
                  key={`cell-${key}`}
                  style={{
                    backgroundColor: bg,
                    padding: '8px 4px',
                    textAlign: 'center',
                    color: textColor,
                    borderRadius: '2px',
                    fontVariantNumeric: 'tabular-nums',
                  }}
                  title={`Edad vehículo: ${vAge}, Edad conductor: ${dAge}, Frecuencia: ${freq.toFixed(4)}`}
                >
                  {freq.toFixed(3)}
                </div>
              )
            })}
          </>
        ))}
      </div>

      {/* Legend */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginTop: '12px',
          fontSize: '11px',
          color: '#6b6b6b',
        }}
      >
        <span>Baja frecuencia</span>
        <div
          style={{
            width: '120px',
            height: '12px',
            borderRadius: '2px',
            background: `linear-gradient(to right, #F0EFEB, rgb(180, 40, 40))`,
          }}
        />
        <span>Alta frecuencia</span>
      </div>
    </div>
  )
}
