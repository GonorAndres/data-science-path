'use client'
import { motion } from 'framer-motion'

interface Tab {
  id: string
  label: string
}

interface TabNavProps {
  tabs: Tab[]
  activeTab: string
  onTabChange: (id: string) => void
}

export function TabNav({ tabs, activeTab, onTabChange }: TabNavProps) {
  return (
    <nav className="py-6 flex flex-wrap gap-2">
      {tabs.map((tab) => {
        const isActive = tab.id === activeTab
        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`relative font-sans text-xs uppercase tracking-widest px-5 py-2.5 rounded-full transition-colors ${
              isActive
                ? 'text-paper dark:text-ink'
                : 'text-muted hover:bg-surface/60 dark:hover:bg-[#2a2a2a]'
            }`}
          >
            {isActive && (
              <motion.div
                layoutId="activeTab"
                className="absolute inset-0 bg-ink dark:bg-[#F0EFEB] rounded-full"
                transition={{ type: 'spring', stiffness: 380, damping: 30 }}
              />
            )}
            <span className="relative z-10">{tab.label}</span>
          </button>
        )
      })}
    </nav>
  )
}
