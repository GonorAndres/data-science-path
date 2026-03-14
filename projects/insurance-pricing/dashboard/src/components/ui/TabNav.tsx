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
    <nav className="border-b border-border dark:border-[#2a2a2a] flex gap-6">
      {tabs.map((tab) => {
        const isActive = tab.id === activeTab
        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`relative pb-3 font-sans text-xs uppercase tracking-widest transition-colors ${
              isActive
                ? 'text-ink dark:text-[#F0EFEB]'
                : 'text-muted hover:text-ink dark:hover:text-[#F0EFEB]'
            }`}
          >
            {tab.label}
            {isActive && (
              <motion.div
                layoutId="activeTab"
                className="absolute bottom-0 left-0 right-0 h-px bg-ink dark:bg-[#F0EFEB]"
                transition={{ type: 'spring', stiffness: 380, damping: 30 }}
              />
            )}
          </button>
        )
      })}
    </nav>
  )
}
