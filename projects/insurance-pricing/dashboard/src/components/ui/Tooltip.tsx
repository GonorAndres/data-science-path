'use client'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface TooltipProps {
  term: string
  children: React.ReactNode
}

export function Tooltip({ term, children }: TooltipProps) {
  const [open, setOpen] = useState(false)

  return (
    <span
      className="relative inline-block"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onClick={() => setOpen((v) => !v)}
    >
      <span className="border-b border-dotted border-muted cursor-help">
        {term}
      </span>
      <AnimatePresence>
        {open && (
          <motion.span
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 bg-ink dark:bg-[#F0EFEB] text-paper dark:text-ink rounded-lg px-3 py-2 text-sm shadow-lg max-w-xs w-max font-sans font-normal not-italic normal-case tracking-normal pointer-events-none"
          >
            {children}
            <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-ink dark:border-t-[#F0EFEB]" />
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  )
}
