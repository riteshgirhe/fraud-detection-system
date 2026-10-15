import React, { useEffect, useState } from 'react'

export default function CountUp({ value, delay = 0 }) {
  const text = String(value ?? '—')
  const numeric = Number(text.replaceAll(',', '').match(/-?\d+(?:\.\d+)?/)?.[0])
  const [display, setDisplay] = useState(text)
  useEffect(() => {
    if (!Number.isFinite(numeric) || window.matchMedia('(prefers-reduced-motion: reduce)').matches || document.documentElement.dataset.reduceMotion === 'true') { setDisplay(text); return undefined }
    const match = text.match(/(-?\d[\d,]*(?:\.\d+)?)/)
    if (!match) { setDisplay(text); return undefined }
    const prefix = text.slice(0, match.index)
    const suffix = text.slice(match.index + match[0].length)
    const decimals = match[0].split('.').at(1)?.length || 0
    const start = performance.now()
    let frame
    const tick = (now) => {
      const progress = Math.min(1, Math.max(0, (now - start - delay) / 650))
      const eased = 1 - (1 - progress) ** 3
      const current = (numeric * eased).toFixed(decimals)
      const formatted = Number(current).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
      setDisplay(`${prefix}${formatted}${suffix}`)
      if (progress < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [text, numeric, delay])
  return <span aria-label={text}>{display}</span>
}
