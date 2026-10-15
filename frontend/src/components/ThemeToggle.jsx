import React from 'react'

export default function ThemeToggle({ theme, onChange }) {
  const isDark = theme === 'dark'
  return <button className="topbar-toggle" type="button" onClick={() => onChange(isDark ? 'light' : 'dark')} aria-label={`Switch to ${isDark ? 'light' : 'dark'} theme`} title="Toggle color theme"><svg className="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">{isDark ? <><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></> : <path d="M20.5 15.2A8.5 8.5 0 0 1 8.8 3.5 8.5 8.5 0 1 0 20.5 15.2Z"/>}</svg><small>{isDark ? 'Dark' : 'Light'}</small></button>
}
