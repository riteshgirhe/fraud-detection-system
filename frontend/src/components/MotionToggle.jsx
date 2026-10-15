import React from 'react'

export default function MotionToggle({ reduced, onChange }) {
  return <button className="topbar-toggle motion-toggle" type="button" aria-pressed={reduced} onClick={() => onChange(!reduced)} title="Toggle reduced motion"><svg className="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">{reduced ? <><path d="M8 5v14M16 5v14"/><circle cx="12" cy="12" r="10"/></> : <><path d="M3 12h3l2-6 4 12 3-8 2 4h4"/></>}</svg><small>{reduced ? 'Motion off' : 'Motion on'}</small></button>
}
