import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

export default function ThreeCanvas({ createScene, label, fallback, className = '' }) {
  const hostRef = useRef(null)
  const [available, setAvailable] = useState(true)
  const factoryRef = useRef(createScene)
  factoryRef.current = createScene

  useEffect(() => {
    const host = hostRef.current
    if (!host) return undefined
    let renderer
    try {
      renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' })
    } catch {
      host.dataset.webgl = 'unavailable'
      setAvailable(false)
      return undefined
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setClearColor(0x000000, 0)
    host.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100)
    camera.position.set(0, 0, 8)
    const controller = factoryRef.current?.({ THREE, scene, camera, renderer, host }) || {}
    let visible = true
    let frame = 0
    let previous = 0
    let started = 0
    const resize = () => {
      const width = Math.max(host.clientWidth, 1)
      const height = Math.max(host.clientHeight, 1)
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
    }
    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(host)
    const intersection = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; if (visible) schedule() })
    intersection.observe(host)
    const schedule = () => {
      if (!visible || document.hidden || frame) return
      frame = requestAnimationFrame(draw)
    }
    const draw = (now) => {
      frame = 0
      if (!visible || document.hidden) return
      if (!started) started = now
      const delta = Math.min((now - (previous || now)) / 1000, 0.05)
      previous = now
      controller.update?.((now - started) / 1000, delta)
      renderer.render(scene, camera)
      schedule()
    }
    const onVisibility = () => { if (document.hidden) { cancelAnimationFrame(frame); frame = 0 } else schedule() }
    document.addEventListener('visibilitychange', onVisibility)
    resize()
    schedule()
    return () => {
      cancelAnimationFrame(frame)
      document.removeEventListener('visibilitychange', onVisibility)
      resizeObserver.disconnect()
      intersection.disconnect()
      controller.dispose?.()
      scene.traverse((object) => {
        object.geometry?.dispose?.()
        const materials = Array.isArray(object.material) ? object.material : [object.material]
        materials.filter(Boolean).forEach((material) => {
          Object.values(material).forEach((value) => { if (value?.isTexture) value.dispose() })
          material.dispose?.()
        })
      })
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [])

  return <div className={`three-canvas ${className}`} ref={hostRef} role="img" aria-label={label}>
    {!available && <div className="three-fallback">{fallback}</div>}
  </div>
}
