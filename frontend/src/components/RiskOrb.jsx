import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'

export default function RiskOrb({ probability = 0, active = false, scoring = false }) {
  const mountRef = useRef(null)
  const probabilityRef = useRef(probability)
  const activeRef = useRef(active)
  const scoringRef = useRef(scoring)
  useEffect(() => { probabilityRef.current = probability }, [probability])
  useEffect(() => { activeRef.current = active }, [active])
  useEffect(() => { scoringRef.current = scoring }, [scoring])

  useEffect(() => {
    const host = mountRef.current
    if (!host) return undefined
    const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)')
    let reducedMotion = motionPreference.matches || document.documentElement.dataset.reduceMotion === 'true'
    let renderer
    try {
      renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' })
    } catch {
      host.dataset.webgl = 'unavailable'
      return undefined
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7))
    renderer.setClearColor(0x000000, 0)
    host.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100)
    camera.position.set(0, 0, 8)
    const root = new THREE.Group()
    scene.add(root)

    const coreGeometry = new THREE.IcosahedronGeometry(1.35, 5)
    const coreMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x69e2af,
      metalness: 0.18,
      roughness: 0.15,
      transmission: 0.12,
      clearcoat: 1,
      clearcoatRoughness: 0.08,
      emissive: 0x104d3b,
      emissiveIntensity: 0.8,
    })
    const core = new THREE.Mesh(coreGeometry, coreMaterial)
    root.add(core)

    const wireframe = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.43, 2),
      new THREE.MeshBasicMaterial({ color: 0x8cebc5, wireframe: true, transparent: true, opacity: 0.14 }),
    )
    root.add(wireframe)

    const rings = []
    for (let index = 0; index < 3; index += 1) {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(1.76 + index * 0.17, 0.006, 4, 160),
        new THREE.MeshBasicMaterial({ color: 0x68dcae, transparent: true, opacity: 0.25 - index * 0.045 }),
      )
      ring.rotation.set(0.55 + index * 0.65, 0.25 + index * 0.58, index * 0.3)
      root.add(ring)
      rings.push(ring)
    }
    const scoreRingMaterial = new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.8 })
    const scoreRing = new THREE.Mesh(new THREE.TorusGeometry(1.9, 0.012, 5, 160), scoreRingMaterial)
    scoreRing.rotation.set(1.15, 0.3, 0)
    scoreRing.visible = false
    root.add(scoreRing)
    const shockwaveMaterial = new THREE.MeshBasicMaterial({ color: 0xf43f5e, transparent: true, opacity: 0, side: THREE.DoubleSide })
    const shockwave = new THREE.Mesh(new THREE.TorusGeometry(1.5, 0.025, 6, 80), shockwaveMaterial)
    shockwave.rotation.x = Math.PI / 2
    shockwave.visible = false
    root.add(shockwave)

    const particleCount = 220
    const positions = new Float32Array(particleCount * 3)
    for (let index = 0; index < particleCount; index += 1) {
      const radius = 2.1 + Math.random() * 2.4
      const theta = Math.random() * Math.PI * 2
      const vertical = Math.random() * 2 - 1
      const planar = Math.sqrt(1 - vertical * vertical)
      positions[index * 3] = radius * planar * Math.cos(theta)
      positions[index * 3 + 1] = radius * vertical
      positions[index * 3 + 2] = radius * planar * Math.sin(theta)
    }
    const particleGeometry = new THREE.BufferGeometry()
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const particles = new THREE.Points(
      particleGeometry,
      new THREE.PointsMaterial({ color: 0x84eac5, size: 0.022, transparent: true, opacity: 0.72, sizeAttenuation: true }),
    )
    scene.add(particles)

    const ambient = new THREE.AmbientLight(0x9debd1, 1.5)
    const keyLight = new THREE.PointLight(0x78edc1, 22, 12)
    keyLight.position.set(3, 2, 4)
    const rimLight = new THREE.PointLight(0x527df5, 11, 10)
    rimLight.position.set(-3, -2, -3)
    scene.add(ambient, keyLight, rimLight)

    const resize = () => {
      const width = Math.max(host.clientWidth, 1)
      const height = Math.max(host.clientHeight, 1)
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
    }
    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(host)
    resize()

    let frame = 0
    let visible = true
    let lastProbability = 0
    let shockStarted = -1
    const clock = new THREE.Clock()
    const riskColor = new THREE.Color()
    const animate = () => {
      frame = 0
      if (!visible || document.hidden) return
      const time = reducedMotion ? 0 : clock.getElapsedTime()
      const score = Math.max(0, Math.min(1, probabilityRef.current))
      const riskHue = 0.34 * (1 - score)
      riskColor.setHSL(riskHue, 0.78, 0.52 - score * 0.06)
      coreMaterial.color.lerp(riskColor, 0.055)
      coreMaterial.emissive.copy(riskColor).multiplyScalar(0.28)
      coreMaterial.roughness = scoringRef.current && !reducedMotion ? 0.12 + (Math.sin(time * 5) + 1) * 0.07 : 0.15
      particles.material.color.lerp(riskColor, 0.035)
      const pulse = activeRef.current && !reducedMotion ? Math.sin(time * (1.5 + score * 7)) * 0.04 : 0
      const breath = !activeRef.current && !reducedMotion ? 1 + Math.sin(time * 0.8) * 0.03 : 1
      core.scale.setScalar((breath + pulse) * (1 + (activeRef.current ? score * 0.04 : 0)))
      root.rotation.y = time * 0.12
      root.rotation.x = Math.sin(time * 0.17) * 0.08
      wireframe.rotation.copy(root.rotation)
      rings.forEach((ring, index) => {
        if (!reducedMotion) {
          ring.rotation.z += (index % 2 ? -1 : 1) * 0.0018
          ring.rotation.x += 0.0006
        }
      })
      scoreRing.visible = scoringRef.current && !reducedMotion
      if (scoreRing.visible) { scoreRing.rotation.z += 0.018; scoreRing.rotation.y += 0.004 }
      if (score > 0.995 && lastProbability <= 0.995 && !reducedMotion) { shockStarted = time; shockwave.visible = true; shockwave.scale.setScalar(0.55) }
      if (shockwave.visible) {
        const elapsed = time - shockStarted
        shockwave.scale.setScalar(0.55 + Math.min(1, elapsed / 0.8) * 0.95)
        shockwaveMaterial.opacity = Math.max(0, 0.8 * (1 - elapsed / 0.8))
        if (elapsed >= 0.8) shockwave.visible = false
      }
      lastProbability = score
      if (!reducedMotion) particles.rotation.y = -time * 0.025
      renderer.render(scene, camera)
      frame = window.requestAnimationFrame(animate)
    }
    const schedule = () => { if (visible && !document.hidden && !frame) frame = window.requestAnimationFrame(animate) }
    const intersection = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; if (visible) schedule(); else { window.cancelAnimationFrame(frame); frame = 0 } })
    intersection.observe(host)
    const onVisibility = () => { if (document.hidden) { window.cancelAnimationFrame(frame); frame = 0 } else schedule() }
    const onMotionChange = (event) => { reducedMotion = event.matches || document.documentElement.dataset.reduceMotion === 'true' }
    const motionObserver = new MutationObserver(() => { reducedMotion = motionPreference.matches || document.documentElement.dataset.reduceMotion === 'true' })
    motionObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-reduce-motion'] })
    motionPreference.addEventListener?.('change', onMotionChange)
    document.addEventListener('visibilitychange', onVisibility)
    schedule()

    return () => {
      window.cancelAnimationFrame(frame)
      resizeObserver.disconnect()
      intersection.disconnect()
      motionObserver.disconnect()
      motionPreference.removeEventListener?.('change', onMotionChange)
      document.removeEventListener('visibilitychange', onVisibility)
      coreGeometry.dispose()
      coreMaterial.dispose()
      wireframe.geometry.dispose()
      wireframe.material.dispose()
      rings.forEach((ring) => { ring.geometry.dispose(); ring.material.dispose() })
      scoreRing.geometry.dispose(); scoreRingMaterial.dispose()
      shockwave.geometry.dispose(); shockwaveMaterial.dispose()
      particleGeometry.dispose()
      particles.material.dispose()
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [])

  return <div className="risk-orb" data-scoring={scoring} ref={mountRef} role="img" aria-label="Animated three-dimensional transaction risk visual">
    <div className="orb-fallback" aria-hidden="true"><span /><i /><b>◎</b></div>
  </div>
}
