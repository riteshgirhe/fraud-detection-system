import React, { useMemo } from 'react'
import ThreeCanvas from './ThreeCanvas.jsx'

export default function PredictionScatter({ predictions = [] }) {
  const points = predictions.slice(0, 48)
  const signature = points.map((point) => point.transaction_id).join('|')
  const createScene = useMemo(() => ({ THREE, scene, camera, host }) => {
    camera.position.set(0, 0, 8)
    scene.add(new THREE.AmbientLight(0xffffff, 2.1))
    const light = new THREE.PointLight(0x22d3ee, 26, 18)
    light.position.set(2, 3, 6)
    scene.add(light)
    const group = new THREE.Group()
    scene.add(group)
    const entries = points.map((row, index) => {
      const risk = Math.max(0, Math.min(1, Number(row.fraud_probability) || 0))
      const phi = Math.acos(1 - 2 * ((index + 0.5) / Math.max(points.length, 1)))
      const theta = Math.PI * (1 + Math.sqrt(5)) * index
      const target = new THREE.Vector3(2.25 * Math.cos(theta) * Math.sin(phi), 2.25 * Math.sin(theta) * Math.sin(phi), 2.25 * Math.cos(phi))
      const hue = row.is_fraud ? 0xf43f5e : risk > 0.8 ? 0xfbbf24 : 0x34d399
      const material = new THREE.MeshStandardMaterial({ color: hue, emissive: hue, emissiveIntensity: 0.5, roughness: 0.3 })
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(row.is_fraud ? 0.12 : 0.085, 12, 10), material)
      mesh.position.set(0, 0, 0)
      mesh.userData = { row, risk, target, index }
      group.add(mesh)
      if (row.is_fraud) {
        const halo = new THREE.Mesh(new THREE.TorusGeometry(0.18, 0.009, 5, 28), new THREE.MeshBasicMaterial({ color: 0xf43f5e, transparent: true, opacity: 0.65 }))
        halo.userData.halo = true
        mesh.add(halo)
      }
      return mesh
    })
    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    let hovered = null
    let paused = false
    let dragging = false
    let lastX = 0
    const onMove = (event) => {
      if (dragging) { group.rotation.y += (event.clientX - lastX) * 0.008; lastX = event.clientX; return }
      const bounds = host.getBoundingClientRect()
      pointer.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1)
      raycaster.setFromCamera(pointer, camera)
      hovered = raycaster.intersectObjects(entries, false)[0]?.object || null
      paused = Boolean(hovered)
      host.dataset.tooltip = hovered ? `${hovered.userData.row.type} · ${(hovered.userData.risk * 100).toFixed(1)}% risk · ${hovered.userData.row.is_fraud ? 'FRAUD' : 'LEGITIMATE'}` : ''
      host.classList.toggle('has-hover', Boolean(hovered))
    }
    const onDown = (event) => { dragging = true; paused = true; lastX = event.clientX; host.setPointerCapture?.(event.pointerId) }
    const onUp = () => { dragging = false; paused = Boolean(hovered) }
    const onLeave = () => { dragging = false; hovered = null; paused = false; host.dataset.tooltip = ''; host.classList.remove('has-hover') }
    host.addEventListener('pointermove', onMove)
    host.addEventListener('pointerdown', onDown)
    host.addEventListener('pointerup', onUp)
    host.addEventListener('pointerleave', onLeave)
    const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches || document.documentElement.dataset.reduceMotion === 'true'
    return {
      update: (time) => {
        const reduced = reducedMotion()
        entries.forEach((mesh, index) => {
          const { target } = mesh.userData
          const stagger = reduced ? 1 : Math.max(0, Math.min(1, (time - index * 0.018) / 0.9))
          const ease = 1 - (1 - stagger) ** 3
          mesh.position.copy(target).multiplyScalar(ease)
          const selected = mesh === hovered
          const pulse = !reduced && mesh.userData.row.is_fraud ? 1 + Math.sin(time * 2.2 + index) * 0.1 : 1
          mesh.scale.setScalar(selected ? 1.65 : pulse)
          mesh.material.opacity = hovered && !selected ? 0.28 : 1
          mesh.material.transparent = Boolean(hovered && !selected)
          if (!reduced && mesh.children[0]) mesh.children[0].rotation.z = Math.sin(time * 1.7 + index) * 0.4
        })
        if (!reduced && !paused) group.rotation.y = time * 0.055
      },
      dispose: () => { host.removeEventListener('pointermove', onMove); host.removeEventListener('pointerdown', onDown); host.removeEventListener('pointerup', onUp); host.removeEventListener('pointerleave', onLeave); delete host.dataset.tooltip },
    }
  }, [signature])
  const fallback = points.length ? <div className="scatter-fallback">{points.slice(0, 12).map((point) => <span key={point.transaction_id} className={point.is_fraud ? 'risk-danger' : 'risk-safe'} title={`${point.type}: ${(point.fraud_probability * 100).toFixed(1)}%`} />)}</div> : 'Score transactions to see prediction points.'
  return <div className="scatter-wrap"><ThreeCanvas key={signature} createScene={createScene} label={`${points.length} recent prediction points; green is legitimate, red is fraud`} className="prediction-scatter" fallback={fallback} /><ul className="sr-only">{points.map((point) => <li key={point.transaction_id}>{point.type}, {(point.fraud_probability * 100).toFixed(1)} percent risk, {point.is_fraud ? 'fraud' : 'legitimate'}</li>)}</ul></div>
}
