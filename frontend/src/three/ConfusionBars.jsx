import React, { useMemo } from 'react'
import ThreeCanvas from './ThreeCanvas.jsx'
import CountUp from '../components/CountUp.jsx'

export default function ConfusionBars({ values = [] }) {
  const signature = values.map(([label, value]) => `${label}:${value}`).join('|')
  const createScene = useMemo(() => ({ THREE, scene, camera, host }) => {
    camera.position.set(0, 0.6, 8.5)
    camera.lookAt(0, 0, 0)
    scene.add(new THREE.AmbientLight(0xffffff, 2))
    const key = new THREE.DirectionalLight(0x22d3ee, 2.5)
    key.position.set(-3, 5, 6)
    scene.add(key)
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(9, 4), new THREE.MeshBasicMaterial({ color: 0x172033, transparent: true, opacity: 0.5 }))
    floor.rotation.x = -Math.PI / 2
    floor.position.y = -1.65
    scene.add(floor)
    const entries = values.map(([label, value], index) => {
      const isError = label === 'FP' || label === 'FN'
      const material = new THREE.MeshStandardMaterial({ color: isError ? 0xf43f5e : 0x34d399, emissive: isError ? 0x7f1d31 : 0x064e3b, emissiveIntensity: 0.45, roughness: 0.35 })
      const targetHeight = Math.max(0.16, Math.log1p(Number(value) || 0) / 2.35)
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1, 0.8), material)
      mesh.position.set((index - 1.5) * 1.55, -1.65, 0)
      mesh.scale.y = 0.01
      mesh.userData = { label, value: Number(value) || 0, targetHeight, index }
      scene.add(mesh)
      return mesh
    })
    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    let hovered = null
    const onMove = (event) => {
      const rect = host.getBoundingClientRect()
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1)
      raycaster.setFromCamera(pointer, camera)
      hovered = raycaster.intersectObjects(entries, false)[0]?.object || null
      host.dataset.tooltip = hovered ? `${hovered.userData.label}: ${hovered.userData.value.toLocaleString()}` : ''
    }
    const onLeave = () => { hovered = null; host.dataset.tooltip = '' }
    host.addEventListener('pointermove', onMove)
    host.addEventListener('pointerleave', onLeave)
    const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches || document.documentElement.dataset.reduceMotion === 'true'
    return {
      update: (time) => entries.forEach((mesh, index) => {
        const { targetHeight } = mesh.userData
        const progress = reducedMotion() ? 1 : Math.max(0, Math.min(1, (time - index * 0.18) / 0.85))
        const eased = 1 - (1 - progress) ** 3
        mesh.scale.y = targetHeight * eased
        mesh.position.y = -1.65 + mesh.scale.y / 2
        mesh.position.z = mesh === hovered ? 0.18 : 0
        mesh.scale.x = mesh.scale.z = mesh === hovered ? 1.12 : 1
      }),
      dispose: () => { host.removeEventListener('pointermove', onMove); host.removeEventListener('pointerleave', onLeave); delete host.dataset.tooltip },
    }
  }, [signature])
  const fallback = <div className="matrix-fallback">{values.map(([label, value]) => <div key={label}><b>{label}</b><strong>{Number(value || 0).toLocaleString()}</strong></div>)}</div>
  return <div className="confusion-wrap"><ThreeCanvas key={signature} createScene={createScene} label="Three-dimensional confusion matrix. True negatives and positives are green; false positives and negatives are red." className="confusion-canvas" fallback={fallback} /><div className="matrix-axis-labels" aria-hidden="true">{values.map(([label, value], index) => <div key={label}><b>{label}</b><span><CountUp value={Number(value || 0).toLocaleString()} delay={index * 180} /></span></div>)}</div><ul className="sr-only">{values.map(([label, value]) => <li key={label}>{label}: {Number(value || 0).toLocaleString()}</li>)}</ul></div>
}
