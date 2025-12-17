import { useState, useRef } from 'react'
import './App.css'

const apiBase = '' // via Vite proxy, '' maps to http://127.0.0.1:8000 during dev

function App() {
  const [employeeId, setEmployeeId] = useState('')
  const [enrollResult, setEnrollResult] = useState(null)
  const [verifyResult, setVerifyResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const enrollInputRef = useRef(null)
  const verifyInputRef = useRef(null)

  const enroll = async () => {
    if (!employeeId) {
      alert('Enter an employee ID first')
      return
    }
    const file = enrollInputRef.current?.files?.[0]
    if (!file) {
      alert('Choose an image to enroll')
      return
    }
    setBusy(true)
    setEnrollResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${apiBase}/face/enroll/${encodeURIComponent(employeeId)}`, {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      setEnrollResult({ ok: res.ok, data })
    } catch (e) {
      setEnrollResult({ ok: false, error: String(e) })
    } finally {
      setBusy(false)
    }
  }

  const verify = async () => {
    const file = verifyInputRef.current?.files?.[0]
    if (!file) {
      alert('Choose an image to verify')
      return
    }
    setBusy(true)
    setVerifyResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${apiBase}/face/verify`, {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      setVerifyResult({ ok: res.ok, data })
    } catch (e) {
      setVerifyResult({ ok: false, error: String(e) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ maxWidth: 720, margin: '0 auto', padding: 24 }}>
      <h2>Face Service Tester</h2>

      <section style={{ marginBottom: 24, padding: 16, border: '1px solid #ddd', borderRadius: 8 }}>
        <h3>Enroll</h3>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <label>
            Employee ID:
            <input
              type="number"
              value={employeeId}
              onChange={(e) => setEmployeeId(e.target.value)}
              style={{ marginLeft: 8 }}
            />
          </label>
          <input type="file" accept="image/*" ref={enrollInputRef} />
          <button disabled={busy} onClick={enroll}>Enroll</button>
        </div>
        {enrollResult && (
          <pre style={{ background: '#f7f7f7', padding: 12, marginTop: 12, overflowX: 'auto' }}>
            {JSON.stringify(enrollResult, null, 2)}
          </pre>
        )}
      </section>

      <section style={{ marginBottom: 24, padding: 16, border: '1px solid #ddd', borderRadius: 8 }}>
        <h3>Verify</h3>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <input type="file" accept="image/*" ref={verifyInputRef} />
          <button disabled={busy} onClick={verify}>Verify</button>
        </div>
        {verifyResult && (
          <pre style={{ background: '#f7f7f7', padding: 12, marginTop: 12, overflowX: 'auto' }}>
            {JSON.stringify(verifyResult, null, 2)}
          </pre>
        )}
      </section>

      <p style={{ color: '#666' }}>
        Dev note: Vite proxy sends /face/* to FastAPI at 127.0.0.1:8000.
      </p>
    </div>
  )
}

export default App
