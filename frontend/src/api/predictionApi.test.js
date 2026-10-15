import test from 'node:test'
import assert from 'node:assert/strict'
import { predictTransaction } from './predictionApi.js'

test('prediction client posts the transaction and returns the API decision', async () => {
  const payload = { step: 1, type: 'PAYMENT', amount: 12 }
  let request
  const result = await predictTransaction(payload, async (url, options) => {
    request = { url, options }
    return { ok: true, json: async () => ({ fraud_probability: 0.8, decision_threshold: 0.2, is_fraud: true, prediction: 'Fraud Detected' }) }
  })
  assert.match(request.url, /\/api\/v1\/predict$/)
  assert.equal(request.options.method, 'POST')
  assert.deepEqual(JSON.parse(request.options.body), payload)
  assert.equal(result.is_fraud, true)
})

test('prediction client reports API validation errors', async () => {
  await assert.rejects(
    predictTransaction({}, async () => ({ ok: false, status: 422, json: async () => ({ detail: [{ msg: 'Input should be valid' }] }) })),
    /Input should be valid/,
  )
})
