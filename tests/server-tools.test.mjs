import assert from 'node:assert/strict'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js'
import { createServer } from '../lib/server.js'
import test from 'node:test'

test('registers all deployment tools with explicit schemas and behavior hints', async t => {
  const server = createServer()
  const client = new Client({ name: 'dsh-model-deploy-tests', version: '0.0.0' })
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair()
  t.after(async () => {
    await client.close().catch(() => {})
    await server.close().catch(() => {})
  })
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)])

  const { tools } = await client.listTools()
  const byName = new Map(tools.map(tool => [tool.name, tool]))

  const expected = {
    model_inspect: { readOnly: true, idempotent: true, openWorld: false },
    deployment_environment: { readOnly: true, idempotent: true, openWorld: false },
    benchmark_local: { readOnly: true, idempotent: true, openWorld: false },
    ssh_preflight: { readOnly: true, idempotent: true, openWorld: true },
    benchmark_remote_ssh: { readOnly: false, idempotent: false, openWorld: true },
  }

  assert.deepEqual([...byName.keys()].sort(), Object.keys(expected).sort())
  for (const [name, hints] of Object.entries(expected)) {
    const tool = byName.get(name)
    assert.ok(tool, `missing tool ${name}`)
    assert.equal(typeof tool.inputSchema, 'object', `${name} must declare an input schema`)
    assert.deepEqual(tool.annotations, {
      readOnlyHint: hints.readOnly,
      destructiveHint: false,
      idempotentHint: hints.idempotent,
      openWorldHint: hints.openWorld,
    }, `${name} behavior hints must match its handler`)
  }
})
