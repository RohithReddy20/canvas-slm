// Observation bridge: exposes a live tldraw page (src/bridge.ts) to agents as MCP tools and
// logs every step with the canvas before and after.
//
//   node server.mjs [log-root]      WebSocket for the page on :8790, HTTP on :8791
//   POST /mcp                       MCP (streamable HTTP, stateless) with tools look, act, wish
//   POST /session/start {goal,...}  clears the canvas and starts a log folder
//   POST /session/end               saves the tldraw snapshot and closes the log
//   GET  /health                    whether the page is connected
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js'
import { WebSocketServer } from 'ws'
import { z } from 'zod'

const ROOT = path.resolve(process.argv[2] ?? path.join(import.meta.dirname, '..', 'runs', 'observe'))
const COMPONENTS = [...fs
	.readFileSync(path.join(import.meta.dirname, 'src', 'components.tsx'), 'utf8')
	.split('const ICONS = {')[1]
	.split('} as const')[0]
	.matchAll(/^\t(\w+):/gm)].map((m) => m[1])

// --- the page -------------------------------------------------------------------------
let page = null
const pending = new Map()
let nextId = 1
new WebSocketServer({ port: 8790 }).on('connection', (socket) => {
	page = socket
	socket.on('message', (data) => {
		const message = JSON.parse(data)
		pending.get(message.id)?.(message)
		pending.delete(message.id)
	})
	socket.on('close', () => page === socket && (page = null))
})

function call(message, timeout = 20000) {
	return new Promise((resolve, reject) => {
		if (!page) return reject(new Error('The tldraw page is not open.'))
		const id = nextId++
		const timer = setTimeout(() => {
			pending.delete(id)
			reject(new Error('The tldraw page did not answer.'))
		}, timeout)
		pending.set(id, (reply) => {
			clearTimeout(timer)
			resolve(reply)
		})
		page.send(JSON.stringify({ ...message, id }))
	})
}

// --- sessions, short IDs and logs -----------------------------------------------------
let session = null

function newSession(meta) {
	const id = meta.id ?? new Date().toISOString().replace(/[:.]/g, '-')
	const dir = path.join(ROOT, id)
	fs.mkdirSync(dir, { recursive: true })
	fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify({ ...meta, id, started: new Date().toISOString() }, null, 2))
	return { id, dir, step: 0, limit: meta.limit ?? null, short: new Map(), real: new Map(), counts: { box: 0, shape: 0, arrow: 0 } }
}

function log(event) {
	if (!session) return
	fs.appendFileSync(path.join(session.dir, 'events.jsonl'), JSON.stringify({ t: Date.now(), ...event }) + '\n')
}

function shortId(realId, kind) {
	if (!realId) return null
	if (!session) session = newSession({ goal: null, note: 'implicit' })
	if (!session.short.has(realId)) {
		const alias = `${kind}${++session.counts[kind]}`
		session.short.set(realId, alias)
		session.real.set(alias, realId)
	}
	return session.short.get(realId)
}

const round = (n) => Math.round(n)

/** The canvas as agents see it: short IDs, rounded geometry, arrows as links. */
function compact(state) {
	const { context, links, history } = state
	const kinds = new Map()
	context.schemas.forEach((b) => kinds.set(b.id, 'box'))
	links.forEach((l) => kinds.set(l.id, 'arrow'))
	const id = (realId) => (realId ? shortId(realId, kinds.get(realId) ?? 'shape') : null)
	const style = (o) => ({ ...(o.color !== 'black' ? { color: o.color } : {}), ...(o.fill !== 'none' ? { fill: o.fill } : {}) })
	const geometry = (o) => ({ x: round(o.x), y: round(o.y), w: round(o.w), h: round(o.h) })
	return {
		boxes: context.schemas.map((b) => ({ id: id(b.id), name: b.name, fields: b.properties, methods: b.methods, ...geometry(b), ...style(b) })),
		shapes: context.shapes
			.filter((s) => s.kind !== 'arrow')
			.map((s) => ({ id: id(s.id), kind: s.kind, text: s.text, ...geometry(s), ...style(s), ...(s.parent_id ? { parent: id(s.parent_id) } : {}) })),
		links: links.map((l) => ({ id: id(l.id), from: id(l.from), to: id(l.to), label: l.label })),
		selected: context.selected_ids.map(id),
		last_created: id(history.last_created_id),
		last_edited: id(history.last_edited_id),
		camera: { x: round(context.camera.x), y: round(context.camera.y), z: context.camera.z },
		can_undo: context.can_undo,
		can_redo: context.can_redo,
	}
}

// Each action's arguments: required, then optional (the page's zod schema is the authority).
const SIGNATURES = {
	create_component: [['component'], ['label', 'x', 'y']],
	create_shape: [['kind'], ['text', 'x', 'y', 'width', 'height']],
	create_schema_box: [['name'], ['fields', 'methods']],
	add_property: [['schema_id', 'property_name'], []],
	remove_property: [['schema_id', 'property_name'], []],
	add_method: [['schema_id', 'method_name'], []],
	remove_method: [['schema_id', 'method_name'], []],
	rename_schema: [['schema_id', 'new_name'], []],
	connect_schemas: [['source_id', 'target_id'], ['label']],
	set_text: [['shape_id', 'text'], []],
	move_shapes: [['shape_ids', 'dx', 'dy'], []],
	resize_shape: [['shape_id', 'width', 'height'], []],
	delete_shapes: [['shape_ids'], []],
	style_shapes: [['shape_ids'], ['color', 'fill', 'opacity']],
	arrange_shapes: [['shape_ids', 'operation'], []],
	select_shapes: [['shape_ids'], []],
	canvas_command: [['operation'], []],
	pan_canvas: [['dx', 'dy'], []],
	no_action: [['reason'], []],
	frame_shapes: [['shape_ids'], ['name']],
	add_to_frame: [['shape_ids', 'frame_id'], []],
	auto_layout: [['direction'], ['shape_ids']],
}
const signature = (name) => `${name}(${SIGNATURES[name][0].join(', ')}${SIGNATURES[name][1].map((a) => `, ${a}?`).join('')})`

function closest(name) {
	const words = String(name).toLowerCase().split(/[_\s]+/)
	let best = null
	let score = 0
	for (const candidate of Object.keys(SIGNATURES)) {
		const hits = words.filter((w) => w.length > 2 && candidate.includes(w.replace(/s$/, ''))).length
		if (hits > score) [best, score] = [candidate, hits]
	}
	return best
}

/** A clear problem with an action before it reaches the page, or null; fills harmless defaults. */
function precheck(action) {
	const name = action?.name
	if (!SIGNATURES[name]) {
		const guess = closest(name)
		return `Unknown action "${name}".${guess ? ` Did you mean ${signature(guess)}?` : ''} Valid actions: ${Object.keys(SIGNATURES).join(', ')}.`
	}
	const args = (action.arguments ??= {})
	const [required, optional] = SIGNATURES[name]
	if (name === 'create_schema_box') {
		args.fields ??= []
		args.methods ??= []
	}
	if (name === 'connect_schemas') args.label ??= ''
	if (typeof args.shape_ids === 'string') args.shape_ids = [args.shape_ids]
	const extra = Object.keys(args).filter((k) => !required.includes(k) && !optional.includes(k) && !['fields', 'methods'].includes(k))
	const missing = required.filter((k) => !(k in args))
	if (extra.length || missing.length)
		return `${name} takes ${signature(name)}.${missing.length ? ` Missing: ${missing.join(', ')}.` : ''}${extra.length ? ` Not allowed: ${extra.join(', ')}.` : ''}`
	return null
}

/** The page's zod error as one short line. */
function short(error) {
	try {
		const issues = JSON.parse(error)
		return issues
			.map((i) => `${(i.path ?? []).join('.') || 'action'}: ${i.message}${i.values ? ` (one of ${i.values.join(', ')})` : ''}`)
			.join('; ')
	} catch {
		return error
	}
}

const TARGETS = ['schema_id', 'shape_id', 'source_id', 'target_id', 'frame_id']
function toReal(action) {
	const args = { ...(action.arguments ?? {}) }
	const real = (value) => (typeof value === 'string' && session?.real.get(value)) || value
	for (const key of TARGETS) if (key in args) args[key] = real(args[key])
	if (Array.isArray(args.shape_ids)) args.shape_ids = args.shape_ids.map(real)
	return { name: action.name, arguments: args }
}

// --- tools ----------------------------------------------------------------------------
const ACTIONS = `Actions (IDs are the short IDs from look; "?" marks optional arguments):
- create_component(component, label, x?, y?): a system-design icon. component is one of: ${COMPONENTS.join(', ')}
- create_shape(kind, text?, x?, y?, width?, height?): kind is rectangle | ellipse | diamond | triangle | text | note | frame | arrow
- create_schema_box(name, fields, methods): a class or table box
- add_property / remove_property(schema_id, property_name); add_method / remove_method(schema_id, method_name); rename_schema(schema_id, new_name)
- connect_schemas(source_id, target_id, label): a bound arrow between any two objects
- set_text(shape_id, text): text of a shape, component label, arrow label, frame or box name
- move_shapes(shape_ids, dx, dy): right and down are positive
- resize_shape(shape_id, width, height)
- delete_shapes(shape_ids)
- style_shapes(shape_ids, color?, fill?, opacity?): color black | grey | light-violet | violet | blue | light-blue | yellow | orange | green | light-green | light-red | red | white; fill none | semi | solid | pattern
- arrange_shapes(shape_ids, operation): duplicate | group | ungroup | front | back | forward | backward | align_left | align_right | align_top | align_bottom | align_center_horizontal | align_center_vertical | distribute_horizontal | distribute_vertical | flip_horizontal | flip_vertical | stack_horizontal | stack_vertical | pack
- frame_shapes(shape_ids, name?): a new frame around the shapes, which move inside it
- add_to_frame(shape_ids, frame_id): moves shapes into an existing frame, which grows to fit
- auto_layout(direction, shape_ids?): lays the diagram (or just those shapes) out along its arrows; direction left_to_right | top_to_bottom
- select_shapes(shape_ids); canvas_command(operation): undo | redo | select_all | clear_selection | zoom_in | zoom_out | zoom_to_fit | reset_zoom; pan_canvas(dx, dy)
- no_action(reason): missing_target | ambiguous_target | unsupported_request
Each action is {"name": ..., "arguments": {...}}; arguments must be exactly those listed.`

const text = (value) => ({ content: [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value) }] })

function makeServer() {
	const server = new McpServer({ name: 'tldraw-canvas', version: '0.1.0' })
	server.registerTool(
		'look',
		{ description: 'The current tldraw canvas: boxes, shapes (with components), links (arrows), selection, last created and edited objects, camera.', inputSchema: {} },
		async () => {
			const state = await call({ type: 'look' })
			const canvas = compact(state)
			log({ type: 'look', canvas })
			return text(canvas)
		}
	)
	server.registerTool(
		'act',
		{
			description: `Carry out one thing the person says to the voice assistant. "command" is the exact sentence they say; "actions" are the canvas actions, in order, that do it. Actions run one by one and stop at the first error; the reply has each result and the canvas after.\n${ACTIONS}`,
			inputSchema: {
				command: z.string().describe('What the person says, word for word, as spoken English.'),
				actions: z
					.array(z.object({ name: z.string(), arguments: z.record(z.string(), z.any()) }))
					.describe('The canvas actions that carry out the command.'),
			},
		},
		async ({ command, actions }) => {
			const first = await call({ type: 'look' })
			const before = compact(first)
			const results = []
			let state = first
			if (!actions.length)
				return text({ results: [{ ok: false, error: 'Give at least one action. When nothing can be done, answer with no_action(reason): missing_target, ambiguous_target or unsupported_request.' }] })
			if (session && session.limit && session.step >= session.limit)
				return text('The session is over: the step limit is reached. Stop now and give your one-line summary.')
			for (const action of actions) {
				const problem = precheck(action)
				if (problem) {
					results.push({ ok: false, error: problem })
					break
				}
				const reply = await call({ type: 'act', action: toReal(action) })
				state = reply
				compact(reply) // name new objects before reporting them
				results.push(reply.ok ? { ok: true, created: session?.short.get(reply.createdId) ?? null } : { ok: false, error: short(reply.error) })
				if (!reply.ok) break
			}
			const after = compact(state)
			const step = session ? ++session.step : null
			log({ type: 'act', step, command, actions, results, before, after })
			return text({ results, canvas: after })
		}
	)
	server.registerTool(
		'user_select',
		{
			description: 'The person clicks or drag-selects shapes on the canvas themselves (not something they say). Use it before a sentence like "make this red". An empty list clears the selection.',
			inputSchema: { shape_ids: z.array(z.string()).describe('Short IDs the person selects.') },
		},
		async ({ shape_ids }) => {
			const before = compact(await call({ type: 'look' }))
			const real = shape_ids.map((id) => session?.real.get(id) ?? id)
			const after = compact(await call({ type: 'select', ids: real }))
			log({ type: 'user_select', shape_ids, before, after })
			return text({ selected: after.selected })
		}
	)
	server.registerTool(
		'wish',
		{
			description: 'Record something the person asks for that no action can do (a capability the assistant lacks). Nothing changes on the canvas.',
			inputSchema: { command: z.string().describe('What the person says.'), missing: z.string().describe('What the assistant would need to be able to do.') },
		},
		async ({ command, missing }) => {
			log({ type: 'wish', command, missing })
			return text('Recorded. Nothing changed on the canvas.')
		}
	)
	return server
}

// --- HTTP -----------------------------------------------------------------------------
const body = (req) =>
	new Promise((resolve) => {
		let data = ''
		req.on('data', (chunk) => (data += chunk))
		req.on('end', () => resolve(data ? JSON.parse(data) : {}))
	})
const json = (res, status, value) => res.writeHead(status, { 'content-type': 'application/json' }).end(JSON.stringify(value))

http
	.createServer(async (req, res) => {
		try {
			if (req.url.startsWith('/mcp')) {
				if (req.method !== 'POST') return json(res, 405, { error: 'POST only' })
				const payload = await body(req)
				const server = makeServer()
				const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined })
				res.on('close', () => {
					transport.close()
					server.close()
				})
				await server.connect(transport)
				await transport.handleRequest(req, res, payload)
				return
			}
			if (req.url === '/health') return json(res, 200, { page: Boolean(page), session: session?.id ?? null })
			if (req.url === '/session/start' && req.method === 'POST') {
				const meta = await body(req)
				await call({ type: 'reset' })
				if (meta.from) {
					const snapshot = JSON.parse(fs.readFileSync(path.join(ROOT, meta.from, 'snapshot.json'), 'utf8'))
					await call({ type: 'load', snapshot })
				}
				session = newSession(meta)
				log({ type: 'start', canvas: compact(await call({ type: 'look' })) })
				return json(res, 200, { id: session.id, dir: session.dir })
			}
			if (req.url === '/session/end' && req.method === 'POST') {
				if (!session) return json(res, 400, { error: 'no session' })
				const state = await call({ type: 'snapshot' })
				fs.writeFileSync(path.join(session.dir, 'snapshot.json'), JSON.stringify(state.snapshot))
				const picture = await call({ type: 'image' }, 60000)
				if (picture.png) fs.writeFileSync(path.join(session.dir, 'canvas.png'), Buffer.from(picture.png, 'base64'))
				log({ type: 'end', canvas: compact(state) })
				const done = { id: session.id, dir: session.dir, steps: session.step }
				session = null
				return json(res, 200, done)
			}
			json(res, 404, { error: 'not found' })
		} catch (error) {
			if (!res.headersSent) json(res, 500, { error: String(error?.message ?? error) })
		}
	})
	.listen(8791, () => console.log(`bridge on :8791 (page socket :8790), logs in ${ROOT}`))
