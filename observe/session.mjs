// One observation session: an agy model builds toward a goal on the live tldraw page while
// playing the person who talks to the voice assistant. Needs server.mjs, the page open, and
// the bridge registered in agy (agy mcp add canvas http://localhost:8791/mcp).
//
//   node session.mjs --goal "a URL shortener" [--persona "..."] [--model gemini-3.8-flash-medium] [--steps 25]
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { parseArgs } from 'node:util'

const { values } = parseArgs({
	options: {
		goal: { type: 'string' },
		persona: { type: 'string', default: 'a software engineer sketching a design while thinking out loud' },
		model: { type: 'string', default: 'gemini-3.8-flash-medium' },
		steps: { type: 'string', default: '25' },
		id: { type: 'string' },
		timeout: { type: 'string', default: '1200' },
		from: { type: 'string' },
	},
})
if (!values.goal) throw new Error('--goal is required')

const bridge = 'http://localhost:8791'
const post = async (route, payload) => (await fetch(bridge + route, { method: 'POST', body: JSON.stringify(payload ?? {}) })).json()

const EDITING = `The diagram already exists: the person made it earlier and is back to change it. Look first and work with what is there. This person is a real, slightly messy user, not a careful one:
- they often name things loosely or a bit wrong ("the redis thing", "notif service", "the payments one" when it is called Payment Service), abbreviate, or use speech-to-text slips;
- they click things and then say "this" or "these": call user_select first (it is the person's mouse, not speech), then act with a sentence like "make these green";
- they say "it" or "that" for whatever they just made or touched;
- they change their mind ("no wait, the other one", "undo that"), and ask vaguely ("tidy this up", "make it look less messy", "lay it out left to right");
- now and then they mention something that is not on the canvas, or a name that fits two objects; the assistant then answers with no_action (missing_target or ambiguous_target) and the person clarifies next;
- they also extend the design, regroup things into frames, rename, recolor, reconnect, delete, and re-layout.
`

const prompt = `You are running a realistic session on a live tldraw canvas through the "canvas" tools (look, act, wish). Use only those tools: no files, no shell, no other tools.

Role-play ${values.persona}. They are ${values.from ? 'editing' : 'building'} ${values.goal} on the canvas by talking to a voice assistant, and you also operate the assistant.
${values.from ? EDITING : ''}

For every step:
1. Decide the next thing this person would say out loud. Speak like a person, not a programmer: natural spoken English. About half the sentences ask for one thing only ("add a cache", "make it red", "undo that"); others combine two or three edits. Sometimes vague ("put a cache in front of that"), sometimes several edits in one breath, sometimes correcting earlier work ("actually move the queue under the workers"), referring to things the way people do ("it", "that one", "the database", "the two services on the left").
2. Call act with command set to exactly that sentence and actions set to the canvas actions that carry it out, using the short IDs from the latest canvas.
3. Read the result. If an action failed, the person reacts the way a real user would (rephrases, tries something else) in the next step.

${values.from ? `Make about ${values.steps} changes like this.` : `Build the design up over about ${values.steps} steps: start empty, add the main pieces, wire them with labelled arrows, then refine (rename, regroup with frames, recolor, move things into a clean left-to-right layout, add notes, remove what no longer fits).`} Prefer create_component for system pieces (user, server, service, database, cache, queue, load_balancer, api_gateway, cdn, object_storage, worker...). Give positions with x and y so the layout stays readable, about 220 apart.

Never call act to test or explore the tools: every act is one real sentence from the person. The actions (exact names and arguments; IDs come from the canvas; "?" is optional):
- create_component(component, label?, x?, y?), component one of: user, browser, mobile, server, service, database, cache, queue, stream, load_balancer, api_gateway, cdn, dns, object_storage, function, container, auth, monitoring, search, firewall, cloud, external_api, worker, scheduler, data_warehouse, notification, payment, email, ml_model, document
- create_shape(kind, text?, x?, y?, width?, height?), kind one of: rectangle, ellipse, diamond, triangle, text, note, frame, arrow
- create_schema_box(name, fields?, methods?); add_property / remove_property(schema_id, property_name); add_method / remove_method(schema_id, method_name); rename_schema(schema_id, new_name)
- connect_schemas(source_id, target_id, label?): an arrow between any two objects (boxes, shapes, components)
- set_text(shape_id, text): also component labels, arrow labels and frame names
- move_shapes(shape_ids, dx, dy); resize_shape(shape_id, width, height); delete_shapes(shape_ids)
- style_shapes(shape_ids, color?, fill?, opacity?), color one of black, grey, light-violet, violet, blue, light-blue, yellow, orange, green, light-green, light-red, red, white; fill one of none, semi, solid, pattern
- arrange_shapes(shape_ids, operation), operation one of duplicate, group, ungroup, front, back, forward, backward, align_left, align_right, align_top, align_bottom, align_center_horizontal, align_center_vertical, distribute_horizontal, distribute_vertical, flip_horizontal, flip_vertical, stack_horizontal, stack_vertical, pack
- frame_shapes(shape_ids, name?): a new frame around shapes, which move inside it; add_to_frame(shape_ids, frame_id): into an existing frame
- auto_layout(direction, shape_ids?): lay the diagram (or those shapes) out along its arrows, direction left_to_right or top_to_bottom
- select_shapes(shape_ids); canvas_command(operation) with undo, redo, select_all, clear_selection, zoom_in, zoom_out, zoom_to_fit, reset_zoom; pan_canvas(dx, dy)

Examples of act calls:
{"command": "start with a user and a load balancer", "actions": [{"name": "create_component", "arguments": {"component": "user", "label": "User", "x": 0, "y": 200}}, {"name": "create_component", "arguments": {"component": "load_balancer", "label": "Load balancer", "x": 220, "y": 200}}]}
{"command": "connect the user to it", "actions": [{"name": "connect_schemas", "arguments": {"source_id": "shape1", "target_id": "shape2", "label": "HTTPS"}}]}
{"command": "make the database blue", "actions": [{"name": "style_shapes", "arguments": {"shape_ids": ["shape5"], "color": "blue"}}]}

When the person asks for something no action can do (for example auto-layout, exporting, explaining the design), call wish with their sentence and what is missing, then carry on.

Start with look. Finish when the design is complete; your final message is one line saying what was built.`

for (let wait = 0; wait < 30 && !(await (await fetch(bridge + '/health')).json()).page; wait++)
	await new Promise((resolve) => setTimeout(resolve, 1000))
const started = await post('/session/start', { from: values.from, id: values.id, goal: values.goal, persona: values.persona, model: values.model, steps: Number(values.steps), limit: Math.round(Number(values.steps) * 1.6), prompt })
if (!started.id) {
	console.log(JSON.stringify({ error: `could not start: ${started.error}` }))
	process.exit(1)
}
console.log('session', started.id, started.dir)

const empty = fs.mkdtempSync(path.join(os.tmpdir(), 'observe-'))
const run = spawnSync(
	'agy',
	['--model', values.model, '--sandbox', '--disable-slash-commands', '--output-format', 'json',
		// Only the canvas tools are allowed, for this session only (never saved).
		'--settings', JSON.stringify({ permissions: { allow: ['mcp(canvas/*)'] } }), '--print-timeout', `${values.timeout}s`, `-p=${prompt}`],
	{ cwd: empty, encoding: 'utf8', timeout: (Number(values.timeout) + 120) * 1000, maxBuffer: 64 * 1024 * 1024 }
)
fs.rmSync(empty, { recursive: true, force: true })

const ended = await post('/session/end')
const out = run.stdout ?? ''
let reply = null
try {
	reply = JSON.parse(out.slice(out.indexOf('{')))
} catch {
	reply = { raw: out.slice(-2000), stderr: (run.stderr ?? '').slice(-2000) }
}
fs.writeFileSync(path.join(started.dir, 'agent.json'), JSON.stringify({ exit: run.status, reply }, null, 2))
console.log(JSON.stringify({ steps: ended.steps, status: reply?.status, response: reply?.response?.slice?.(0, 300), usage: reply?.usage, error: reply?.error }))
