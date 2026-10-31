// Connects the page's tldraw editor to the observation bridge (server.mjs) over a WebSocket.
import { Editor, TLArrowBinding, TLShapeId } from 'tldraw'
import { canvasContext, executeCanvasAction } from './voiceActions'

const history = { last_created_id: null as string | null, last_edited_id: null as string | null }

function links(editor: Editor) {
	return editor
		.getCurrentPageShapes()
		.filter((shape) => shape.type === 'arrow')
		.map((arrow) => {
			const bindings = editor.getBindingsFromShape<TLArrowBinding>(arrow.id, 'arrow')
			const end = (terminal: string) => bindings.find((b) => b.props.terminal === terminal)?.toId ?? null
			const label = (arrow.meta?.voiceLabel as string | undefined) ?? editor.getShapeUtil(arrow).getText(arrow) ?? ''
			return { id: arrow.id, from: end('start'), to: end('end'), label }
		})
}

function state(editor: Editor) {
	return { context: canvasContext(editor), links: links(editor), history: { ...history } }
}

async function image(editor: Editor) {
	const ids = [...editor.getCurrentPageShapeIds()]
	if (!ids.length) return { png: null }
	const { blob } = await editor.toImage(ids, { format: 'png', background: true, padding: 32, scale: 1 })
	const bytes = new Uint8Array(await blob.arrayBuffer())
	let binary = ''
	for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
	return { png: btoa(binary) }
}

function handle(editor: Editor, message: { type: string; action?: unknown; [key: string]: unknown }) {
	switch (message.type) {
		case 'look':
			return state(editor)
		case 'act': {
			const result = executeCanvasAction(editor, message.action, canvasContext(editor))
			if (result.createdId) history.last_created_id = result.createdId
			if (result.editedId) history.last_edited_id = result.editedId
			return { createdId: result.createdId, editedId: result.editedId, ...state(editor) }
		}
		case 'reset': {
			editor.run(
				() => {
					editor.deleteShapes([...editor.getCurrentPageShapeIds()] as TLShapeId[])
					editor.selectNone()
					editor.setCamera({ x: 0, y: 0, z: 1 })
				},
				{ history: 'ignore' }
			)
			editor.clearHistory()
			history.last_created_id = history.last_edited_id = null
			return state(editor)
		}
		case 'snapshot':
			return { snapshot: editor.getSnapshot(), ...state(editor) }
		case 'image':
			return image(editor)
		case 'load': {
			// A finished canvas from an earlier session, as the start of a new one.
			editor.loadSnapshot({ document: (message.snapshot as { document: never }).document })
			editor.selectNone()
			editor.clearHistory()
			editor.zoomToFit({ immediate: true })
			history.last_created_id = history.last_edited_id = null
			return state(editor)
		}
		case 'select': {
			// The person clicks or drag-selects shapes themselves; not an assistant action.
			const ids = (message.ids as string[]).filter((id) => editor.getShape(id as TLShapeId))
			editor.run(() => (ids.length ? editor.select(...(ids as TLShapeId[])) : editor.selectNone()), { history: 'ignore' })
			return state(editor)
		}
		default:
			throw new Error(`Unknown message ${message.type}`)
	}
}

export function connectBridge(editor: Editor, url = 'ws://localhost:8790') {
	const connect = () => {
		const socket = new WebSocket(url)
		socket.onmessage = async (event) => {
			const message = JSON.parse(event.data)
			let reply
			try {
				reply = { id: message.id, ok: true, ...(await handle(editor, message)) }
			} catch (error) {
				reply = { id: message.id, ok: false, error: String((error as Error)?.message ?? error), ...state(editor) }
			}
			socket.send(JSON.stringify(reply))
		}
		socket.onclose = () => setTimeout(connect, 1000)
		socket.onerror = () => socket.close()
	}
	connect()
}
