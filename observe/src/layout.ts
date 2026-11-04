// Frame and layout actions the observation sessions asked for: wrap shapes in a frame, move
// shapes into a frame, and lay a diagram out along its arrows.
import { createShapeId, Editor, fitFrameToContent, TLArrowBinding, TLShape, TLShapeId } from 'tldraw'

const PADDING = 48
const GAP = 80
export type Side = 'right' | 'left' | 'above' | 'below'

/**
 * A free spot for a new w x h object: beside `near` (else the selection, else right of
 * everything) on `side`, moved on along that side until it overlaps nothing.
 * slm/extensions.py place() mirrors this for the simulator.
 */
export function placeNear(editor: Editor, w: number, h: number, near: TLShape | null, side: Side = 'right') {
	const selected = editor.getSelectedShapes()[0]
	// Frames are room to place in, not obstacles: a shape placed inside one joins it.
	const tops = editor
		.getCurrentPageShapes()
		.filter((shape) => !['arrow', 'group', 'frame'].includes(shape.type))
		.map((shape) => editor.getShapePageBounds(shape)!)
	let anchor = near ? editor.getShapePageBounds(near)! : selected ? editor.getShapePageBounds(selected)! : null
	if (!anchor) {
		if (!tops.length) {
			const center = editor.getViewportPageBounds().center
			return { x: center.x - w / 2, y: center.y - h / 2 }
		}
		const right = Math.max(...tops.map((b) => b.maxX))
		anchor = { x: right - w, y: Math.min(...tops.map((b) => b.minY)), w, h } as never
		side = 'right'
	}
	const a = anchor as { x: number; y: number; w: number; h: number }
	let x = a.x + (side === 'right' ? a.w + GAP : side === 'left' ? -(w + GAP) : (a.w - w) / 2)
	let y = a.y + (side === 'below' ? a.h + GAP : side === 'above' ? -(h + GAP) : (a.h - h) / 2)
	const [sx, sy] = side === 'right' ? [w + GAP, 0] : side === 'left' ? [-(w + GAP), 0] : side === 'below' ? [0, h + GAP] : [0, -(h + GAP)]
	for (let i = 0; i < 40; i++) {
		if (!tops.some((b) => x < b.maxX && b.minX < x + w && y < b.maxY && b.minY < y + h)) break
		x += sx
		y += sy
	}
	return { x: Math.round(x), y: Math.round(y) }
}

function topLevel(editor: Editor, shape: TLShape): TLShape {
	let current = shape
	while (current.parentId !== editor.getCurrentPageId()) {
		const parent = editor.getShape(current.parentId as TLShapeId)
		if (!parent) break
		current = parent
	}
	return current
}

/** A new frame around the shapes, which become its children; returns the frame's ID. */
export function frameShapes(editor: Editor, shapes: TLShape[], name: string) {
	const bounds = shapes.map((shape) => editor.getShapePageBounds(shape)!)
	const minX = Math.min(...bounds.map((b) => b.minX))
	const minY = Math.min(...bounds.map((b) => b.minY))
	const maxX = Math.max(...bounds.map((b) => b.maxX))
	const maxY = Math.max(...bounds.map((b) => b.maxY))
	const id = createShapeId()
	editor.createShape({
		id,
		type: 'frame',
		x: minX - PADDING,
		y: minY - PADDING,
		props: { w: maxX - minX + PADDING * 2, h: maxY - minY + PADDING * 2, name },
	})
	editor.sendToBack([id])
	editor.reparentShapes(shapes.map((shape) => shape.id), id)
	return id
}

/** Moves shapes into an existing frame and grows the frame to fit them. */
export function addToFrame(editor: Editor, shapes: TLShape[], frame: TLShape) {
	if (frame.type !== 'frame') throw new Error('The target is not a frame.')
	if (shapes.some((shape) => shape.id === frame.id)) throw new Error('A frame cannot go inside itself.')
	editor.reparentShapes(shapes.map((shape) => shape.id), frame.id)
	fitFrameToContent(editor, frame.id, { padding: PADDING })
}

/**
 * A layered layout along the arrows: each object goes one layer after the objects pointing
 * at it, layers run left to right (or top to bottom), and objects without arrows line up
 * after the graph. Objects inside frames move with their frame.
 */
export function autoLayout(editor: Editor, only: TLShape[] | null, direction: 'left_to_right' | 'top_to_bottom') {
	const page = editor.getCurrentPageShapes()
	const nodes = (only ?? page)
		.map((shape) => topLevel(editor, shape))
		.filter((shape, index, all) => shape.type !== 'arrow' && all.findIndex((s) => s.id === shape.id) === index)
	if (nodes.length < 2) throw new Error('There is nothing to lay out.')
	const ids = new Set(nodes.map((node) => node.id))
	const edges: [TLShapeId, TLShapeId][] = []
	for (const arrow of page.filter((shape) => shape.type === 'arrow')) {
		const bindings = editor.getBindingsFromShape<TLArrowBinding>(arrow.id, 'arrow')
		const end = (terminal: string) => {
			const binding = bindings.find((b) => b.props.terminal === terminal)
			const shape = binding && editor.getShape(binding.toId)
			return shape ? topLevel(editor, shape).id : null
		}
		const from = end('start')
		const to = end('end')
		if (from && to && from !== to && ids.has(from) && ids.has(to)) edges.push([from, to])
	}
	const linked = new Set(edges.flat())

	// Layers by longest path, ignoring arrows that close a cycle.
	const layer = new Map<TLShapeId, number>()
	const visiting = new Set<TLShapeId>()
	const depth = (id: TLShapeId): number => {
		if (layer.has(id)) return layer.get(id)!
		visiting.add(id)
		let best = 0
		for (const [from, to] of edges) if (to === id && !visiting.has(from)) best = Math.max(best, depth(from) + 1)
		visiting.delete(id)
		layer.set(id, best)
		return best
	}
	nodes.filter((node) => linked.has(node.id)).forEach((node) => depth(node.id))
	const layers: TLShapeId[][] = []
	for (const node of nodes) {
		if (!linked.has(node.id)) continue
		;(layers[layer.get(node.id)!] ??= []).push(node.id)
	}

	// Order within each layer by the average position of the objects pointing in.
	for (let pass = 0; pass < 3; pass++) {
		for (let i = 1; i < layers.length; i++) {
			const index = new Map(layers[i - 1].map((id, k) => [id, k]))
			const score = (id: TLShapeId) => {
				const ins = edges.filter(([from, to]) => to === id && index.has(from)).map(([from]) => index.get(from)!)
				return ins.length ? ins.reduce((a, b) => a + b, 0) / ins.length : Infinity
			}
			layers[i].sort((a, b) => score(a) - score(b))
		}
	}

	const across = direction === 'left_to_right'
	const size = (id: TLShapeId) => editor.getShapePageBounds(id)!
	const start = {
		x: Math.min(...nodes.map((node) => size(node.id).minX)),
		y: Math.min(...nodes.map((node) => size(node.id).minY)),
	}
	const GAP_LAYER = 220
	const GAP_ITEM = 70
	const spans = layers.map((ids) => ids.reduce((sum, id) => sum + (across ? size(id).h : size(id).w), 0) + GAP_ITEM * (ids.length - 1))
	const widest = Math.max(0, ...spans)
	const updates: { id: TLShapeId; type: string; x: number; y: number }[] = []
	let offset = 0
	for (let i = 0; i < layers.length; i++) {
		const thickness = Math.max(...layers[i].map((id) => (across ? size(id).w : size(id).h)))
		let along = (widest - spans[i]) / 2
		for (const id of layers[i]) {
			const box = size(id)
			const shape = editor.getShape(id)!
			const x = across ? offset + (thickness - box.w) / 2 : along
			const y = across ? along : offset + (thickness - box.h) / 2
			updates.push({ id, type: shape.type, x: start.x + x + (shape.x - box.minX), y: start.y + y + (shape.y - box.minY) })
			along += (across ? box.h : box.w) + GAP_ITEM
		}
		offset += thickness + GAP_LAYER
	}
	// Objects without arrows (notes, legends) line up after the graph.
	let along = 0
	for (const node of nodes.filter((node) => !linked.has(node.id))) {
		const box = size(node.id)
		const x = across ? along : offset
		const y = across ? (layers.length ? widest + GAP_LAYER : 0) : along
		updates.push({ id: node.id, type: node.type, x: start.x + x + (node.x - box.minX), y: start.y + y + (node.y - box.minY) })
		along += (across ? box.w : box.h) + GAP_ITEM
	}
	editor.updateShapes(updates as never)
	return nodes[0].id
}
