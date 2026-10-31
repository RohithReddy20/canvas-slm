// System-design components: an icon library like Excalidraw's, as one tldraw shape type.
import {
	BaseBoxShapeUtil,
	DefaultColorStyle,
	HTMLContainer,
	T,
	TLDefaultColorStyle,
	TLShape,
} from 'tldraw'

// Each icon is SVG markup on a 24x24 grid, drawn with the shape's color.
const ICONS = {
	user: '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7"/>',
	browser: '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="M2 9h20M5.5 6.5h.01M8 6.5h.01"/>',
	mobile: '<rect x="6" y="2" width="12" height="20" rx="2.5"/><path d="M11 18.5h2"/>',
	server: '<rect x="3" y="3" width="18" height="7" rx="1.5"/><rect x="3" y="14" width="18" height="7" rx="1.5"/><path d="M7 6.5h.01M7 17.5h.01M11 6.5h6M11 17.5h6"/>',
	service: '<path d="M12 2.5l8.2 4.75v9.5L12 21.5l-8.2-4.75v-9.5z"/><path d="M12 8v8M8.5 10l7 4M15.5 10l-7 4"/>',
	database: '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v13c0 1.7 3.6 3 8 3s8-1.3 8-3v-13"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
	cache: '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M13 6l-5 7h4l-1 5 5-7h-4z"/>',
	queue: '<rect x="2" y="8" width="5" height="8" rx="1"/><rect x="9.5" y="8" width="5" height="8" rx="1"/><rect x="17" y="8" width="5" height="8" rx="1"/><path d="M2 20h20M18 18l2 2-2 2"/>',
	stream: '<path d="M2 7c3-3 5 3 8 0s5 3 8 0 3-2 4-1"/><path d="M2 12c3-3 5 3 8 0s5 3 8 0 3-2 4-1"/><path d="M2 17c3-3 5 3 8 0s5 3 8 0 3-2 4-1"/>',
	load_balancer: '<circle cx="12" cy="5" r="2.5"/><circle cx="5" cy="19" r="2.5"/><circle cx="12" cy="19" r="2.5"/><circle cx="19" cy="19" r="2.5"/><path d="M12 7.5v9M11 8l-5 8.5M13 8l5 8.5"/>',
	api_gateway: '<path d="M4 21V9a8 8 0 0116 0v12"/><path d="M2 21h20M9 21v-6h6v6M8 11h8"/>',
	cdn: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18"/>',
	dns: '<path d="M12 2v20M12 4h7l2 2.5L19 9h-7M12 12H5l-2 2.5L5 17h7"/>',
	object_storage: '<path d="M4 6h16l-2 15H6z"/><ellipse cx="12" cy="6" rx="8" ry="2.5"/>',
	function: '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M8 18l4.5-9M10 6c2-1 3 0 3.5 2l3.5 10"/>',
	container: '<path d="M12 2l9 5v10l-9 5-9-5V7z"/><path d="M3 7l9 5 9-5M12 12v10"/>',
	auth: '<rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 018 0v3M12 14.5v3"/>',
	monitoring: '<rect x="2" y="3" width="20" height="15" rx="2"/><path d="M5 14l4-4 3 3 5-6M8 21h8"/>',
	search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/>',
	firewall: '<rect x="2" y="4" width="20" height="16" rx="1"/><path d="M2 9.3h20M2 14.6h20M8 4v5.3M16 4v5.3M5 9.3v5.3M12 9.3v5.3M19 9.3v5.3M8 14.6V20M16 14.6V20"/>',
	cloud: '<path d="M7 19a5 5 0 01-.6-9.96A6.5 6.5 0 0119 9a4.5 4.5 0 01-.5 10z"/>',
	external_api: '<path d="M9 2v5M15 2v5M6 7h12v4a6 6 0 01-12 0zM12 17v5"/>',
	worker: '<circle cx="12" cy="12" r="3.5"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9L7 7M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1"/>',
	scheduler: '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l3 2M9 2h6"/>',
	data_warehouse: '<ellipse cx="8" cy="5" rx="5.5" ry="2.2"/><path d="M2.5 5v12c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2V5"/><ellipse cx="16" cy="9" rx="5.5" ry="2.2"/><path d="M10.5 9v10c0 1.2 2.5 2.2 5.5 2.2s5.5-1 5.5-2.2V9"/>',
	notification: '<path d="M6 16V11a6 6 0 0112 0v5l2 2H4zM10 20a2 2 0 004 0"/>',
	payment: '<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 10h20M6 15h4"/>',
	email: '<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 7l10 7 10-7"/>',
	ml_model: '<circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="19" cy="18" r="2"/><path d="M7 6.8l3.3 4M7 17.2l3.3-4M14 11.2L17 7M14 12.8l3 4.4"/>',
	document: '<path d="M6 2h8l5 5v15H6z"/><path d="M14 2v5h5M9 13h7M9 17h7"/>',
} as const

export type ComponentKind = keyof typeof ICONS
export const COMPONENT_KINDS = Object.keys(ICONS) as [ComponentKind, ...ComponentKind[]]

declare module 'tldraw' {
	export interface TLGlobalShapePropsMap {
		'voice-component': { w: number; h: number; component: string; label: string; color: TLDefaultColorStyle }
	}
}

export type ComponentShape = TLShape<'voice-component'>

export class ComponentShapeUtil extends BaseBoxShapeUtil<ComponentShape> {
	static override type = 'voice-component' as const
	static override props = {
		w: T.positiveNumber,
		h: T.positiveNumber,
		component: T.string,
		label: T.string,
		color: DefaultColorStyle,
	}

	override getDefaultProps() {
		return { w: 120, h: 110, component: 'service', label: '', color: 'black' as const }
	}

	override canEdit() {
		return false
	}

	override getText(shape: ComponentShape) {
		return shape.props.label
	}

	override component(shape: ComponentShape) {
		const color = this.editor.getCurrentTheme().colors[this.editor.getColorMode()][shape.props.color]
		const icon = ICONS[shape.props.component as ComponentKind] ?? ICONS.service
		const size = Math.max(24, Math.min(shape.props.w, shape.props.h - 28) - 16)
		return (
			<HTMLContainer
				style={{
					display: 'flex',
					flexDirection: 'column',
					alignItems: 'center',
					justifyContent: 'center',
					gap: 4,
					color: color.solid,
					pointerEvents: 'all',
				}}
			>
				<svg
					width={size}
					height={size}
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					strokeWidth={1.6}
					strokeLinecap="round"
					strokeLinejoin="round"
					dangerouslySetInnerHTML={{ __html: icon }}
				/>
				<div style={{ font: '13px/1.2 Menlo, monospace', textAlign: 'center', maxWidth: shape.props.w + 40 }}>
					{shape.props.label}
				</div>
			</HTMLContainer>
		)
	}

	override getIndicatorPath(shape: ComponentShape) {
		const path = new Path2D()
		path.rect(0, 0, shape.props.w, shape.props.h)
		return path
	}
}
