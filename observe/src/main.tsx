import { createRoot } from 'react-dom/client'
import { Editor, Tldraw } from 'tldraw'
import 'tldraw/tldraw.css'
import { connectBridge } from './bridge'
import { ComponentShapeUtil } from './components'
import './schema.css'
import { SchemaBoxShapeUtil } from './SchemaBoxShapeUtil'

const shapeUtils = [SchemaBoxShapeUtil, ComponentShapeUtil]

function App() {
	return (
		<div style={{ position: 'fixed', inset: 0 }}>
			<Tldraw
				shapeUtils={shapeUtils}
				onMount={(editor: Editor) => {
					;(window as unknown as { editor: Editor }).editor = editor
					connectBridge(editor)
				}}
			/>
		</div>
	)
}

createRoot(document.getElementById('root')!).render(<App />)
