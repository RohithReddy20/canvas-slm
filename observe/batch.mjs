// Runs the sessions in goals.json one after another (the page holds one canvas at a time).
//   node batch.mjs [--model gemini-3.8-flash-medium] [--steps 22] [--only id,id]
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { parseArgs } from 'node:util'

const { values } = parseArgs({
	options: {
		model: { type: 'string', default: 'gemini-3.8-flash-medium' },
		steps: { type: 'string', default: '22' },
		only: { type: 'string' },
		edit: { type: 'boolean', default: false }, // start each from the finished first-batch canvas
	},
})
const goals = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, 'goals.json'), 'utf8'))
const chosen = values.only ? goals.filter((g) => values.only.split(',').includes(g.id)) : goals
for (const goal of chosen) {
	const model = values.model.replace(/[^a-z0-9.]+/gi, '-')
	const id = `${goal.id}-${values.edit ? 'edit-' : ''}${model}`
	const from = values.edit ? ['--from', `${goal.id}-gemini-3.8-flash-medium`] : []
	if (fs.existsSync(path.join(import.meta.dirname, '..', 'runs', 'observe', id, 'snapshot.json'))) {
		console.log(`${id}: done already`)
		continue
	}
	let run
	for (let attempt = 0; attempt < 2; attempt++) {
		run = spawnSync(
			'node',
			['session.mjs', '--goal', goal.goal, '--persona', goal.persona, '--model', values.model, '--steps', values.steps, '--id', id, ...from],
			{ cwd: import.meta.dirname, encoding: 'utf8' }
		)
		if (run.status === 0 && /"steps":\d+/.test(run.stdout ?? '')) break
		fs.rmSync(path.join(import.meta.dirname, '..', 'runs', 'observe', id), { recursive: true, force: true })
	}
	const last = (run.stdout ?? '').trim().split('\n').pop()
	console.log(`${id}: ${last?.slice(0, 240) ?? run.stderr?.slice(-240)}`)
}
