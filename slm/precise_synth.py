"""Precise author-style instructions on diagrams, each with its exact script.

The person is the author and the model is the writer: "between Client and Load Balancer
add a Rate Limiter" means one new node wired in on that path and nothing else. No teacher
is needed: diagram, instruction and script are generated together from templates, the
script is run in the simulator, and the row is kept only when it applies and changes the
diagram. Rows use the synthetic format (diagram, command, script, outcome) with
"source": "precise" and "op": the edit, so they train and score next to the teacher rows.

Diagrams are generated system designs (components wired client to storage) plus the
diagrams of earlier synthetic rows (boxes, shapes and components), so the edits cover
icons, schema boxes and plain shapes alike.

    python precise_synth.py --count 8000 --diagrams ../runs/colab/synthetic-all.jsonl.xz \\
        --output ../runs/colab/precise.jsonl
"""

import argparse
import copy
import hashlib
import json
import lzma
import random
from collections import Counter
from pathlib import Path

from contract import run_script, semantic_state
from distil_synth import session_of

# Several names per icon, so the model learns which icon a name means ("Rate Limiter" is
# the gateway icon, "Kafka" the stream icon).
NAMES = {
    "user": ["User", "Customer", "Admin", "End User"],
    "browser": ["Browser", "Web App", "React Frontend", "Web Client"],
    "mobile": ["Mobile App", "iOS App", "Android App"],
    "server": ["App Server", "Web Server", "Backend Server", "Nginx"],
    "service": ["Order Service", "User Service", "Inventory Service", "Catalog Service", "Checkout Service",
                "Rate Limiter", "Throttling Service", "Report Service"],
    "database": ["Postgres", "MySQL", "MongoDB", "Users DB", "Orders DB", "Main Database"],
    "cache": ["Redis", "Cache", "Memcached", "Redis Cache", "Session Cache"],
    "queue": ["Message Queue", "RabbitMQ", "SQS Queue", "Job Queue"],
    "stream": ["Kafka", "Event Stream", "Kinesis"],
    "load_balancer": ["Load Balancer", "LB", "HAProxy", "ALB"],
    "api_gateway": ["API Gateway", "Gateway", "Kong Gateway", "Edge Gateway"],
    "cdn": ["CDN", "CloudFront", "Edge Cache"],
    "dns": ["DNS", "Route 53"],
    "object_storage": ["S3 Bucket", "Object Storage", "Blob Storage", "File Storage"],
    "function": ["Lambda", "Cloud Function", "Resize Function", "Webhook Handler"],
    "container": ["Docker Container", "Kubernetes Pod", "App Container"],
    "auth": ["Auth Service", "Identity Provider", "OAuth Server", "Login Service"],
    "monitoring": ["Monitoring", "Prometheus", "Grafana", "Logging"],
    "search": ["Elasticsearch", "Search Index", "Search Service"],
    "firewall": ["Firewall", "WAF"],
    "cloud": ["AWS", "Cloud Provider", "GCP"],
    "external_api": ["Stripe API", "Twilio", "Maps API", "Third-party API"],
    "worker": ["Background Worker", "Image Worker", "Job Worker", "Email Worker"],
    "scheduler": ["Scheduler", "Cron Scheduler", "Job Scheduler"],
    "data_warehouse": ["Data Warehouse", "Snowflake", "BigQuery"],
    "notification": ["Notification Service", "Push Notifications", "Notifier"],
    "payment": ["Payment Gateway", "Billing Service", "Payments"],
    "email": ["Email Service", "SendGrid", "Mailer"],
    "ml_model": ["ML Model", "Recommendation Model", "Fraud Model"],
    "document": ["Invoice PDF", "Report Document", "Contract"],
}
LAYERS = [
    ["user", "browser", "mobile"],
    ["dns", "cdn", "load_balancer", "api_gateway", "firewall"],
    ["server", "service", "container", "function", "auth"],
    ["database", "cache", "queue", "stream", "object_storage", "search"],
    ["worker", "scheduler", "ml_model", "data_warehouse", "notification", "email", "payment",
     "external_api", "monitoring", "document", "cloud"],
]
EDGE_LABELS = ["", "", "", "", "calls", "reads", "writes", "https", "queues", "publishes", "gRPC", "events"]
BOX_NAMES = ["Payment", "Invoice", "Supplier", "Shipment", "Review", "Coupon", "Ticket", "Session",
             "Address", "Category", "Warehouse", "Refund", "Notification", "Report", "Audit Log"]
SHAPE_NAMES = ["Deployment", "Review Step", "QA", "Approval", "Staging", "Backup", "Archive", "Validation"]
COLORS = ["red", "blue", "green", "orange", "violet", "grey", "yellow"]
SIDES = {"right": ["to the right of", "right of"], "left": ["to the left of", "left of"],
         "below": ["below", "under", "underneath"], "above": ["above", "over"]}
JOINERS = [" and ", ", then ", ". Also ", "; ", " and also ", ". After that, ", ", and then "]
WEIGHTS = {"insert_between": 22, "add_linked": 18, "replace": 8, "redirect": 8, "delete_edge": 7,
           "relabel_edge": 6, "connect": 8, "rename": 7, "delete_node": 5, "add_beside": 5, "style": 3}


def architecture(rng):
    """A generated system design: components in layers, wired left to right."""
    taken, nodes, layers = set(), [], []
    sizes = [rng.randint(1, 2), rng.randint(0, 2), rng.randint(1, 3), rng.randint(1, 3), rng.randint(0, 3)]
    if sizes[1] == 0 and rng.random() < 0.5:
        sizes[1] = 1
    for layer, count in enumerate(sizes):
        chosen = []
        for kind in rng.sample(LAYERS[layer], min(count, len(LAYERS[layer]))):
            name = rng.choice([n for n in NAMES[kind] if n not in taken] or [None])
            if name is None:
                continue
            taken.add(name)
            chosen.append({"id": f"c{len(nodes) + 1}", "kind": f"component:{kind}", "text": name,
                           "x": 80 + 240 * layer, "y": 120 + 180 * len(chosen)})
            nodes.append(chosen[-1])
        layers.append(chosen)
    links, seen = [], set()

    def link(source, target):
        if source["id"] != target["id"] and (source["id"], target["id"]) not in seen:
            seen.add((source["id"], target["id"]))
            links.append({"from": source["id"], "to": target["id"], "label": rng.choice(EDGE_LABELS)})

    filled = [layer for layer in layers if layer]
    for earlier, later in zip(filled, filled[1:]):
        for target in later:
            link(rng.choice(earlier), target)
        for source in earlier:
            if not any(x["from"] == source["id"] for x in links):
                link(source, rng.choice(later))
    for _ in range(rng.randint(0, 2)):  # a backend node reading or writing storage
        if layers[4] and layers[3]:
            link(rng.choice(layers[4]), rng.choice(layers[3]))
    return {"boxes": [], "shapes": nodes, "connections": links}


def loaded_diagrams(path):
    opener = lzma.open if str(path).endswith(".xz") else open
    seen, diagrams = set(), []
    with opener(path, "rt") as source:
        for line in source:
            diagram = json.loads(line)["diagram"]
            key = json.dumps(diagram, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            diagram = {k: v for k, v in diagram.items() if k in ("boxes", "shapes", "connections")}
            if len(diagram.get("boxes", [])) + len(diagram.get("shapes", [])) >= 3:
                diagrams.append(diagram)
    return diagrams


class Canvas:
    """What the generator may name on a diagram: unique nodes and their arrows."""

    def __init__(self, diagram):
        self.diagram = diagram
        self.session = session_of(diagram)
        objects = self.session.objects()
        counts = Counter(o["name"].casefold() for o in objects.values())
        self.nodes = [o for o in objects.values() if o.get("kind") not in ("arrow", "frame", "group")
                      and counts[o["name"].casefold()] == 1 and o["name"].strip()]
        by_id = {o["id"]: o for o in self.nodes}
        self.edges = [(by_id[x["source_id"]], by_id[x["target_id"]], x["label"])
                      for x in self.session.connections if x["source_id"] in by_id and x["target_id"] in by_id
                      and x["source_id"] != x["target_id"]]
        pairs = Counter((a["id"], b["id"]) for a, b, _ in self.edges)
        self.edges = [e for e in self.edges if pairs[(e[0]["id"], e[1]["id"])] == 1]
        kinds = Counter("box" if "kind" not in o else "component" if str(o["kind"]).startswith("component:")
                        else "shape" for o in self.nodes)
        self.style = kinds.most_common(1)[0][0] if kinds else "component"

    def has_edge(self, a, b):
        return any(x["id"] == a["id"] and y["id"] == b["id"] for x, y, _ in self.edges)


def ref(node):
    return {"name": node["name"]}


def mention(node, rng):
    name = node["name"]
    roll = rng.random()
    if roll < 0.12:
        return f'"{name}"'
    if roll < 0.22:
        return f"the {name}"
    if roll < 0.32 and name.lower() != name:
        return name.lower()
    return name


def new_node(canvas, rng):
    """(phrase, create action) for a node the diagram lacks, in the diagram's own style."""
    taken = {o["name"].casefold() for o in canvas.nodes}
    if canvas.style == "component":
        kind = rng.choice(list(NAMES))
        names = [n for n in NAMES[kind] if n.casefold() not in taken]
        if not names:
            return None
        name = rng.choice(names)
        word = kind.replace("_", " ")
        generic = name.casefold() == word or name.casefold() == word + " service"
        phrases = [f"a {name} node", f"a {name}", name, f"a new {name}", f"a {name} component",
                   f"a {word} called {name}", f'a {word} named "{name}"', f"a {word} labelled {name}",
                   f"a new {word} node, {name}"]
        if generic:
            phrases += [f"a {word}", f"a {word} node", f"an {word}" if word[0] in "aeiou" else f"a {word}"]
        action = {"name": "create_component", "arguments": {"component": kind, "label": name}}
        return rng.choice(phrases), action
    if canvas.style == "box":
        names = [n for n in BOX_NAMES if n.casefold() not in taken]
        if not names:
            return None
        name = rng.choice(names)
        phrases = [f"a {name} table", f"a {name} box", f"a {name} entity", f"{name}", f"a new {name} class",
                   f"a box called {name}"]
        return rng.choice(phrases), {"name": "create_schema_box", "arguments": {"name": name, "fields": [], "methods": []}}
    names = [n for n in SHAPE_NAMES if n.casefold() not in taken]
    if not names:
        return None
    name = rng.choice(names)
    phrases = [f"a {name} box", f"a {name} step", f"{name}", f"a rectangle that says {name}", f'a box labelled "{name}"']
    return rng.choice(phrases), {"name": "create_shape", "arguments": {"kind": "rectangle", "text": name}}


def placed(action, near, side):
    if action["name"] == "create_component":
        action = copy.deepcopy(action)
        action["arguments"].update(near=ref(near), side=side)
    return action


def connect(source, target, label=""):
    return {"name": "connect_schemas", "arguments": {"source_id": source, "target_id": target, "label": label}}


def delete(targets):
    return {"name": "delete_shapes", "arguments": {"shape_ids": targets}}


def edge_ref(a, b):
    return {"connection": {"from": ref(a), "to": ref(b)}}


def label_phrase(rng):
    label = rng.choice([x for x in EDGE_LABELS if x])
    return rng.choice([f' labelled "{label}"', f" with the label {label}", f', label it "{label}"',
                       f" (label: {label})"]), label


def op_insert_between(canvas, rng, step):
    if not canvas.edges:
        return None
    a, b, label = rng.choice(canvas.edges)
    made = new_node(canvas, rng)
    if not made:
        return None
    phrase, create = made
    A, B = mention(a, rng), mention(b, rng)
    text = rng.choice([
        "between {A} and {B} add {N}", "insert {N} between {A} and {B}", "add {N} between {A} and {B}",
        "put {N} in between {A} and {B}", "{A} should go through {N} before reaching {B}",
        "route {A} to {B} via {N}", "add {N} on the path from {A} to {B}", "stick {N} between {A} and {B}",
        "we need {N} between {A} and {B}", "{N} goes between {A} and {B}", "{A} -> {B} should pass through {N}",
        "break the {A} to {B} link with {N} in the middle",
    ]).format(A=A, B=B, N=phrase)
    actions = [placed(create, a, "right"), delete([edge_ref(a, b)]),
               connect(ref(a), {"new": step}, label), connect({"new": step}, ref(b), "")]
    return text, actions


def op_add_linked(canvas, rng, step):
    if not canvas.nodes:
        return None
    anchor = rng.choice(canvas.nodes)
    made = new_node(canvas, rng)
    if not made:
        return None
    phrase, create = made
    X = mention(anchor, rng)
    outward = rng.random() < 0.6
    if outward:
        text = rng.choice([
            "add {N} and connect {X} to it", "add {N} that {X} calls", "attach {N} to {X}",
            "hang {N} off {X}", "add {N} downstream of {X}", "add {N} after {X}", "{X} should talk to {N}, add it",
            "add {N} and wire {X} to it", "create {N} connected from {X}", "add {N} after {X} and link them",
        ])
    else:
        text = rng.choice([
            "add {N} that feeds into {X}", "add {N} pointing to {X}", "add {N} in front of {X}",
            "add {N} upstream of {X}", "add {N} and connect it to {X}", "add {N} before {X}",
            "create {N} with an arrow to {X}", "add {N} and draw an arrow from it to {X}",
        ])
    text = text.format(N=phrase, X=X)
    label = ""
    if rng.random() < 0.3:
        suffix, label = label_phrase(rng)
        text += suffix
    link = connect(ref(anchor), {"new": step}, label) if outward else connect({"new": step}, ref(anchor), label)
    return text, [placed(create, anchor, "right" if outward else "left"), link]


def op_add_beside(canvas, rng, step):
    if not canvas.nodes:
        return None
    anchor = rng.choice(canvas.nodes)
    made = new_node(canvas, rng)
    if not made:
        return None
    phrase, create = made
    side = rng.choice(list(SIDES))
    where = rng.choice(SIDES[side]) if rng.random() < 0.85 or side != "right" else "next to"
    text = rng.choice(["add {N} {W} {X}", "put {N} {W} {X}", "place {N} {W} {X}", "{W} {X}, add {N}",
                       "add {N} {W} {X}, no arrow"]).format(N=phrase, W=where, X=mention(anchor, rng))
    return text[0].upper() + text[1:] if text.startswith(where) else text, [placed(create, anchor, side)]


def op_delete_edge(canvas, rng, step):
    if not canvas.edges:
        return None
    a, b, _ = rng.choice(canvas.edges)
    text = rng.choice([
        "remove the arrow from {A} to {B}", "disconnect {A} from {B}", "delete the connection between {A} and {B}",
        "{A} no longer talks to {B}, remove that arrow", "drop the {A} -> {B} link", "cut the edge from {A} to {B}",
        "unlink {A} and {B}", "take out the arrow {A} to {B}",
    ]).format(A=mention(a, rng), B=mention(b, rng))
    return text, [delete([edge_ref(a, b)])]


def op_redirect(canvas, rng, step):
    if not canvas.edges:
        return None
    a, b, label = rng.choice(canvas.edges)
    others = [n for n in canvas.nodes if n["id"] not in (a["id"], b["id"]) and not canvas.has_edge(a, n)]
    if not others:
        return None
    c = rng.choice(others)
    text = rng.choice([
        "point {A} at {C} instead of {B}", "make {A} connect to {C} rather than {B}", "{A} should call {C}, not {B}",
        "move the arrow from {A} to {B} so it goes to {C}", "reroute {A} from {B} to {C}",
        "redirect {A}'s arrow from {B} to {C}", "{A} -> {B} becomes {A} -> {C}",
    ]).format(A=mention(a, rng), B=mention(b, rng), C=mention(c, rng))
    return text, [delete([edge_ref(a, b)]), connect(ref(a), ref(c), label)]


def op_replace(canvas, rng, step):
    if not canvas.nodes:
        return None
    old = rng.choice(canvas.nodes)
    made = new_node(canvas, rng)
    if not made:
        return None
    phrase, create = made
    text = rng.choice([
        "replace {A} with {N}", "swap {A} for {N}", "swap out {A} with {N}", "switch {A} to {N}",
        "use {N} instead of {A}", "{A} becomes {N}, keep its arrows", "change {A} into {N}",
    ]).format(A=mention(old, rng), N=phrase)
    actions = [placed(create, old, "right")]
    for a, b, label in canvas.edges:
        if a["id"] == old["id"]:
            actions.append(connect({"new": step}, ref(b), label))
        elif b["id"] == old["id"]:
            actions.append(connect(ref(a), {"new": step}, label))
    actions.append(delete([ref(old)]))
    return text, actions


def op_rename(canvas, rng, step):
    if not canvas.nodes:
        return None
    node = rng.choice(canvas.nodes)
    taken = {o["name"].casefold() for o in canvas.nodes}
    if "kind" not in node:
        pool = BOX_NAMES
    elif str(node["kind"]).startswith("component:"):
        pool = NAMES.get(node["kind"].split(":", 1)[1], []) + ["Primary " + node["name"], node["name"] + " v2"]
    else:
        pool = SHAPE_NAMES
    names = [n for n in pool if n.casefold() not in taken]
    if not names:
        return None
    new = rng.choice(names)
    text = rng.choice([
        "rename {A} to {X}", "call {A} {X}", "change {A}'s label to {X}", "relabel {A} as {X}",
        "{A} should be called {X}", 'rename {A} to "{X}"', "change the name of {A} to {X}",
    ]).format(A=mention(node, rng), X=new)
    if "kind" not in node:
        return text, [{"name": "rename_schema", "arguments": {"schema_id": ref(node), "new_name": new}}]
    return text, [{"name": "set_text", "arguments": {"shape_id": ref(node), "text": new}}]


def op_relabel_edge(canvas, rng, step):
    if not canvas.edges:
        return None
    a, b, old = rng.choice(canvas.edges)
    label = rng.choice([x for x in EDGE_LABELS if x and x != old])
    text = rng.choice([
        "label the arrow from {A} to {B} {L}", 'set the label on {A} -> {B} to "{L}"',
        "name the {A} to {B} connection {L}", 'the edge from {A} to {B} should say "{L}"',
        "put {L} on the arrow between {A} and {B}", "change the {A} to {B} label to {L}",
    ]).format(A=mention(a, rng), B=mention(b, rng), L=label)
    # The arrow is redrawn with its new label: set_text would change only the arrow shape's text.
    return text, [delete([edge_ref(a, b)]), connect(ref(a), ref(b), label)]


def op_connect(canvas, rng, step):
    if len(canvas.nodes) < 2:
        return None
    a, b = rng.sample(canvas.nodes, 2)
    if canvas.has_edge(a, b):
        return None
    text = rng.choice([
        "connect {A} to {B}", "draw an arrow from {A} to {B}", "link {A} to {B}", "{A} should call {B}",
        "add an edge {A} -> {B}", "wire {A} into {B}", "{A} talks to {B}, add that arrow",
    ]).format(A=mention(a, rng), B=mention(b, rng))
    label = ""
    if rng.random() < 0.35:
        suffix, label = label_phrase(rng)
        text += suffix
    return text, [connect(ref(a), ref(b), label)]


def op_delete_node(canvas, rng, step):
    if not canvas.nodes:
        return None
    node = rng.choice(canvas.nodes)
    text = rng.choice(["delete {A}", "remove {A}", "drop {A}", "get rid of {A}", "take {A} out",
                       "{A} is not needed, remove it"]).format(A=mention(node, rng))
    return text, [delete([ref(node)])]


def op_style(canvas, rng, step):
    if not canvas.nodes:
        return None
    node, color = rng.choice(canvas.nodes), rng.choice(COLORS)
    text = rng.choice(["make {A} {C}", "color {A} {C}", "paint {A} {C}", "turn {A} {C}"]).format(
        A=mention(node, rng), C=color)
    return text, [{"name": "style_shapes", "arguments": {"shape_ids": [ref(node)], "color": color}}]


OPS = {name[3:]: fn for name, fn in list(globals().items()) if name.startswith("op_")}


def polish(text, rng):
    roll = rng.random()
    if roll < 0.15:
        text = rng.choice(["please ", "pls ", "can you ", "could you "]) + text
    elif roll < 0.5:
        text = text[0].upper() + text[1:]
    if rng.random() < 0.3:
        text += rng.choice([".", ".", "!", " please", " pls"])
    return text


def applied(diagram, script):
    """The diagram after the script, or None when it does not apply or changes nothing."""
    before = session_of(diagram)
    after = copy.deepcopy(before)
    try:
        if run_script(script, after, "", "new"):
            return None
    except (ValueError, KeyError, TypeError):
        return None
    if semantic_state(before) == semantic_state(after):
        return None
    return after


def make_row(diagram, rng, compound):
    canvas = Canvas(diagram)
    names = list(WEIGHTS)
    first = rng.choices(names, [WEIGHTS[n] for n in names])[0]
    made = OPS[first](canvas, rng, 1)
    if not made:
        return None
    text, actions, ops = made[0], made[1], [first]
    if compound:
        second = rng.choices(names, [WEIGHTS[n] for n in names])[0]
        more = OPS[second](canvas, rng, len(actions) + 1)
        if not more:
            return None
        # Both parts name the diagram as it was; a later part must not touch what the
        # earlier one changed, which the simulator run below checks by applying them in order.
        text = text + rng.choice(JOINERS) + more[0]
        actions, ops = actions + more[1], ops + [second]
    script = {"actions": actions}
    if applied(diagram, script) is None:
        return None
    return {"diagram": diagram, "command": polish(text, rng), "script": script, "outcome": "applied",
            "source": "precise", "op": "+".join(ops)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8000)
    parser.add_argument("--diagrams", type=Path, help="synthetic rows whose diagrams to edit as well")
    parser.add_argument("--generated", type=float, default=0.6, help="share of rows on generated system designs")
    parser.add_argument("--compound", type=float, default=0.15, help="share of rows with two edits")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    loaded = loaded_diagrams(args.diagrams) if args.diagrams else []
    rows, seen, ops, failures = [], set(), Counter(), 0
    while len(rows) < args.count and failures < args.count * 20:
        if loaded and rng.random() >= args.generated:
            diagram = rng.choice(loaded)
        else:
            diagram = architecture(rng)
        row = make_row(diagram, rng, rng.random() < args.compound)
        key = row and (json.dumps(row["diagram"], sort_keys=True), row["command"].casefold())
        if not row or key in seen:
            failures += 1
            continue
        seen.add(key)
        rows.append(row)
        ops[row["op"]] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    dev = sum(int(hashlib.sha256(json.dumps(r["diagram"], sort_keys=True).encode()).hexdigest(), 16) % 10 == 0
              for r in rows)
    print(json.dumps({"rows": len(rows), "held_out_by_prepare_scripts": dev, "ops": dict(ops.most_common())}, indent=1))


if __name__ == "__main__":
    main()
