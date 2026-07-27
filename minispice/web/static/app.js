/* MiniSPICE browser UI: netlist -> POST /simulate -> circuit diagram + results.
 *
 * The diagram layout is a small, self-contained force-directed graph
 * (Fruchterman-Reingold style): node positions aren't part of a SPICE
 * netlist at all (it's purely topological), so we compute a reasonable
 * layout ourselves rather than depending on an external graphing library.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

const netlistEl = document.getElementById("netlist");
const simulateBtn = document.getElementById("simulate-btn");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const diagramEl = document.getElementById("diagram");
const resultsEl = document.getElementById("results");

simulateBtn.addEventListener("click", simulate);
document.addEventListener("DOMContentLoaded", simulate);
// Also allow Ctrl+Enter / Cmd+Enter from the textarea as a shortcut.
netlistEl.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        simulate();
    }
});

async function simulate() {
    setStatus("Simulating...");
    simulateBtn.disabled = true;

    try {
        const response = await fetch("/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ netlist: netlistEl.value }),
        });
        const data = await response.json();

        if (!data.ok) {
            showError(data.error);
            clearDiagram();
            resultsEl.innerHTML = "";
            setStatus("Error");
            return;
        }

        hideError();
        renderDiagram(data.circuit);
        renderResults(data.results);
        setStatus(`OK — ${data.circuit.components.length} component(s), ${data.circuit.nodes.length} node(s)`);
    } catch (err) {
        showError(`Request failed: ${err}`);
        setStatus("Error");
    } finally {
        simulateBtn.disabled = false;
    }
}

function setStatus(text) {
    statusEl.textContent = text;
}

function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = false;
}

function hideError() {
    errorEl.hidden = true;
    errorEl.textContent = "";
}

// ---------------------------------------------------------------------------
// Diagram: force-directed layout + SVG rendering
// ---------------------------------------------------------------------------

function clearDiagram() {
    while (diagramEl.firstChild) diagramEl.removeChild(diagramEl.firstChild);
}

function renderDiagram(circuit) {
    clearDiagram();

    const width = 600;
    const height = 420;

    const nodes = circuit.nodes.map((name) => ({ id: name, isGround: name === "0" }));
    const edges = circuit.components.map((c) => ({
        source: c.node_pos,
        target: c.node_neg,
        label: `${c.name} ${c.value_display}`,
        type: c.type,
    }));

    layoutGraph(nodes, edges, width, height);

    const nodeById = Object.fromEntries(nodes.map((n) => [n.id, n]));

    // Edges first, so node circles draw on top of the lead wires.
    edges.forEach((edge) => {
        const a = nodeById[edge.source];
        const b = nodeById[edge.target];
        if (!a || !b) return;

        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const length = Math.max(Math.hypot(dx, dy), 0.01);
        const midX = (a.x + b.x) / 2;
        const midY = (a.y + b.y) / 2;
        const angleDeg = (Math.atan2(dy, dx) * 180) / Math.PI;

        const symbol = buildComponentSymbol(edge.type, length);
        symbol.setAttribute("transform", `translate(${midX} ${midY}) rotate(${angleDeg})`);
        diagramEl.appendChild(symbol);

        // Labels stay in world space (never rotated) so they're always
        // readable, regardless of which way the component is oriented.
        // Offset perpendicular to the wire so it clears the symbol body.
        const perpX = -dy / length;
        const perpY = dx / length;
        const offset = 20;
        const label = document.createElementNS(SVG_NS, "text");
        label.setAttribute("x", midX + perpX * offset);
        label.setAttribute("y", midY + perpY * offset);
        label.setAttribute("class", "edge-label");
        label.textContent = edge.label;
        diagramEl.appendChild(label);
    });

    nodes.forEach((node) => {
        const circle = document.createElementNS(SVG_NS, "circle");
        circle.setAttribute("cx", node.x);
        circle.setAttribute("cy", node.y);
        circle.setAttribute("r", node.isGround ? 9 : 7);
        circle.setAttribute("class", "node-circle" + (node.isGround ? " ground" : ""));
        diagramEl.appendChild(circle);

        const label = document.createElementNS(SVG_NS, "text");
        label.setAttribute("x", node.x);
        label.setAttribute("y", node.y - 14);
        label.setAttribute("class", "node-label");
        label.textContent = node.isGround ? "0 (gnd)" : node.id;
        diagramEl.appendChild(label);
    });
}

// ---------------------------------------------------------------------------
// Component symbols
//
// Each builder draws its symbol in local coordinates along the local
// x-axis, spanning from -length/2 to +length/2 with the centerline at
// y=0 -- as if the component's two nodes were "in" (left) and "out"
// (right). The caller then positions the whole group with a single
// translate+rotate transform, so these functions never need to know
// the component's actual position or orientation on the canvas.
// ---------------------------------------------------------------------------

function buildComponentSymbol(type, length) {
    const g = document.createElementNS(SVG_NS, "g");
    switch (type) {
        case "Resistor":
            buildResistorSymbol(g, length);
            break;
        case "Capacitor":
            buildCapacitorSymbol(g, length);
            break;
        case "Inductor":
            buildInductorSymbol(g, length);
            break;
        case "VoltageSource":
            buildSourceCircleSymbol(g, length, "voltage");
            break;
        case "CurrentSource":
            buildSourceCircleSymbol(g, length, "current");
            break;
        default:
            // Unknown/future component type: fall back to a plain wire
            // rather than failing to render anything at all.
            addLead(g, -length / 2, length / 2);
    }
    return g;
}

/** A straight lead-wire segment along the local x-axis at y=0. */
function addLead(g, x1, x2) {
    if (x2 - x1 <= 0.5) return; // no room left (nodes very close together)
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", 0);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", 0);
    line.setAttribute("class", "lead-line");
    g.appendChild(line);
}

/** The classic zigzag resistor symbol. */
function buildResistorSymbol(g, length) {
    const half = length / 2;
    const bodyWidth = Math.min(44, Math.max(length - 12, 20));
    const bodyHalf = bodyWidth / 2;
    addLead(g, -half, -bodyHalf);
    addLead(g, bodyHalf, half);

    const peakHeight = 9;
    const segments = 6;
    let d = `M ${-bodyHalf} 0`;
    for (let i = 1; i < segments; i++) {
        const x = -bodyHalf + (bodyWidth * i) / segments;
        const y = i % 2 === 1 ? -peakHeight : peakHeight;
        d += ` L ${x} ${y}`;
    }
    d += ` L ${bodyHalf} 0`;

    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("class", "component-body");
    g.appendChild(path);
}

/** Two parallel plates, the standard capacitor symbol. */
function buildCapacitorSymbol(g, length) {
    const half = length / 2;
    const gap = 8;
    const plateHalfHeight = 13;
    addLead(g, -half, -gap / 2);
    addLead(g, gap / 2, half);

    [-gap / 2, gap / 2].forEach((x) => {
        const plate = document.createElementNS(SVG_NS, "line");
        plate.setAttribute("x1", x);
        plate.setAttribute("y1", -plateHalfHeight);
        plate.setAttribute("x2", x);
        plate.setAttribute("y2", plateHalfHeight);
        plate.setAttribute("class", "component-body");
        g.appendChild(plate);
    });
}

/** A row of loops, the standard coil/inductor symbol. */
function buildInductorSymbol(g, length) {
    const half = length / 2;
    const bodyWidth = Math.min(48, Math.max(length - 12, 24));
    const bodyHalf = bodyWidth / 2;
    addLead(g, -half, -bodyHalf);
    addLead(g, bodyHalf, half);

    const bumps = 4;
    const r = bodyWidth / (2 * bumps);
    let d = `M ${-bodyHalf} 0`;
    for (let i = 0; i < bumps; i++) {
        const x = -bodyHalf + r * (2 * i + 2);
        d += ` A ${r} ${r} 0 1 1 ${x} 0`;
    }

    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("class", "component-body");
    g.appendChild(path);
}

/** A circle with either a +/- pair (voltage source) or a directional
 * arrow (current source) inside, the standard ideal-source symbols.
 * The arrow points node_pos -> node_neg, matching the sign convention
 * used everywhere else in MiniSPICE (see CurrentSource's docstring).
 */
function buildSourceCircleSymbol(g, length, kind) {
    const half = length / 2;
    const r = 15;
    addLead(g, -half, -r);
    addLead(g, r, half);

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", 0);
    circle.setAttribute("cy", 0);
    circle.setAttribute("r", r);
    circle.setAttribute("class", "source-circle");
    g.appendChild(circle);

    if (kind === "voltage") {
        const plus = document.createElementNS(SVG_NS, "text");
        plus.setAttribute("x", -r * 0.5);
        plus.setAttribute("y", 4);
        plus.setAttribute("class", "source-sign");
        plus.textContent = "+";
        g.appendChild(plus);

        const minus = document.createElementNS(SVG_NS, "text");
        minus.setAttribute("x", r * 0.5);
        minus.setAttribute("y", 4);
        minus.setAttribute("class", "source-sign");
        minus.textContent = "−";
        g.appendChild(minus);
    } else {
        const shaftHalf = r * 0.55;
        const headLength = 5;
        const headWidth = 4;
        const arrow = document.createElementNS(SVG_NS, "path");
        arrow.setAttribute(
            "d",
            `M ${-shaftHalf} 0 L ${shaftHalf} 0 ` +
                `M ${shaftHalf - headLength} ${-headWidth} L ${shaftHalf} 0 L ${shaftHalf - headLength} ${headWidth}`
        );
        arrow.setAttribute("class", "component-body");
        g.appendChild(arrow);
    }
}

/** Fruchterman-Reingold style force-directed layout, mutating each node's x/y. */
function layoutGraph(nodes, edges, width, height) {
    const n = nodes.length;
    if (n === 0) return;

    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.3;

    nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / n;
        node.x = cx + radius * Math.cos(angle);
        node.y = cy + radius * Math.sin(angle);
    });

    if (n === 1) return; // nothing to layout beyond centering

    const k = Math.sqrt((width * height) / n);
    const iterations = 300;
    const damping = 0.85;
    const maxDisplacement = 20;

    for (let iter = 0; iter < iterations; iter++) {
        nodes.forEach((node) => {
            node.fx = 0;
            node.fy = 0;
        });

        // Repulsion between every pair of nodes.
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const a = nodes[i];
                const b = nodes[j];
                const dx = a.x - b.x;
                const dy = a.y - b.y;
                const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 0.01);
                const force = (k * k) / dist;
                const ux = dx / dist;
                const uy = dy / dist;
                a.fx += ux * force;
                a.fy += uy * force;
                b.fx -= ux * force;
                b.fy -= uy * force;
            }
        }

        // Spring attraction along each component (edge).
        edges.forEach((edge) => {
            const a = nodes.find((node) => node.id === edge.source);
            const b = nodes.find((node) => node.id === edge.target);
            if (!a || !b || a === b) return;
            const dx = a.x - b.x;
            const dy = a.y - b.y;
            const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 0.01);
            const force = (dist * dist) / k;
            const ux = dx / dist;
            const uy = dy / dist;
            a.fx -= ux * force;
            a.fy -= uy * force;
            b.fx += ux * force;
            b.fy += uy * force;
        });

        nodes.forEach((node) => {
            // Mild pull toward the canvas center so the graph doesn't drift off-screen.
            node.fx += (cx - node.x) * 0.01;
            node.fy += (cy - node.y) * 0.01;
            // Conventional circuit-diagram bias: ground tends to sit at the bottom.
            if (node.isGround) {
                node.fy += (height * 0.85 - node.y) * 0.03;
            }

            const disp = Math.max(Math.sqrt(node.fx * node.fx + node.fy * node.fy), 0.01);
            const limited = Math.min(disp, maxDisplacement);
            node.x += (node.fx / disp) * limited * damping;
            node.y += (node.fy / disp) * limited * damping;
            node.x = Math.max(30, Math.min(width - 30, node.x));
            node.y = Math.max(30, Math.min(height - 30, node.y));
        });
    }
}

// ---------------------------------------------------------------------------
// Results panel
// ---------------------------------------------------------------------------

function renderResults(results) {
    resultsEl.innerHTML = "";
    resultsEl.appendChild(buildTable("Node Voltages", results.node_voltages));
    resultsEl.appendChild(buildTable("Resistor Currents", results.resistor_currents));
    resultsEl.appendChild(buildTable("Resistor Power Dissipation", results.resistor_power));
    resultsEl.appendChild(buildTable("Voltage Source Currents", results.source_currents));
    resultsEl.appendChild(buildTable("Inductor Currents", results.inductor_currents));
}

function buildTable(caption, namedValues) {
    const wrapper = document.createElement("div");
    const entries = Object.entries(namedValues || {});

    const captionEl = document.createElement("p");
    captionEl.className = "table-caption";
    captionEl.textContent = caption;

    if (entries.length === 0) {
        wrapper.appendChild(captionEl);
        const empty = document.createElement("p");
        empty.className = "empty";
        empty.textContent = "(none)";
        wrapper.appendChild(empty);
        return wrapper;
    }

    const table = document.createElement("table");
    const captionTag = document.createElement("caption");
    captionTag.textContent = caption;
    table.appendChild(captionTag);

    entries.forEach(([name, entry]) => {
        const row = document.createElement("tr");
        const nameCell = document.createElement("td");
        nameCell.textContent = name;
        const valueCell = document.createElement("td");
        valueCell.className = "value";
        valueCell.textContent = entry.display;
        row.appendChild(nameCell);
        row.appendChild(valueCell);
        table.appendChild(row);
    });

    wrapper.appendChild(table);
    return wrapper;
}
