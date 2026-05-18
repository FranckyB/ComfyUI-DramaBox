/**
 * DramaBox TTS — prompt ghost widget
 *
 * When the "prompt_input" connector has an incoming link:
 *   - Ghost the "prompt" text widget (read-only, grayed out)
 *   - Keep pointerEvents so the user can still scroll
 *   - After execution the server syncs the actual text into the widget
 *
 * When disconnected the widget becomes editable again; whatever text
 * is currently in it is preserved (not cleared).
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME   = "DramaBoxTTS";
const TEXT_WIDGET = "text";
const INPUT_SLOT  = "prompt";
const TOGGLE_WIDGET = "use_prompt_input";

// ── helpers ──────────────────────────────────────────────────────────────────

function getTextWidget(node) {
    return node.widgets?.find(w => w.name === TEXT_WIDGET);
}

function getToggleWidget(node) {
    return node.widgets?.find(w => w.name === TOGGLE_WIDGET);
}

function isInputConnected(node) {
    return !!(node.inputs?.find(i => i.name === INPUT_SLOT)?.link);
}

function shouldGhost(node) {
    return isInputConnected(node) && (getToggleWidget(node)?.value === true);
}

function applyGhost(node) {
    const w = getTextWidget(node);
    if (!w) return;
    w.disabled = true;
    if (w.inputEl) {
        w.inputEl.readOnly            = true;
        w.inputEl.style.pointerEvents = "auto";   // keep scroll alive
    }
}

function removeGhost(node) {
    const w = getTextWidget(node);
    if (!w) return;
    w.disabled = false;
    if (w.inputEl) {
        w.inputEl.readOnly            = false;
        w.inputEl.style.pointerEvents = "";
    }
}

function refreshGhost(node) {
    if (shouldGhost(node)) {
        applyGhost(node);
    } else {
        removeGhost(node);
    }
}

// ── extension ────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "DramaBox.PromptGhost",

    nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME) return;

        // Give the node a comfortable initial height on first creation only.
        // onConfigure fires synchronously (before rAF) when loading a saved
        // workflow and sets the flag, so saved sizes are always respected.
        requestAnimationFrame(() => {
            if (!node._configuredFromWorkflow) {
                node.size[1] = Math.max(node.size[1], 500);
                node.setDirtyCanvas(true, true);
            }
        });

        // Mark as loaded-from-workflow so the rAF above is suppressed.
        const _origConfigure = node.onConfigure?.bind(node);
        node.onConfigure = function (info) {
            node._configuredFromWorkflow = true;
            _origConfigure?.(info);
        };

        // Set pointerEvents once so the textarea always scrolls,
        // even before the first sync (matches PM behaviour).
        setTimeout(() => {
            const w = getTextWidget(node);
            if (w?.inputEl) {
                w.inputEl.style.pointerEvents = "auto";
                // Allow scroll but block editing clicks while ghosted
                w.inputEl.addEventListener("mousedown", function (e) {
                    if (this.readOnly) e.stopPropagation();
                });
            }

            // Hook the toggle widget so ghosting updates immediately when flipped
            const toggle = getToggleWidget(node);
            if (toggle) {
                const _origCb = toggle.callback;
                toggle.callback = function (...args) {
                    _origCb?.(...args);
                    refreshGhost(node);
                };
            }

            // Restore ghost state when loading a saved workflow
            refreshGhost(node);
        }, 50);

        // ── connection change hook ────────────────────────────────────────
        const _orig = node.onConnectionsChange?.bind(node);
        node.onConnectionsChange = function (...args) {
            _orig?.(...args);
            // Defer so node.inputs reflects the updated link state
            setTimeout(() => refreshGhost(this), 0);
        };
    },
});

// ── server sync ──────────────────────────────────────────────────────────────
// After each execution Python sends the prompt text that was actually used.
// We push it into the widget so the user can see / scroll the live text.

api.addEventListener("dramabox-tts-update", (event) => {
    const detail = event?.detail;
    if (!detail) return;

    const nodeId = String(detail.node_id ?? "");
    if (!nodeId) return;

    const node = app.graph?.getNodeById(Number(nodeId));
    if (!node || node.comfyClass !== NODE_NAME) return;

    const w = getTextWidget(node);
    if (!w) return;

    if (detail.prompt !== undefined) {
        w.value = detail.prompt;
        if (w.inputEl) w.inputEl.value = detail.prompt;
    }

    // Re-apply ghost in case the sync arrives before onConnectionsChange settles
    refreshGhost(node);
});

// ── DramaBox settings ────────────────────────────────────────────────────────

app.registerExtension({
    name: "DramaBox.Settings",
    settings: [
        {
            id: "DramaBox.defaultTextEncoder",
            name: "Default Text Encoder filename",
            category: ["DramaBox", "Text Encoder", "Default"],
            type: "text",
            defaultValue: "",
            tooltip:
                "Filename of the Gemma text encoder to auto-load (.safetensors or .gguf). " +
                "File must exist in your text_encoders folder (or clip_gguf when available). " +
                "Leave blank to use gemma_3_12B_it_fp4_mixed.safetensors " +
                "(downloaded automatically on first use). Connect a DramaBox CLIP Loader node " +
                "to override per-workflow.",
        },
        {
            id: "DramaBox.autoOffload",
            name: "Automatic model offload (text encoder + post-generate)",
            category: ["DramaBox", "Memory"],
            type: "boolean",
            defaultValue: true,
            tooltip:
                "When enabled (recommended), DramaBox offloads Gemma right after prompt " +
                "encoding and offloads DramaBox models to CPU after generation. Disable " +
                "if you have very high VRAM and prefer maximum throughput.",
        },
    ],
});

