/**
 * FluxFlow Custom Types Registration for ComfyUI
 * 
 * This extension registers custom FluxFlow types with proper colors
 * and enables better node connectivity visualization.
 */

import { app } from "../../scripts/app.js";

console.log("[FluxFlow] Extension loading...");

// Define FluxFlow custom type colors
const FLUXFLOW_COLORS = {
    "FLUXFLOW_MODEL": "#8B5CF6",           // Purple - main model
    "FLUXFLOW_TEXT_ENCODER": "#10B981",   // Green - text encoder
    "FLUXFLOW_TOKENIZER": "#059669",      // Dark green - tokenizer
    "FLUXFLOW_CONDITIONING": "#F59E0B",   // Amber - conditioning data
    "FLUXFLOW_LATENT": "#3B82F6",         // Blue - latent space
};

// Register extension
app.registerExtension({
    name: "FluxFlow.CustomTypes",
    
    async setup() {
        console.log("[FluxFlow] Extension setup() called");
        
        // Access LiteGraph directly from window
        const LiteGraph = window.LiteGraph;
        const LGraphCanvas = window.LGraphCanvas;
        
        if (!LiteGraph || !LGraphCanvas) {
            console.error("[FluxFlow] LiteGraph not found in window!");
            return;
        }
        
        console.log("[FluxFlow] LiteGraph found, registering types...");
        console.log("[FluxFlow] Current link_type_colors:", LGraphCanvas.link_type_colors);
        
        // Force register custom type colors with LiteGraph
        for (const [type, color] of Object.entries(FLUXFLOW_COLORS)) {
            // Register in multiple places to ensure it sticks
            LiteGraph.registered_slot_types = LiteGraph.registered_slot_types || {};
            LiteGraph.registered_slot_types[type] = { color: color };
            
            // Force link colors
            LGraphCanvas.link_type_colors = LGraphCanvas.link_type_colors || {};
            LGraphCanvas.link_type_colors[type] = color;
            
            // Also try slot_types_out and slot_types_in
            if (LGraphCanvas.slot_types_out) {
                LGraphCanvas.slot_types_out[type] = { color: color };
            }
            if (LGraphCanvas.slot_types_in) {
                LGraphCanvas.slot_types_in[type] = { color: color };
            }
            
            console.log(`[FluxFlow] Registered type: ${type} with color ${color}`);
        }
        
        console.log("[FluxFlow] All types registered successfully");
        console.log("[FluxFlow] Final link_type_colors keys:", Object.keys(LGraphCanvas.link_type_colors || {}));
        console.log("[FluxFlow] Our FLUXFLOW types in link_type_colors:", {
            FLUXFLOW_MODEL: LGraphCanvas.link_type_colors.FLUXFLOW_MODEL,
            FLUXFLOW_LATENT: LGraphCanvas.link_type_colors.FLUXFLOW_LATENT,
        });
        console.log("[FluxFlow] Registered types:", Object.keys(FLUXFLOW_COLORS));
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only process FluxFlow nodes
        if (!nodeData.name || !nodeData.name.startsWith("FluxFlow")) {
            return;
        }
        
        console.log(`[FluxFlow] Processing node: ${nodeData.name}`);
        
        // Override getSlotMenuOptions to add color info
        const originalGetSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
        nodeType.prototype.getSlotMenuOptions = function(slot) {
            const options = originalGetSlotMenuOptions ? originalGetSlotMenuOptions.call(this, slot) : [];
            const color = FLUXFLOW_COLORS[slot.type];
            if (color) {
                slot.color_on = color;
                slot.color_off = color + "88";
            }
            return options;
        };
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            
            console.log(`[FluxFlow] Node created: ${this.type}`, this);
            
            // Apply colors to output slots
            if (this.outputs) {
                for (let i = 0; i < this.outputs.length; i++) {
                    const output = this.outputs[i];
                    const color = FLUXFLOW_COLORS[output.type];
                    if (color) {
                        output.color_on = color;
                        output.color_off = color + "88";
                        console.log(`[FluxFlow]   Output slot ${i} (${output.type}):`, output);
                    }
                }
            }
            
            // Apply colors to input slots
            if (this.inputs) {
                for (let i = 0; i < this.inputs.length; i++) {
                    const input = this.inputs[i];
                    const color = FLUXFLOW_COLORS[input.type];
                    if (color) {
                        input.color_on = color;
                        input.color_off = color + "88";
                        console.log(`[FluxFlow]   Input slot ${i} (${input.type}):`, input);
                    }
                }
            }
            
            // Force redraw
            if (this.setDirtyCanvas) {
                this.setDirtyCanvas(true, true);
            }
            
            return result;
        };
    },
});
