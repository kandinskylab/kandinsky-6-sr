import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "K6VSR.LoaderDefaults",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "K6VSRLoadModel") return;

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            const values = info.widgets_values;
            // Older workflows stored checkpoint, VAE, upscalers, backend, device.
            const legacy = Array.isArray(values) && values.length === 5;
            for (const [name, index, fallback] of [
                ["vae_backend", 3, "torch"],
                ["device", 4, "cuda:0"],
            ]) {
                const widget = this.widgets?.find((item) => item.name === name);
                if (!widget) continue;
                const value = legacy ? values[index] : widget.value;
                widget.value = typeof value === "string" && value.trim() ? value : fallback;
            }
            return result;
        };
    },
});
