export default function App() {
  const backbone = [
    { id: 0, label: "Conv", sub: "3×3 s=2", shape: "384×384×32", color: "#3b82f6", tag: "STEM" },
    { id: 1, label: "GhostConv", sub: "3×3 s=2", shape: "192×192×64", color: "#8b5cf6", tag: "P2" },
    { id: 2, label: "C3Ghost ×2", sub: "64ch", shape: "192×192×64", color: "#7c3aed", tag: "P2" },
    { id: 3, label: "GhostConv", sub: "3×3 s=2", shape: "96×96×128", color: "#06b6d4", tag: "P3★" },
    { id: 4, label: "C3Ghost ×3", sub: "128ch", shape: "96×96×128", color: "#0891b2", tag: "P3★" },
    { id: 5, label: "GhostConv", sub: "3×3 s=2", shape: "48×48×160", color: "#f59e0b", tag: "P4★" },
    { id: 6, label: "C3Ghost ×3", sub: "160ch", shape: "48×48×160", color: "#d97706", tag: "P4★" },
    { id: 7, label: "GhostConv", sub: "3×3 s=2", shape: "24×24×160", color: "#6b7280", tag: "P5" },
    { id: 8, label: "C3Ghost ×2", sub: "160ch", shape: "24×24×160", color: "#4b5563", tag: "P5" },
    { id: 9, label: "SPPF", sub: "k=5", shape: "24×24×160", color: "#0ea5e9", tag: "SPPF" },
  ];

  const headTD = [
    { id: 10, label: "Upsample ×2", sub: "nearest", shape: "48×48×160", color: "#f97316", merge: null },
    { id: 11, label: "Concat", sub: "L10+L6", shape: "48×48×320", color: "#ea580c", merge: "← FPN skip from L6 (P4★)" },
    { id: 12, label: "DWConv", sub: "3×3→80ch", shape: "48×48×80", color: "#ef4444", merge: null },
    { id: 13, label: "Conv 1×1", sub: "80ch", shape: "48×48×80", color: "#dc2626", merge: null },
    { id: 14, label: "Upsample ×2", sub: "nearest", shape: "96×96×80", color: "#f97316", merge: null },
    { id: 15, label: "Concat", sub: "L14+L4", shape: "96×96×208", color: "#ea580c", merge: "← FPN skip from L4 (P3★)" },
    { id: 16, label: "DWConv", sub: "3×3→64ch", shape: "96×96×64", color: "#ef4444", merge: null },
    { id: 17, label: "Conv 1×1 ★", sub: "P3 OUTPUT", shape: "96×96×64", color: "#16a34a", merge: null },
  ];

  const headBU = [
    { id: 18, label: "DWConv", sub: "3×3 s=2", shape: "48×48×64", color: "#ef4444", merge: null },
    { id: 19, label: "Conv 1×1", sub: "80ch", shape: "48×48×80", color: "#dc2626", merge: null },
    { id: 20, label: "Concat", sub: "L19+L13", shape: "48×48×160", color: "#22c55e", merge: "← PAN skip from L13" },
    { id: 21, label: "DWConv", sub: "3×3→80ch", shape: "48×48×80", color: "#ef4444", merge: null },
    { id: 22, label: "Conv 1×1 ★", sub: "P4 OUTPUT", shape: "48×48×80", color: "#16a34a", merge: null },
  ];

  const tagColor = { "STEM":"#3b82f6","P2":"#8b5cf6","P3★":"#06b6d4","P4★":"#f59e0b","P5":"#6b7280","SPPF":"#0ea5e9" };

  return (
    <div style={{
      width: "100%", aspectRatio: "16/9", background: "#050d1a",
      fontFamily: "'Courier New', monospace", color: "#e2e8f0",
      display: "flex", flexDirection: "column", padding: "14px 18px", boxSizing: "border-box",
      overflow: "hidden"
    }}>
      {/* Title row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 7, letterSpacing: 4, color: "#475569" }}>ARCHITECTURE DIAGRAM</div>
          <div style={{ fontSize: 14, fontWeight: 700, color: "#f8fafc", letterSpacing: 0.5 }}>YOLO11n-Ghost-Hybrid-P3P4-Medium</div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {[["867K","Params"],["5.7","GFLOPs"],["768×768","Input"],["96.10%","mAP50"]].map(([v,l]) => (
            <div key={l} style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6, padding: "3px 10px", textAlign: "center" }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#38bdf8" }}>{v}</div>
              <div style={{ fontSize: 7, color: "#475569" }}>{l}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Main 4-column layout */}
      <div style={{ display: "flex", gap: 8, flex: 1, overflow: "hidden" }}>

        {/* BACKBONE */}
        <div style={{ flex: 1.1, display: "flex", flexDirection: "column" }}>
          <div style={{ fontSize: 7, letterSpacing: 3, color: "#3b82f6", fontWeight: 700, marginBottom: 4, borderBottom: "1px solid #1e3a5f", paddingBottom: 3 }}>
            BACKBONE — GhostConv + C3Ghost
          </div>
          <div style={{ border: "1px dashed #334155", borderRadius: 5, padding: "3px 8px", textAlign: "center", marginBottom: 3, background: "#0f172a" }}>
            <div style={{ fontSize: 7, color: "#64748b" }}>INPUT</div>
            <div style={{ fontSize: 9, fontWeight: 700, color: "#94a3b8" }}>768 × 768 × 3</div>
          </div>
          <div style={{ fontSize: 9, color: "#1e3a5f", textAlign: "center", lineHeight: "9px", marginBottom: 2 }}>↓</div>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            {backbone.map((layer, i) => (
              <div key={layer.id} style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
                <div style={{ width: "100%", background: layer.color + "18", border: `1px solid ${layer.color}55`, borderLeft: `3px solid ${layer.color}`, borderRadius: 4, padding: "2px 7px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <span style={{ fontSize: 7, color: "#475569" }}>[{layer.id}]</span>
                    <span style={{ fontSize: 7, padding: "1px 4px", borderRadius: 2, background: (tagColor[layer.tag]||"#475569")+"22", color: tagColor[layer.tag]||"#94a3b8", fontWeight: 700 }}>{layer.tag}</span>
                    <span style={{ fontSize: 9, fontWeight: 700, color: "#f1f5f9" }}>{layer.label}</span>
                    <span style={{ fontSize: 7, color: "#64748b" }}>{layer.sub}</span>
                  </div>
                  <span style={{ fontSize: 7, color: "#38bdf8", fontFamily: "monospace" }}>{layer.shape}</span>
                </div>
                {i < backbone.length - 1 && <div style={{ fontSize: 9, color: "#1e3a5f", lineHeight: "9px" }}>↓</div>}
              </div>
            ))}
          </div>
        </div>

        {/* Skip connection visual */}
        <div style={{ width: 24, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "space-around", paddingTop: 20 }}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
            <div style={{ fontSize: 6, color: "#f59e0b", writingMode: "vertical-rl", transform: "rotate(180deg)", letterSpacing: 1 }}>P4★</div>
            <div style={{ width: 1, height: 60, background: "linear-gradient(#f59e0b44,#f59e0b,#f59e0b44)" }} />
            <div style={{ fontSize: 8, color: "#f59e0b" }}>→</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
            <div style={{ fontSize: 6, color: "#06b6d4", writingMode: "vertical-rl", transform: "rotate(180deg)", letterSpacing: 1 }}>P3★</div>
            <div style={{ width: 1, height: 60, background: "linear-gradient(#06b6d444,#06b6d4,#06b6d444)" }} />
            <div style={{ fontSize: 8, color: "#06b6d4" }}>→</div>
          </div>
        </div>

        {/* HEAD TOP-DOWN */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
          <div style={{ fontSize: 7, letterSpacing: 3, color: "#f97316", fontWeight: 700, marginBottom: 4, borderBottom: "1px solid #431407", paddingBottom: 3 }}>
            HEAD TOP-DOWN — FPN P5→P4→P3
          </div>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            {headTD.map((layer, i) => (
              <div key={layer.id} style={{ display: "flex", flexDirection: "column", width: "100%" }}>
                {layer.merge && (
                  <div style={{ fontSize: 6, color: "#22c55e", background: "#052e16", border: "1px solid #166534", borderRadius: 3, padding: "1px 5px", marginBottom: 1, alignSelf: "flex-start" }}>
                    {layer.merge}
                  </div>
                )}
                <div style={{ width: "100%", background: layer.color + "18", border: `1px solid ${layer.color}55`, borderLeft: `3px solid ${layer.color}`, borderRadius: 4, padding: "2px 7px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <span style={{ fontSize: 7, color: "#475569" }}>[{layer.id}]</span>
                    <span style={{ fontSize: 9, fontWeight: 700, color: layer.id === 17 ? "#4ade80" : "#f1f5f9" }}>{layer.label}</span>
                    <span style={{ fontSize: 7, color: "#64748b" }}>{layer.sub}</span>
                  </div>
                  <span style={{ fontSize: 7, color: "#38bdf8", fontFamily: "monospace" }}>{layer.shape}</span>
                </div>
                {i < headTD.length - 1 && <div style={{ fontSize: 9, color: "#431407", textAlign: "center", lineHeight: "9px" }}>↓</div>}
              </div>
            ))}
          </div>
        </div>

        {/* HEAD BOTTOM-UP + OUTPUTS */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
          <div style={{ fontSize: 7, letterSpacing: 3, color: "#22c55e", fontWeight: 700, marginBottom: 4, borderBottom: "1px solid #14532d", paddingBottom: 3 }}>
            HEAD BOTTOM-UP — PAN P3→P4
          </div>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
            {headBU.map((layer, i) => (
              <div key={layer.id} style={{ display: "flex", flexDirection: "column", width: "100%" }}>
                {layer.merge && (
                  <div style={{ fontSize: 6, color: "#22c55e", background: "#052e16", border: "1px solid #166534", borderRadius: 3, padding: "1px 5px", marginBottom: 1, alignSelf: "flex-start" }}>
                    {layer.merge}
                  </div>
                )}
                <div style={{ width: "100%", background: layer.color + "18", border: `1px solid ${layer.color}55`, borderLeft: `3px solid ${layer.color}`, borderRadius: 4, padding: "2px 7px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <span style={{ fontSize: 7, color: "#475569" }}>[{layer.id}]</span>
                    <span style={{ fontSize: 9, fontWeight: 700, color: layer.id === 22 ? "#4ade80" : "#f1f5f9" }}>{layer.label}</span>
                    <span style={{ fontSize: 7, color: "#64748b" }}>{layer.sub}</span>
                  </div>
                  <span style={{ fontSize: 7, color: "#38bdf8", fontFamily: "monospace" }}>{layer.shape}</span>
                </div>
                {i < headBU.length - 1 && <div style={{ fontSize: 9, color: "#14532d", textAlign: "center", lineHeight: "9px" }}>↓</div>}
              </div>
            ))}
          </div>

          {/* Detection outputs */}
          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ fontSize: 7, letterSpacing: 3, color: "#475569", marginBottom: 2 }}>DETECTION OUTPUTS</div>
            <div style={{ background: "#0c4a6e", border: "1.5px solid #06b6d4", borderRadius: 6, padding: "4px 8px" }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: "#7dd3fc" }}>P3 — 96×96 grid · stride-8</div>
              <div style={{ fontSize: 7, color: "#0891b2" }}>Damaged_1 (tiny cracks & burns)</div>
            </div>
            <div style={{ background: "#451a03", border: "1.5px solid #f59e0b", borderRadius: 6, padding: "4px 8px" }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: "#fcd34d" }}>P4 — 48×48 grid · stride-16</div>
              <div style={{ fontSize: 7, color: "#d97706" }}>Insulator (medium objects)</div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer legend */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 6, borderTop: "1px solid #0f172a", paddingTop: 4 }}>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {[["#8b5cf6","GhostConv — 50% param savings"],["#7c3aed","C3Ghost — CSP bottleneck"],["#0ea5e9","SPPF — multi-scale pool"],["#ef4444","DWConv — 9× head reduction"],["#f97316","Concat — FPN/PAN skip"],["#16a34a","★ Detection output"]].map(([c,l]) => (
            <div key={l} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div style={{ width: 7, height: 7, borderRadius: 1, background: c }} />
              <span style={{ fontSize: 6.5, color: "#475569" }}>{l}</span>
            </div>
          ))}
        </div>
        <div style={{ fontSize: 6.5, color: "#334155" }}>P P Satya Karthikeya · B Karthikeya · M Karthik Reddy · P Rohit</div>
      </div>
    </div>
  );
}