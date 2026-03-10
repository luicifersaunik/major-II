import { useState, useEffect } from "react";
import {
  LineChart, Line, BarChart, Bar, Cell, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const MODELS = ["QENN", "ENN-10", "ENN-20", "ENN-40", "ENN-70", "ENN-100"];
const MODEL_COLORS = {                                                                        
  "QENN":    "#00f5d4",
  "ENN-10":  "#f72585",
  "ENN-20":  "#fb8500",
  "ENN-40":  "#8338ec",
  "ENN-70":  "#3a86ff",
  "ENN-100": "#06d6a0"
};
const STOCK_LABELS = {
  BSE:"Bombay SE", NASDAQ:"NASDAQ", HSI:"Hang Seng",
  SSE:"Shanghai", Russell2000:"Russell 2000", TAIEX:"TAIEX"
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{background:"#0d0d1a",border:"1px solid #2a2a4a",borderRadius:8,padding:"10px 14px",fontSize:11}}>
      <p style={{color:"#888",marginBottom:6}}>Day {label}</p>
      {payload.map(p => (
        <p key={p.name} style={{color:p.color,margin:"2px 0"}}>
          {p.name}: <span style={{color:"#fff",fontWeight:600}}>
            {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
          </span>
        </p>
      ))}
    </div>
  );
};

const MetricCard = ({ label, value, unit="" }) => (
  <div style={{background:"#0d0d1a",border:"1px solid #1e1e3a",borderRadius:10,padding:"14px 18px",flex:1,minWidth:120}}>
    <div style={{color:"#555",fontSize:10,letterSpacing:2,textTransform:"uppercase",marginBottom:4}}>{label}</div>
    <div style={{color:"#fff",fontSize:20,fontWeight:700,fontFamily:"monospace"}}>
      {typeof value === "number" ? value.toFixed(4) : value ?? "—"}{unit}
    </div>
  </div>
);

export default function Dashboard() {
  const [DATA, setDATA]               = useState(null);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState(null);
  const [activeStock, setActiveStock] = useState("BSE");
  const [activeTab, setActiveTab]     = useState("overview");
  const [selectedModel, setSelectedModel] = useState("ENN-10");

  // ── Load results.json from public folder ──────────────────────────────
  useEffect(() => {
    fetch("/results.json")
      .then(r => {
        if (!r.ok) throw new Error("results.json not found in public folder");
        return r.json();
      })
      .then(json => {
        // Normalise: support both flat {nmse,preds,...} and nested {models:{...}}
        const normalised = {};
        for (const [stock, val] of Object.entries(json)) {
          if (val.models) {
            normalised[stock] = val.models;
            normalised[stock]._series = val.series;
          } else {
            normalised[stock] = val;
          }
        }
        setDATA(normalised);
        setActiveStock(Object.keys(normalised)[0]);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // ── Loading / error screens ───────────────────────────────────────────
  if (loading) return (
    <div style={{minHeight:"100vh",background:"#060610",display:"flex",alignItems:"center",justifyContent:"center",flexDirection:"column",gap:16}}>
      <div style={{width:40,height:40,border:"3px solid #1e1e3a",borderTop:"3px solid #00f5d4",borderRadius:"50%",animation:"spin 1s linear infinite"}}/>
      <p style={{color:"#444",fontFamily:"monospace"}}>Loading results.json…</p>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );

  if (error) return (
    <div style={{minHeight:"100vh",background:"#060610",display:"flex",alignItems:"center",justifyContent:"center",flexDirection:"column",gap:16,padding:40}}>
      <div style={{fontSize:32}}>⚠️</div>
      <p style={{color:"#f72585",fontFamily:"monospace",fontSize:14}}>{error}</p>
      <div style={{background:"#0a0a1a",border:"1px solid #2a2a4a",borderRadius:12,padding:24,maxWidth:500,fontSize:12,color:"#888",lineHeight:1.8}}>
        <p style={{color:"#fff",marginBottom:12,fontWeight:700}}>Fix: Copy results.json to the public folder</p>
        <p>1. Run <code style={{color:"#00f5d4"}}>python run_experiment_fast.py</code> to generate results.json</p>
        <p>2. Copy <code style={{color:"#00f5d4"}}>results.json</code> into <code style={{color:"#00f5d4"}}>qenn-dashboard/public/</code></p>
        <p>3. Refresh this page</p>
      </div>
    </div>
  );

  const STOCKS       = Object.keys(DATA);
  const stockData    = DATA[activeStock] ?? {};
  const availModels  = MODELS.filter(m => stockData[m]);
  const qenn         = stockData["QENN"];
  const bestENN      = Math.min(...availModels.filter(m => m !== "QENN").map(m => stockData[m]?.nmse ?? 9));

  // prediction chart
  const compareModel = availModels.includes(selectedModel) ? selectedModel : availModels.find(m => m !== "QENN") ?? "ENN-10";
  const predLen      = Math.min(qenn?.preds?.length ?? 0, stockData[compareModel]?.preds?.length ?? 0);
  const predData     = Array.from({length: predLen}, (_, i) => ({
    day:    i + 1,
    Actual: qenn?.actual?.[i] ?? null,
    QENN:   qenn?.preds?.[i]  ?? null,
    [compareModel]: stockData[compareModel]?.preds?.[i] ?? null,
  }));

  // training convergence
  const convLen  = Math.min(...availModels.map(m => stockData[m]?.train_history?.length ?? 0));
  const convData = convLen > 0
    ? Array.from({length: convLen}, (_, i) => ({
        epoch: i,
        ...Object.fromEntries(availModels.map(m => [m, stockData[m].train_history[i]]))
      }))
    : [];

  const tabs = [
    {id:"overview",   label:"📊 Overview"},
    {id:"prediction", label:"📈 Predictions"},
    {id:"training",   label:"⚡ Training"},
    {id:"heatmap",    label:"🗺 Heatmap"},
  ];

  return (
    <div style={{minHeight:"100vh",background:"#060610",fontFamily:"'Inter',sans-serif",color:"#e0e0e0"}}>

      {/* ── Header ── */}
      <div style={{background:"linear-gradient(135deg,#0a0a1f,#0d0d28)",borderBottom:"1px solid #1a1a3a",padding:"20px 32px",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:12}}>
        <div>
          <h1 style={{margin:0,fontSize:24,fontWeight:800,background:"linear-gradient(90deg,#00f5d4,#3a86ff,#8338ec)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
            QENN vs Classical ENN
          </h1>
        </div>
        <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
          {STOCKS.map(s => (
            <button key={s} onClick={() => setActiveStock(s)} style={{
              padding:"6px 14px",borderRadius:20,fontSize:11,fontWeight:600,cursor:"pointer",
              background: activeStock===s ? "#00f5d4" : "transparent",
              color:      activeStock===s ? "#060610" : "#666",
              border:`1px solid ${activeStock===s ? "#00f5d4" : "#2a2a4a"}`,
            }}>{STOCK_LABELS[s] ?? s}</button>
          ))}
        </div>
      </div>

      {/* ── Tabs ── */}
      <div style={{display:"flex",borderBottom:"1px solid #1a1a3a",background:"#080814",padding:"0 32px"}}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)} style={{
            padding:"11px 18px",background:"none",border:"none",cursor:"pointer",fontSize:12,fontWeight:500,
            color: activeTab===t.id ? "#00f5d4" : "#555",
            borderBottom: activeTab===t.id ? "2px solid #00f5d4" : "2px solid transparent",
          }}>{t.label}</button>
        ))}
      </div>

      <div style={{padding:"24px 32px"}}>

        {/* ══ OVERVIEW ══ */}
        {activeTab==="overview" && (
          <div>
            <div style={{display:"flex",gap:12,marginBottom:20,flexWrap:"wrap"}}>
              <MetricCard label="QENN NMSE"  value={qenn?.nmse} />
              <MetricCard label="Best ENN"   value={bestENN} />
              <MetricCard label="QENN RMSE"  value={qenn?.rmse} />
              <MetricCard label="QENN MAPE"  value={qenn?.mape} unit="%" />
            </div>

            <div style={{background:"#0a0a1a",border:"1px solid #1e1e3a",borderRadius:12,padding:"20px",marginBottom:16}}>
              <div style={{marginBottom:12,fontSize:13,fontWeight:700}}>NMSE Comparison — {STOCK_LABELS[activeStock] ?? activeStock}</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={availModels.map(m => ({model:m, nmse:stockData[m]?.nmse??0}))} margin={{top:5,right:10,left:-10,bottom:5}}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2e" vertical={false}/>
                  <XAxis dataKey="model" tick={{fill:"#777",fontSize:11}}/>
                  <YAxis tick={{fill:"#777",fontSize:10}}/>
                  <Tooltip content={<CustomTooltip/>}/>
                  <Bar dataKey="nmse" name="NMSE" radius={[4,4,0,0]}>
                    {availModels.map(m => <Cell key={m} fill={MODEL_COLORS[m] ?? "#888"}/>)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Summary table */}
            <div style={{background:"#0a0a1a",border:"1px solid #1e1e3a",borderRadius:12,padding:"20px",overflowX:"auto"}}>
              <div style={{marginBottom:12,fontSize:13,fontWeight:700}}>All Markets — NMSE Summary</div>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
                <thead>
                  <tr style={{borderBottom:"1px solid #2a2a4a"}}>
                    <th style={{textAlign:"left",padding:"8px 12px",color:"#555"}}>Market</th>
                    {availModels.map(m => (
                      <th key={m} style={{textAlign:"center",padding:"8px 12px",color:MODEL_COLORS[m],fontSize:11}}>{m}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {STOCKS.map((s,si) => {
                    const row  = DATA[s] ?? {};
                    const vals = availModels.map(m => row[m]?.nmse ?? Infinity);
                    const best = Math.min(...vals);
                    return (
                      <tr key={s} style={{borderBottom:"1px solid #141428",background:si%2===0?"transparent":"#0d0d1e"}}>
                        <td style={{padding:"8px 12px",color:"#aaa",fontWeight:600}}>{STOCK_LABELS[s]??s}</td>
                        {availModels.map((m,i) => (
                          <td key={m} style={{textAlign:"center",padding:"8px 12px",
                            color: vals[i]===best ? "#00f5d4":"#666",
                            fontWeight: vals[i]===best ? 700:400,
                            fontFamily:"monospace",fontSize:11}}>
                            {vals[i]===Infinity ? "—" : vals[i].toFixed(4)}
                            {vals[i]===best && <span style={{marginLeft:4,fontSize:9}}>★</span>}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ══ PREDICTIONS ══ */}
        {activeTab==="prediction" && (
          <div>
            <div style={{display:"flex",gap:8,marginBottom:16,alignItems:"center",flexWrap:"wrap"}}>
              <span style={{color:"#555",fontSize:12}}>Compare QENN vs:</span>
              {availModels.filter(m => m!=="QENN").map(m => (
                <button key={m} onClick={() => setSelectedModel(m)} style={{
                  padding:"5px 14px",borderRadius:16,fontSize:11,cursor:"pointer",
                  background: compareModel===m ? MODEL_COLORS[m]+"33":"transparent",
                  color:      compareModel===m ? MODEL_COLORS[m]:"#555",
                  border:`1px solid ${compareModel===m ? MODEL_COLORS[m]:"#2a2a4a"}`,
                }}>{m}</button>
              ))}
            </div>

            <div style={{background:"#0a0a1a",border:"1px solid #1e1e3a",borderRadius:12,padding:"20px",marginBottom:16}}>
              <div style={{marginBottom:12,fontSize:13,fontWeight:700}}>
                Price Prediction — {STOCK_LABELS[activeStock]??activeStock} · Test Set ({predLen} days)
              </div>
              {predLen > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={predData} margin={{top:5,right:20,left:0,bottom:20}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2e"/>
                    <XAxis dataKey="day" tick={{fill:"#666",fontSize:10}} label={{value:"Day",position:"insideBottom",offset:-10,fill:"#444",fontSize:10}}/>
                    <YAxis tick={{fill:"#666",fontSize:10}}/>
                    <Tooltip content={<CustomTooltip/>}/>
                    <Legend wrapperStyle={{fontSize:11}}/>
                    <Line type="monotone" dataKey="Actual"       stroke="#ffffff" strokeWidth={2} dot={false} strokeDasharray="6 3"/>
                    <Line type="monotone" dataKey="QENN"         stroke="#00f5d4" strokeWidth={2} dot={false}/>
                    <Line type="monotone" dataKey={compareModel} stroke={MODEL_COLORS[compareModel]??"#f72585"} strokeWidth={2} dot={false}/>
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{height:300,display:"flex",alignItems:"center",justifyContent:"center",color:"#444"}}>
                  No prediction data available for {activeStock}
                </div>
              )}
            </div>

            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
              {[{k:"nmse",label:"NMSE"},{k:"rmse",label:"RMSE"},{k:"mae",label:"MAE"},{k:"mape",label:"MAPE %"}].map(({k,label}) => (
                <div key={k} style={{background:"#0a0a1a",border:"1px solid #1e1e3a",borderRadius:12,padding:"16px 20px"}}>
                  <div style={{marginBottom:10,fontSize:12,fontWeight:700,color:"#888"}}>{label} by Model</div>
                  <ResponsiveContainer width="100%" height={140}>
                    <BarChart data={availModels.map(m => ({model:m,value:stockData[m]?.[k]??0}))} margin={{top:0,right:0,left:-20,bottom:0}}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2e" vertical={false}/>
                      <XAxis dataKey="model" tick={{fill:"#666",fontSize:9}}/>
                      <YAxis tick={{fill:"#666",fontSize:9}}/>
                      <Tooltip content={<CustomTooltip/>}/>
                      <Bar dataKey="value" name={label} radius={[3,3,0,0]}>
                        {availModels.map(m => <Cell key={m} fill={MODEL_COLORS[m] ?? "#888"}/>)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ══ TRAINING ══ */}
        {activeTab==="training" && (
          <div style={{background:"#0a0a1a",border:"1px solid #1e1e3a",borderRadius:12,padding:"20px"}}>
            <div style={{marginBottom:12,fontSize:13,fontWeight:700}}>Training Convergence — {STOCK_LABELS[activeStock]??activeStock}</div>
            {convData.length > 0 ? (
              <ResponsiveContainer width="100%" height={340}>
                <LineChart data={convData} margin={{top:5,right:20,left:0,bottom:5}}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2e"/>
                  <XAxis dataKey="epoch" tick={{fill:"#666",fontSize:10}} label={{value:"Epoch",position:"insideBottom",fill:"#444",fontSize:10}}/>
                  <YAxis tick={{fill:"#666",fontSize:10}}/>
                  <Tooltip content={<CustomTooltip/>}/>
                  <Legend wrapperStyle={{fontSize:11}}/>
                  {availModels.map(m => (
                    <Line key={m} type="monotone" dataKey={m}
                      stroke={MODEL_COLORS[m]??"#888"}
                      strokeWidth={m==="QENN"?2.5:1.5}
                      dot={false}
                      strokeDasharray={m==="QENN"?"none":"4 2"}/>
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{height:340,display:"flex",alignItems:"center",justifyContent:"center",color:"#444"}}>
                No training history in results.json
              </div>
            )}
          </div>
        )}

        {/* ══ HEATMAP ══ */}
        {activeTab==="heatmap" && (
          <div style={{background:"#0a0a1a",border:"1px solid #1e1e3a",borderRadius:12,padding:"20px",overflowX:"auto"}}>
            <div style={{marginBottom:14,fontSize:13,fontWeight:700}}>NMSE Heatmap — All Markets × All Models</div>
            <table style={{width:"100%",borderCollapse:"separate",borderSpacing:3}}>
              <thead>
                <tr>
                  <th style={{padding:"8px 16px",textAlign:"left",color:"#444",fontSize:11}}>Market</th>
                  {availModels.map(m => (
                    <th key={m} style={{padding:"8px 12px",textAlign:"center",color:MODEL_COLORS[m],fontSize:11}}>{m}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {STOCKS.map(s => {
                  const row  = DATA[s] ?? {};
                  const vals = availModels.map(m => row[m]?.nmse ?? 0);
                  const minV = Math.min(...vals), maxV = Math.max(...vals);
                  return (
                    <tr key={s}>
                      <td style={{padding:"6px 16px",color:"#aaa",fontSize:12,fontWeight:600}}>{STOCK_LABELS[s]??s}</td>
                      {vals.map((v,i) => {
                        const t = maxV===minV ? 0.5 : (v-minV)/(maxV-minV);
                        const r = Math.round(t*200), g = Math.round((1-t)*245), b = Math.round((1-t)*200);
                        return (
                          <td key={i} style={{padding:"8px 12px",textAlign:"center",
                            background:`rgba(${r},${g},${b},0.15)`,borderRadius:6,
                            color:`rgb(${r},${g},${b})`,fontWeight:700,fontFamily:"monospace",fontSize:11}}>
                            {v.toFixed(4)}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{marginTop:12,fontSize:11,color:"#444"}}>🟢 Green = lower NMSE (better) · 🔴 Red = higher NMSE (worse)</div>
          </div>
        )}

      </div>

      <div style={{borderTop:"1px solid #141428",padding:"12px 32px"}} />
    </div>
  );
}

