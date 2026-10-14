/* Prompt2GPU V1.1 — recorded T4 evidence replay, never simulated or live. */
"use strict";
const COLORS = {initial:"#ffbd73",repeat:"#58d2c8"};
const METRICS = [
  "output_tokens_per_sec","running_requests","waiting_requests",
  "kv_cache_usage_percent","p95_ttft_seconds","p95_tpot_seconds",
  "p95_queue_seconds","p95_prefill_seconds","prefix_cache_hit_rate_percent"
];
const CONFIG = {displayModeBar:false,responsive:true,scrollZoom:false};
const STATE = {run:"initial",initial:null,repeat:null,index:0,timer:null};
const $ = id => document.getElementById(id);
const numeric = (value,digits=1) => Number(value).toLocaleString("en-US",{minimumFractionDigits:digits,maximumFractionDigits:digits});
const choice = () => STATE.run === "compare" ? ["initial","repeat"] : [STATE.run];
const span = name => STATE[name];
const relative = (run,unix) => +(unix-run.benchmark_start_unix_s).toFixed(3);
const chartLayout = (name,extra={}) => Object.assign({
  paper_bgcolor:"#111b2b",plot_bgcolor:"#111b2b",font:{color:"#91a5ba",size:10},
  margin:{l:47,r:15,t:18,b:34},height:name==="batch"?238:200,
  showlegend:false,hovermode:"closest",xaxis:{gridcolor:"#26374c",zerolinecolor:"#526276",tickfont:{size:10}},
  yaxis:{gridcolor:"#26374c",zerolinecolor:"#2b3a50",rangemode:"tozero",tickfont:{size:10}},
  uirevision:"preserve"
},extra);
function timelines() {
  const active=choice();
  const all=active.flatMap(key=>span(key).metrics.running_requests.samples.map(s=>relative(span(key),s[0])));
  // Unique timestamps for a logical cursor, never interpolate sensor values.
  return [...new Set(all.map(t=>t.toFixed(3)))].map(Number).sort((a,b)=>a-b);
}
function cursorPosition(){
  const t=timelines();
  return t.length?t[Math.min(STATE.index,t.length-1)]:null;
}
function status(message,isError=false){
  $("message").hidden=false;
  $("message").textContent=message;
  $("message").classList.toggle("error",isError);
}
function select(){
  clearInterval(STATE.timer);STATE.timer=null;
  $("play").textContent="▶ Play";
  STATE.run=$("run-choice").value;
  STATE.index=0;
  rebuild();
}
function updateCursorLabel(){
  const cursor=cursorPosition();
  $("time-indicator").textContent=cursor===null?"No samples":`${cursor>=0?"+":""}${cursor.toFixed(1)} s`;
}
function syncCursor(){
  const items=timelines();
  $("time-cursor").max=Math.max(0,items.length-1);
  $("time-cursor").value=STATE.index;
  $("time-cursor").disabled=items.length<2;
  $("play").disabled=items.length<2;
  updateCursorLabel();
}
function summaryKpi(field,formatter){
  const keys=choice();
  return keys.map(key=>formatter(span(key).summary[field])).join(" / ");
}
function renderKpis(){
  $("kpi-throughput").textContent=summaryKpi("mean_aggregate_tokens_per_sec",v=>numeric(v,1));
  $("kpi-ttft").textContent=summaryKpi("p95_ttft_ms",v=>numeric(v,1)+" ms");
  $("kpi-latency").textContent=summaryKpi("p95_request_latency_sec",v=>numeric(v,3)+" s");
  $("kpi-tpot").textContent=summaryKpi("p95_tpot_ms",v=>numeric(v,2)+" ms");
  for(const card of document.querySelectorAll(".kpicard")) card.classList.toggle("comparison",STATE.run==="compare");
}
function drawBatch(id,field,suffix){
  const series=choice().map(key=>({type:"bar",name:key==="initial"?"Initial":"Warm repeat",
    x:span(key).batches.map(b=>`Batch ${b.batch}`),
    y:span(key).batches.map(b=>b[field]),
    marker:{color:COLORS[key]},
    hovertemplate:`%{x}<br>%{y:.2f} ${suffix}<extra>${key}</extra>`
  }));
  Plotly.react(id,series,chartLayout("batch",{
    barmode:"group",showlegend:STATE.run==="compare",
    legend:{orientation:"h",y:1.15,x:0,font:{color:"#c5d2e0"}},
    xaxis:{gridcolor:"#26374c",tickfont:{size:11}},
    yaxis:{gridcolor:"#26374c",rangemode:"tozero",tickfont:{size:10}}
  }),CONFIG);
}
function shapesForRun(key){
  const run=span(key);
  const shapes=[{type:"line",xref:"x",yref:"paper",x0:0,x1:0,y0:0,y1:1,
    line:{color:"#57667b",width:1,dash:"dot"}}];
  // JIT warning was logged to the nearest second, not measured at sub-second precision.
  if(key==="initial"){
    const t=1791867725-run.benchmark_start_unix_s;
    if(t>=-10 && t<=90) shapes.push({type:"rect",xref:"x",yref:"paper",
      x0:t,x1:t+1,y0:0,y1:1,line:{width:0},fillcolor:"rgba(255,178,86,0.09)"});
  }
  return shapes;
}
function drawMetric(name){
  const cursor=cursorPosition();
  const traces=[];
  for(const key of choice()){
    const run=span(key);
    const rows=run.metrics[name].samples;
    const xs=rows.map(s=>relative(run,s[0]));
    const ys=rows.map(s=>s[1]);
    traces.push({type:"scatter",mode:"lines+markers",name:key==="initial"?"Initial":"Warm repeat",
      x:xs,y:ys,connectgaps:false,
      line:{color:COLORS[key],width:2},marker:{size:5,color:COLORS[key]},
      customdata:rows.map(s=>new Date(s[0]*1000).toISOString()),
      hovertemplate:"Elapsed %{x:.1f}s<br>%{y:.3f}<br>UTC %{customdata}<extra>%{fullData.name}</extra>"
    });
    // A cursor can only highlight a genuine recorded scrape, never an interpolated value.
    if(cursor!==null){
      const index=xs.findIndex((x,i)=>Math.abs(x-cursor)<0.002 && ys[i]!==null);
      if(index>=0) traces.push({type:"scatter",mode:"markers",showlegend:false,
        x:[xs[index]],y:[ys[index]],marker:{size:11,color:COLORS[key],line:{color:"#fff",width:2}},
        hoverinfo:"skip"});
    }
  }
  const xRange=choice().flatMap(key=>span(key).metrics[name].samples.map(s=>relative(span(key),s[0])));
  const min=Math.min(...xRange),max=Math.max(...xRange);
  const shapes=shapesForRun(STATE.run==="repeat"?"repeat":"initial");
  if(cursor!==null) shapes.push({type:"line",xref:"x",yref:"paper",x0:cursor,x1:cursor,y0:0,y1:1,
    line:{color:"#9eb1c8",width:1,dash:"dash"}});
  Plotly.react("metric-"+name,traces,chartLayout("series",{
    showlegend:STATE.run==="compare",legend:{orientation:"h",x:0,y:1.24,font:{color:"#c5d2e0"}},
    xaxis:{title:{text:"Seconds from benchmark start",font:{size:10}},gridcolor:"#26374c",range:[min,max],zerolinecolor:"#6a778b"},
    shapes
  }),CONFIG);
}
function drawDecodeTimeline(){
  const cursor=cursorPosition();
  const traces=[];
  const labels=[];
  for(const key of choice()){
    const run=span(key);
    const points=run.batches;
    // Only five actual measured batch-end timestamps per run.
    const xs=points.map(b=>relative(run,b.end_unix_s));
    const ys=points.map(b=>b.p95_decode_phase_client_proxy_ms);
    traces.push({type:"scatter",mode:"lines+markers",connectgaps:false,
      name:key==="initial"?"Initial":"Warm repeat",x:xs,y:ys,
      customdata:points.map(b=>b.batch),
      line:{color:COLORS[key],width:2},marker:{size:8,color:COLORS[key]},
      hovertemplate:"Batch %{customdata} completed +%{x:.1f}s<br>Decode phase P95 %{y:.1f} ms<extra>%{fullData.name}</extra>"});
    // Highlight the latest completed batch, never invent intermediate samples.
    let latest=-1;
    if(cursor!==null) for(let i=0;i<xs.length;i++) if(xs[i]<=cursor+0.002) latest=i;
    if(latest>=0){
      traces.push({type:"scatter",mode:"markers",showlegend:false,
        x:[xs[latest]],y:[ys[latest]],
        marker:{size:15,color:COLORS[key],line:{color:"#fff",width:2}},
        hoverinfo:"skip"});
      labels.push(`${key==="initial"?"Initial":"Warm repeat"}: batch ${points[latest].batch} · ${numeric(ys[latest],1)} ms`);
    }else{
      labels.push(`${key==="initial"?"Initial":"Warm repeat"}: awaiting first completed batch`);
    }
  }
  $("decode-replay-status").textContent=labels.join("  |  ");
  const times=choice().flatMap(key=>span(key).metrics.running_requests.samples
    .map(s=>relative(span(key),s[0])));
  const endTimes=choice().flatMap(key=>span(key).batches
    .map(b=>relative(span(key),b.end_unix_s)));
  const xMin=Math.min(...times,...endTimes);
  const xMax=Math.max(...times,...endTimes);
  const shapes=shapesForRun(STATE.run==="repeat"?"repeat":"initial");
  if(cursor!==null) shapes.push({type:"line",xref:"x",yref:"paper",
    x0:cursor,x1:cursor,y0:0,y1:1,
    line:{color:"#9eb1c8",width:1,dash:"dash"}});
  Plotly.react("metric-decode-proxy",traces,chartLayout("series",{
    showlegend:STATE.run==="compare",
    legend:{orientation:"h",x:0,y:1.24,font:{color:"#c5d2e0"}},
    xaxis:{title:{text:"Seconds from benchmark start",font:{size:10}},
      gridcolor:"#26374c",range:[xMin,xMax],zerolinecolor:"#6a778b"},
    yaxis:{title:{text:"Decode phase P95 · ms",font:{size:10}},
      gridcolor:"#26374c",rangemode:"tozero"},
    shapes
  }),CONFIG);
}
function renderAll(){
  renderKpis();drawBatch("batch-throughput","throughput_tok_s","tok/s");
  drawBatch("batch-ttft","mean_ttft_ms","ms");
  drawBatch("batch-decode-proxy","p95_decode_phase_client_proxy_ms","ms");
  for(const name of METRICS)drawMetric(name);
  drawDecodeTimeline();
  $("jit-note").hidden=STATE.run==="repeat";
}
function rebuild(){syncCursor();renderAll()}
function onPlay(){
  if(STATE.timer){clearInterval(STATE.timer);STATE.timer=null;$("play").textContent="▶ Play";return;}
  const samples=timelines();
  if(STATE.index>=samples.length-1) STATE.index=0;
  $("play").textContent="❚❚ Pause";
  STATE.timer=setInterval(()=>{
    if(STATE.index>=samples.length-1){clearInterval(STATE.timer);STATE.timer=null;$("play").textContent="▶ Play";return;}
    STATE.index+=1;syncCursor();for(const name of METRICS)drawMetric(name);drawDecodeTimeline();
  },650);
}
async function main(){
  try{
    const pair=await Promise.all(["initial","repeat"].map(async name=>{
      const response=await fetch(`v1_1_data/${name}.json`,{cache:"no-store"});
      if(!response.ok)throw Error(`${name} data HTTP ${response.status}`);
      const data=await response.json();
      if(data.classification!=="RECORDED_MEASUREMENT_NOT_LIVE"||data.label!==name||
         data.batches.length!==5||!data.metrics||METRICS.some(k=>!data.metrics[k]))
        throw Error(`Invalid published ${name} evidence`);
      return data;
    }));
    [STATE.initial,STATE.repeat]=pair;
    if(typeof Plotly==="undefined")throw Error("Plotly.js did not load. This page needs access to the pinned Plotly CDN; no GPU or live server is required.");
    $("run-choice").addEventListener("change",select);
    $("time-cursor").addEventListener("input",event=>{
      clearInterval(STATE.timer);STATE.timer=null;$("play").textContent="▶ Play";
      STATE.index=Number(event.target.value);syncCursor();for(const name of METRICS)drawMetric(name);drawDecodeTimeline();
    });
    $("play").addEventListener("click",onPlay);
    $("dashboard").hidden=false;$("message").hidden=true;rebuild();
  }catch(error){status("Unable to display measured replay: "+error.message+". Use make serve (not file://), verify docs/v1_1_data files and network access to Plotly CDN.",true);}
}
window.addEventListener("DOMContentLoaded",main);
