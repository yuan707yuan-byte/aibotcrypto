import Head from 'next/head';
import { useState, useEffect, useRef, useCallback } from 'react';
import { ethers } from 'ethers';

// ─── CHAINLINK CONFIG ───────────────────────────────────────────────────────
const CHAINLINK_BTC_USD = '0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88';
const CHAINLINK_ABI = [
  'function latestRoundData() external view returns (uint80,int256,uint256,uint256,uint80)',
  'function decimals() external view returns (uint8)',
];
const RPC_ENDPOINTS = [
  'https://eth.llamarpc.com',
  'https://cloudflare-eth.com',
  'https://rpc.ankr.com/eth',
  'https://ethereum.publicnode.com',
];

// ─── MATH / INDICATORS ──────────────────────────────────────────────────────
function ema(data, period) {
  const k = 2 / (period + 1);
  const result = [];
  let prev = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
  result.push(...Array(period - 1).fill(null), prev);
  for (let i = period; i < data.length; i++) {
    prev = data[i] * k + prev * (1 - k);
    result.push(prev);
  }
  return result;
}

function rsiCalc(closes, period = 14) {
  const gains = [], losses = [];
  for (let i = 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    gains.push(Math.max(diff, 0));
    losses.push(Math.max(-diff, 0));
  }
  const result = Array(period).fill(null);
  let ag = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let al = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
  result.push(al === 0 ? 100 : 100 - 100 / (1 + ag / al));
  for (let i = period; i < gains.length; i++) {
    ag = (ag * (period - 1) + gains[i]) / period;
    al = (al * (period - 1) + losses[i]) / period;
    result.push(al === 0 ? 100 : 100 - 100 / (1 + ag / al));
  }
  return result;
}

function macdCalc(closes) {
  const e12 = ema(closes, 12), e26 = ema(closes, 26);
  const ml = closes.map((_, i) =>
    e12[i] != null && e26[i] != null ? e12[i] - e26[i] : null
  );
  const valid = ml.filter(v => v != null);
  const sr = ema(valid, 9);
  const signal = Array(ml.length - valid.length).fill(null);
  let si = 0;
  ml.forEach(v => { if (v != null) signal.push(sr[si++]); });
  const hist = ml.map((v, i) => v != null && signal[i] != null ? v - signal[i] : null);
  return { ml, signal, hist };
}

function bbCalc(closes, period = 20, mult = 2) {
  return closes.map((_, i) => {
    if (i < period - 1) return null;
    const sl = closes.slice(i - period + 1, i + 1);
    const mean = sl.reduce((a, b) => a + b, 0) / period;
    const std = Math.sqrt(sl.reduce((s, v) => s + (v - mean) ** 2, 0) / period);
    return { upper: mean + mult * std, middle: mean, lower: mean - mult * std, std };
  });
}

function atrCalc(candles, period = 14) {
  const tr = candles.map((c, i) => {
    if (i === 0) return c.high - c.low;
    const p = candles[i - 1];
    return Math.max(c.high - c.low, Math.abs(c.high - p.close), Math.abs(c.low - p.close));
  });
  let v = tr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  const result = [...Array(period - 1).fill(null), v];
  for (let i = period; i < tr.length; i++) {
    v = (v * (period - 1) + tr[i]) / period;
    result.push(v);
  }
  return result;
}

// ─── CNN-LSTM MODEL ──────────────────────────────────────────────────────────
const sig = x => 1 / (1 + Math.exp(-x));
const th = x => Math.tanh(x);

function lstmCell(seq, w) {
  let h = 0, c = 0;
  for (const x of seq) {
    const fg = sig(w.wf * x + w.uf * h + w.bf);
    const ig = sig(w.wi * x + w.ui * h + w.bi);
    const cg = th(w.wc * x + w.uc * h + w.bc);
    const og = sig(w.wo * x + w.uo * h + w.bo);
    c = fg * c + ig * cg;
    h = og * th(c);
  }
  return h;
}

const WEIGHTS = {
  momentum: { wf:.6,uf:.3,bf:.1, wi:.7,ui:.2,bi:0,  wc:.5,uc:.4,bc:.1, wo:.8,uo:.2,bo:0 },
  rsi:      { wf:.4,uf:.5,bf:0,  wi:.6,ui:.3,bi:.1, wc:.7,uc:.2,bc:0,  wo:.5,uo:.4,bo:.1 },
  macd:     { wf:.5,uf:.4,bf:.1, wi:.8,ui:.1,bi:0,  wc:.6,uc:.3,bc:0,  wo:.7,uo:.2,bo:.1 },
  bb:       { wf:.7,uf:.2,bf:0,  wi:.5,ui:.4,bi:.1, wc:.8,uc:.1,bc:0,  wo:.6,uo:.3,bo:.1 },
  volume:   { wf:.3,uf:.6,bf:.1, wi:.7,ui:.2,bi:0,  wc:.4,uc:.5,bc:.1, wo:.8,uo:.1,bo:0 },
};

const TF_CFG = {
  '1m': { win:8,  atrP:7,  rsiP:9,  stackW:[.30,.20,.20,.15,.15], thresh:.015 },
  '3m': { win:10, atrP:10, rsiP:12, stackW:[.28,.19,.21,.16,.16], thresh:.018 },
  '5m': { win:12, atrP:14, rsiP:14, stackW:[.28,.18,.22,.16,.16], thresh:.020 },
};

function runModel(candles, tf, livePrice) {
  const cfg = TF_CFG[tf];
  if (!candles || candles.length < 50) return null;

  const c = [...candles];
  if (livePrice && c.length > 0) {
    c[c.length - 1] = { ...c[c.length - 1], close: livePrice };
  }

  const closes = c.map(x => x.close);
  const volumes = c.map(x => x.volume);
  const n = closes.length;

  const e3 = ema(closes, 3), e8 = ema(closes, 8), e21 = ema(closes, 21);
  const rsiV = rsiCalc(closes, cfg.rsiP);
  const { ml, hist } = macdCalc(closes);
  const bbV = bbCalc(closes, 20);
  const atrV = atrCalc(c, cfg.atrP);
  const volE = ema(volumes, 10);

  const start = n - cfg.win;
  const build = fn => Array.from({ length: cfg.win }, (_, k) => fn(start + k));

  const momentumSeq = build(i => {
    if (!e3[i] || !e8[i] || !e21[i]) return 0;
    return th((e3[i]-e8[i])/(atrV[i]||1)*.5 + (e8[i]-e21[i])/(atrV[i]||1)*.3);
  });
  const rsiSeq  = build(i => th((rsiV[i] != null ? rsiV[i]-50 : 0)/50));
  const macdSeq = build(i => th((hist[i] != null ? hist[i]/(atrV[i]||1) : 0)*10));
  const bbSeq   = build(i => {
    const b = bbV[i];
    if (!b || b.std===0) return 0;
    return th(((closes[i]-b.lower)/(b.upper-b.lower)-.5)*2);
  });
  const volSeq  = build(i => {
    const ve = volE[i]||volumes[i];
    const anom = (volumes[i]-ve)/(ve||1);
    const dir = i>0 ? Math.sign(closes[i]-closes[i-1]) : 0;
    return th(anom*dir);
  });

  const mo  = lstmCell(momentumSeq, WEIGHTS.momentum);
  const rs  = lstmCell(rsiSeq,      WEIGHTS.rsi);
  const mc  = lstmCell(macdSeq,     WEIGHTS.macd);
  const bb2 = lstmCell(bbSeq,       WEIGHTS.bb);
  const vo  = lstmCell(volSeq,      WEIGHTS.volume);

  const features = [mo, rs, mc, bb2, vo];
  const rawScore = features.reduce((s, f, i) => s + f * cfg.stackW[i], 0);

  const curATR = atrV[n-1]||0;
  const atrPct = (curATR/closes[n-1])*100;
  const consensus = features.filter(f => Math.sign(f)===Math.sign(rawScore)).length/features.length;
  const volFactor = Math.min(Math.max(atrPct/0.15, .5), 1.0);
  const confidence = Math.min(Math.max(Math.abs(rawScore)*consensus*volFactor*100, 50), 97);

  const direction = rawScore > cfg.thresh ? 'UP' : rawScore < -cfg.thresh ? 'DOWN' : 'NEUTRAL';
  const expectedMove = Math.abs(rawScore)*curATR*2;

  return {
    direction, confidence, rawScore, expectedMove,
    signals: {
      momentum: { score: mo, label: 'EMA Crossover' },
      rsi:      { score: rs, label: 'RSI Momentum' },
      macd:     { score: mc, label: 'MACD Histogram' },
      bb:       { score: bb2, label: 'Bollinger Band' },
      volume:   { score: vo, label: 'Volume Anomaly' },
    },
    currentRSI: rsiV[n-1],
    currentATR: curATR,
    currentPrice: closes[n-1],
    bbBands: bbV[n-1],
    macdVal: ml[n-1],
    macdHist: hist[n-1],
    ema3: e3[n-1], ema8: e8[n-1], ema21: e21[n-1],
  };
}

// ─── CHAINLINK FETCH ─────────────────────────────────────────────────────────
async function fetchChainlinkPrice() {
  for (const rpcUrl of RPC_ENDPOINTS) {
    try {
      const provider = new ethers.providers.JsonRpcProvider(rpcUrl);
      const feed = new ethers.Contract(CHAINLINK_BTC_USD, CHAINLINK_ABI, provider);
      const rd = await Promise.race([
        feed.latestRoundData(),
        new Promise((_, r) => setTimeout(() => r(new Error('timeout')), 5000)),
      ]);
      return {
        price: parseFloat(ethers.utils.formatUnits(rd[1], 8)),
        updatedAt: rd[3].toNumber() * 1000,
      };
    } catch { continue; }
  }
  return null;
}

// ─── BINANCE KLINES ──────────────────────────────────────────────────────────
async function fetchKlines(interval, limit = 100) {
  const res = await fetch(
    `https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=${interval}&limit=${limit}`
  );
  if (!res.ok) throw new Error('klines error');
  return (await res.json()).map(k => ({
    time: k[0], open: +k[1], high: +k[2], low: +k[3], close: +k[4], volume: +k[5],
  }));
}

// ─── SPARKLINE ───────────────────────────────────────────────────────────────
function Sparkline({ candles, direction }) {
  if (!candles || candles.length < 2) return null;
  const last = candles.slice(-55);
  const prices = last.map(c => c.close);
  const vols   = last.map(c => c.volume);
  const minP = Math.min(...prices), maxP = Math.max(...prices);
  const maxV = Math.max(...vols);
  const rng = maxP - minP || 1;
  const W = 560, H = 100;
  const pts = prices.map((p, i) => {
    const x = (i/(prices.length-1))*W;
    const y = 6 + ((maxP-p)/rng)*(H*.80);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const col = direction==='UP'?'#00ff9d':direction==='DOWN'?'#ff4757':'#f1c40f';
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={col} stopOpacity=".2"/>
          <stop offset="100%" stopColor={col} stopOpacity=".01"/>
        </linearGradient>
        <filter id="gl"><feGaussianBlur stdDeviation="1.2" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      {vols.map((v, i) => {
        const x = (i/(vols.length-1))*W;
        const bh = (v/maxV)*H*.14;
        return <rect key={i} x={x} y={H-bh} width={W/vols.length} height={bh} fill={col} opacity=".09"/>;
      })}
      <polyline points={`0,${H} ${pts} ${W},${H}`} fill="url(#ag)" stroke="none"/>
      <polyline points={pts} fill="none" stroke={col} strokeWidth="1.6" strokeLinejoin="round" filter="url(#gl)"/>
    </svg>
  );
}

// ─── SIGNAL BAR ──────────────────────────────────────────────────────────────
function SignalBar({ label, score }) {
  const dir = score>.025?'UP':score<-.025?'DOWN':'FLAT';
  const col = dir==='UP'?'#00ff9d':dir==='DOWN'?'#ff4757':'#f1c40f';
  const w = Math.min(Math.abs(score)*180, 100);
  return (
    <div style={{ display:'flex', alignItems:'center', gap:9, marginBottom:8 }}>
      <span style={{ width:128, fontSize:10.5, color:'#354d66', flexShrink:0 }}>{label}</span>
      <div style={{ flex:1, height:5, background:'#050b14', borderRadius:3, overflow:'hidden' }}>
        <div style={{ width:`${w}%`, height:'100%', background:col, borderRadius:3,
          transition:'width .7s ease', boxShadow:`0 0 5px ${col}50` }}/>
      </div>
      <span style={{ width:32, fontSize:10.5, color:col, textAlign:'right' }}>{dir}</span>
    </div>
  );
}

// ─── FORECAST CARD ────────────────────────────────────────────────────────────
function ForecastCard({ tf, label, pred, livePrice, isActive, onClick }) {
  const [endTime, setEndTime] = useState('');
  const [remaining, setRemaining] = useState('');

  useEffect(() => {
    const update = () => {
      const now = new Date();
      const mins = parseInt(tf);
      const end = new Date(now.getTime() + mins * 60000);
      setEndTime(end.toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' }));
      const rem = Math.round((end - now) / 1000);
      const m = Math.floor(rem / 60), s = rem % 60;
      setRemaining(`${m}:${String(s).padStart(2,'0')}`);
    };
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, [tf]);

  if (!pred) return (
    <div onClick={onClick} style={{
      background:'rgba(0,10,22,.88)', border:'1px solid #0a1a2e',
      borderRadius:11, padding:'18px 16px', cursor:'pointer',
      display:'flex', flexDirection:'column', alignItems:'center', gap:10, minHeight:170,
      justifyContent:'center',
    }}>
      <div style={{ fontSize:9, color:'#1e3050', letterSpacing:'2px' }}>{label}</div>
      <div style={{ width:18, height:18, border:'2px solid #00ff9d25',
        borderTopColor:'#00ff9d', borderRadius:'50%', animation:'spin 1s linear infinite' }}/>
    </div>
  );

  const col  = pred.direction==='UP'?'#00ff9d':pred.direction==='DOWN'?'#ff4757':'#f1c40f';
  const icon = pred.direction==='UP'?'▲':pred.direction==='DOWN'?'▼':'◆';
  const fmt  = n => n?.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
  const moveSign = pred.direction==='UP'?'+':pred.direction==='DOWN'?'-':'~';

  return (
    <div onClick={onClick} style={{
      background: isActive ? `linear-gradient(135deg,${col}0d,${col}05)` : 'rgba(0,10,22,.88)',
      border:`1px solid ${isActive?col+'40':'#0a1a2e'}`,
      borderRadius:11, padding:'16px 15px', cursor:'pointer',
      transition:'all .25s ease', position:'relative', overflow:'hidden',
      boxShadow: isActive ? `0 0 20px ${col}18` : 'none',
    }}>
      {/* TF badge top-right */}
      <div style={{
        position:'absolute', top:10, right:10,
        background:`${col}18`, border:`1px solid ${col}32`,
        borderRadius:4, padding:'2px 7px',
        fontSize:8.5, color:col, letterSpacing:'1.5px', fontWeight:700,
      }}>{label}</div>

      {/* Countdown */}
      <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:10 }}>
        <div style={{
          width:5, height:5, borderRadius:'50%',
          background:col, animation:'pulse 1.5s infinite',
          boxShadow:`0 0 5px ${col}`,
        }}/>
        <span style={{ fontSize:8.5, color:'#1e3050', letterSpacing:'1.5px' }}>
          ENDS {endTime} · <span style={{ color:col+'aa' }}>{remaining}</span>
        </span>
      </div>

      {/* Direction */}
      <div style={{
        fontFamily:"'Orbitron',sans-serif",
        fontSize:24, fontWeight:900, color:col,
        textShadow:`0 0 18px ${col}60`, marginBottom:9,
      }}>{icon} {pred.direction}</div>

      {/* Confidence */}
      <div style={{ marginBottom:10 }}>
        <div style={{ display:'flex', justifyContent:'space-between', fontSize:8.5, color:'#1e3050', marginBottom:4 }}>
          <span>CONFIDENCE</span>
          <span style={{ color:col, fontWeight:700 }}>{pred.confidence.toFixed(1)}%</span>
        </div>
        <div style={{ height:4, background:'#04080f', borderRadius:2, overflow:'hidden' }}>
          <div style={{
            width:`${pred.confidence}%`, height:'100%',
            background:`linear-gradient(90deg,${col}55,${col})`,
            borderRadius:2, boxShadow:`0 0 5px ${col}`,
            transition:'width .9s ease',
          }}/>
        </div>
      </div>

      {/* Price target */}
      {livePrice && pred.direction!=='NEUTRAL' && (
        <div style={{
          background:'#040810', borderRadius:6, padding:'7px 9px',
          display:'grid', gridTemplateColumns:'1fr 1fr', gap:6,
          border:'1px solid #08121e',
        }}>
          <div>
            <div style={{ fontSize:7.5, color:'#162030', marginBottom:2, letterSpacing:'1px' }}>NOW</div>
            <div style={{ fontSize:11, color:'#5a7080' }}>${fmt(livePrice)}</div>
          </div>
          <div>
            <div style={{ fontSize:7.5, color:'#162030', marginBottom:2, letterSpacing:'1px' }}>±MOVE</div>
            <div style={{ fontSize:11, color:col, fontWeight:700 }}>
              {moveSign}${fmt(pred.expectedMove)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── MAIN ────────────────────────────────────────────────────────────────────
export default function Home() {
  const [livePrice, setLivePrice] = useState(null);
  const [priceChange, setPriceChange] = useState(0);
  const [chainlinkAge, setChainlinkAge] = useState(null);
  const [candles, setCandles] = useState({ '1m':[], '3m':[], '5m':[] });
  const [predictions, setPredictions] = useState({ '1m':null, '3m':null, '5m':null });
  const [activeTF, setActiveTF] = useState('5m');
  const [history, setHistory] = useState([]);
  const [wsStatus, setWsStatus] = useState('Connecting…');
  const [lastTickMs, setLastTickMs] = useState(null);
  const [candlesAt, setCandlesAt] = useState(null);
  const [tickCount, setTickCount] = useState(0);

  const livePriceRef = useRef(null);
  const prevPriceRef = useRef(null);
  const candlesRef   = useRef({ '1m':[], '3m':[], '5m':[] });
  const wsRef        = useRef(null);

  const fmt = n => n?.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});

  // ── Recompute all 3 models
  const recompute = useCallback((candleMap, price) => {
    const p = price || livePriceRef.current;
    const p1 = runModel(candleMap['1m'], '1m', p);
    const p3 = runModel(candleMap['3m'], '3m', p);
    const p5 = runModel(candleMap['5m'], '5m', p);
    setPredictions({ '1m':p1, '3m':p3, '5m':p5 });
    if (p5) {
      setHistory(prev => [{ ...p5, ts:Date.now(), price:p||p5.currentPrice },...prev].slice(0,10));
    }
  }, []);

  // ── Fetch candles + Chainlink once per minute
  const fetchAll = useCallback(async () => {
    const [r1, r3, r5, cl] = await Promise.allSettled([
      fetchKlines('1m', 100),
      fetchKlines('3m', 100),
      fetchKlines('5m', 100),
      fetchChainlinkPrice(),
    ]);
    const newMap = { ...candlesRef.current };
    if (r1.status==='fulfilled') newMap['1m'] = r1.value;
    if (r3.status==='fulfilled') newMap['3m'] = r3.value;
    if (r5.status==='fulfilled') newMap['5m'] = r5.value;
    candlesRef.current = newMap;
    setCandles({ ...newMap });
    setCandlesAt(new Date());

    if (cl.status==='fulfilled' && cl.value) {
      const { price: cp, updatedAt } = cl.value;
      setChainlinkAge(Math.round((Date.now()-updatedAt)/1000));
      // Use Chainlink as authoritative anchor
      if (!livePriceRef.current) {
        livePriceRef.current = cp;
        setLivePrice(cp);
      }
    }
    recompute(newMap, livePriceRef.current);
  }, [recompute]);

  // ── WebSocket: millisecond aggTrade ticks
  useEffect(() => {
    let ws, reconnTimer;
    const connect = () => {
      ws = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@aggTrade');
      wsRef.current = ws;
      ws.onopen  = () => setWsStatus('Live');
      ws.onerror = () => setWsStatus('Reconnecting…');
      ws.onclose = () => {
        setWsStatus('Reconnecting…');
        reconnTimer = setTimeout(connect, 2000);
      };
      ws.onmessage = e => {
        const d = JSON.parse(e.data);
        const p = parseFloat(d.p);
        const now = Date.now();
        if (prevPriceRef.current !== null) setPriceChange(p - prevPriceRef.current);
        prevPriceRef.current = p;
        livePriceRef.current = p;
        setLivePrice(p);
        setLastTickMs(now);
        setTickCount(t => t + 1);
      };
    };
    connect();
    return () => { clearTimeout(reconnTimer); ws?.close(); };
  }, []);

  // ── Re-run models every 5s with live price
  useEffect(() => {
    const id = setInterval(() => {
      if (Object.values(candlesRef.current).every(a => a.length > 0)) {
        recompute(candlesRef.current, livePriceRef.current);
      }
    }, 5000);
    return () => clearInterval(id);
  }, [recompute]);

  // ── Chainlink age counter
  useEffect(() => {
    const id = setInterval(() => setChainlinkAge(a => a!=null?a+1:null), 1000);
    return () => clearInterval(id);
  }, []);

  // ── Initial load + 60s refresh
  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 60000);
    return () => clearInterval(id);
  }, []); // eslint-disable-line

  const activePred = predictions[activeTF];
  const dirCol = activePred?.direction==='UP'?'#00ff9d':activePred?.direction==='DOWN'?'#ff4757':'#f1c40f';
  const msAgo = lastTickMs ? Date.now()-lastTickMs : null;

  return (
    <>
      <Head>
        <title>Quantum BTC AI — 1m · 3m · 5m Predictions</title>
        <meta name="description" content="Millisecond BTC/USD price + 1m/3m/5m CNN-LSTM forecasts via Chainlink oracle"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
      </Head>

      <div style={{
        background:'linear-gradient(160deg,#060a0f 0%,#07101a 60%,#050912 100%)',
        minHeight:'100vh',
        fontFamily:"'JetBrains Mono','Fira Code',monospace",
        color:'#c9d1d9',
      }}>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');
          *{box-sizing:border-box;margin:0;padding:0}
          @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
          @keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}
          @keyframes slideDown{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
          @keyframes scanline{0%{top:-4%}100%{top:104%}}
          @keyframes flash{0%{opacity:1}50%{opacity:.5}100%{opacity:1}}
          .scan{position:absolute;left:0;right:0;height:1px;background:linear-gradient(transparent,rgba(0,255,157,.04),transparent);animation:scanline 5s linear infinite;pointer-events:none}
          .fade{animation:slideDown .35s ease}
          ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:#0d1e30;border-radius:2px}
          @media(max-width:700px){
            .tf-grid{grid-template-columns:1fr!important}
            .sig-grid{grid-template-columns:1fr!important}
            .hero-price{font-size:30px!important}
          }
        `}</style>

        {/* ── HEADER */}
        <header style={{
          borderBottom:'1px solid #07182a', padding:'11px 20px',
          display:'flex', alignItems:'center', justifyContent:'space-between',
          flexWrap:'wrap', gap:10,
          background:'rgba(3,7,14,.8)', backdropFilter:'blur(14px)',
          position:'sticky', top:0, zIndex:100,
        }}>
          <div style={{ display:'flex', alignItems:'center', gap:12 }}>
            <div style={{
              width:7, height:7, borderRadius:'50%',
              background: wsStatus==='Live'?'#00ff9d':'#f1c40f',
              animation:'pulse 1.6s infinite',
              boxShadow:`0 0 8px ${wsStatus==='Live'?'#00ff9d':'#f1c40f'}`,
            }}/>
            <span style={{
              fontFamily:"'Orbitron',sans-serif",
              fontSize:12, letterSpacing:'3.5px', color:'#00ff9d', fontWeight:700,
            }}>QUANTUM BTC AI</span>
            <span style={{ fontSize:9, color:'#0e2a3e', letterSpacing:'2px' }}>v3.1</span>
          </div>
          <div style={{ display:'flex', alignItems:'center', gap:12, fontSize:9.5, color:'#1e3a5a', flexWrap:'wrap' }}>
            <span style={{
              background: wsStatus==='Live'?'#00ff9d12':'#f1c40f12',
              border:`1px solid ${wsStatus==='Live'?'#00ff9d28':'#f1c40f28'}`,
              padding:'2px 9px', borderRadius:4,
              color: wsStatus==='Live'?'#00ff9d':'#f1c40f', fontSize:9.5,
            }}>⚡ WS {wsStatus}{wsStatus==='Live'&&msAgo!==null&&msAgo<2000?` · ${msAgo}ms`:''}</span>
            <span style={{
              background:'#00ff9d0e', border:'1px solid #00ff9d22',
              padding:'2px 8px', borderRadius:4, color:'#00ff9d80', fontSize:9,
            }}>⛓ Chainlink{chainlinkAge!==null?` ${chainlinkAge}s`:''}</span>
            <span style={{ fontSize:9, color:'#0e2240' }}>
              {tickCount>0 && `${tickCount.toLocaleString()} ticks`}
            </span>
          </div>
        </header>

        <main style={{ padding:'16px 18px', maxWidth:980, margin:'0 auto' }}>

          {/* ── PRICE HERO */}
          <div style={{
            background:'linear-gradient(135deg,rgba(0,15,32,.9),rgba(0,7,18,.95))',
            border:'1px solid #091c30', borderRadius:14, padding:'20px 22px',
            marginBottom:14, position:'relative', overflow:'hidden',
          }}>
            <div className="scan"/>
            <div style={{
              display:'flex', justifyContent:'space-between',
              alignItems:'flex-start', flexWrap:'wrap', gap:18, marginBottom:14,
            }}>
              {/* Live price */}
              <div>
                <div style={{ display:'flex', alignItems:'center', gap:9, marginBottom:8 }}>
                  <span style={{ fontSize:9, color:'#1a3550', letterSpacing:'2px' }}>BTC / USD</span>
                  <span style={{
                    fontSize:8.5, background:'#00ff9d10', border:'1px solid #00ff9d25',
                    padding:'1px 7px', borderRadius:3, color:'#00ff9d80', letterSpacing:'1px',
                  }}>⛓ CHAINLINK</span>
                  <span style={{
                    fontSize:8.5, background:'#00b4d810', border:'1px solid #00b4d825',
                    padding:'1px 7px', borderRadius:3, color:'#00b4d870', letterSpacing:'1px',
                  }}>⚡ LIVE TICK</span>
                </div>
                <div style={{ display:'flex', alignItems:'baseline', gap:14, flexWrap:'wrap' }}>
                  <span
                    className="hero-price"
                    style={{
                      fontFamily:"'Orbitron',sans-serif",
                      fontSize:46, fontWeight:900, color:'#fff',
                      letterSpacing:'-1px',
                      textShadow:'0 0 40px rgba(0,255,157,.10)',
                    }}
                  >
                    {livePrice ? `$${fmt(livePrice)}` : '—'}
                  </span>
                  {priceChange!==0 && (
                    <span style={{
                      fontSize:15, fontWeight:700,
                      color: priceChange>=0?'#00ff9d':'#ff4757',
                    }}>
                      {priceChange>=0?'+':''}{fmt(priceChange)}
                    </span>
                  )}
                </div>
                {candlesAt && (
                  <div style={{ marginTop:5, fontSize:9, color:'#162535' }}>
                    candles: {candlesAt.toLocaleTimeString()} · model refresh: 5s
                  </div>
                )}
              </div>

              {/* Active TF prediction badge */}
              {activePred ? (
                <div className="fade" style={{
                  textAlign:'center',
                  background:`linear-gradient(135deg,${dirCol}0e,${dirCol}04)`,
                  border:`1px solid ${dirCol}38`,
                  borderRadius:12, padding:'15px 20px', minWidth:162,
                }}>
                  <div style={{ fontSize:8, color:'#1e3a5a', letterSpacing:'2px', marginBottom:5 }}>
                    ACTIVE · NEXT {activeTF.toUpperCase()}
                  </div>
                  <div style={{
                    fontFamily:"'Orbitron',sans-serif",
                    fontSize:26, fontWeight:900, color:dirCol,
                    textShadow:`0 0 20px ${dirCol}65`, marginBottom:9,
                  }}>
                    {activePred.direction==='UP'?'▲':activePred.direction==='DOWN'?'▼':'◆'} {activePred.direction}
                  </div>
                  <div style={{ fontSize:8, color:'#1e3a5a', letterSpacing:'1.5px', marginBottom:4 }}>CONFIDENCE</div>
                  <div style={{
                    fontFamily:"'Orbitron',sans-serif",
                    fontSize:19, fontWeight:700, color:dirCol, marginBottom:7,
                  }}>{activePred.confidence.toFixed(1)}%</div>
                  <div style={{ height:3, background:'#04080f', borderRadius:2, overflow:'hidden' }}>
                    <div style={{
                      width:`${activePred.confidence}%`, height:'100%',
                      background:`linear-gradient(90deg,${dirCol}55,${dirCol})`,
                      borderRadius:2, boxShadow:`0 0 6px ${dirCol}`,
                      transition:'width .8s ease',
                    }}/>
                  </div>
                </div>
              ) : (
                <div style={{ display:'flex', alignItems:'center', gap:10, color:'#1e3050', fontSize:11 }}>
                  <div style={{ width:16, height:16, border:'2px solid #00ff9d25',
                    borderTopColor:'#00ff9d', borderRadius:'50%', animation:'spin 1s linear infinite' }}/>
                  Computing…
                </div>
              )}
            </div>

            {/* Chart */}
            <div style={{ borderTop:'1px solid #071828', paddingTop:11 }}>
              <Sparkline
                candles={candles[activeTF]}
                direction={activePred?.direction}
              />
            </div>
          </div>

          {/* ── 3 FORECAST CARDS */}
          <div style={{ marginBottom:14 }}>
            <div style={{ fontSize:8.5, color:'#162840', letterSpacing:'2.5px', marginBottom:9 }}>
              ◈ MULTI-TIMEFRAME FORECASTS — click to switch active view
            </div>
            <div className="tf-grid" style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:11 }}>
              {[
                { tf:'1m', label:'NEXT 1 MINUTE' },
                { tf:'3m', label:'NEXT 3 MINUTES' },
                { tf:'5m', label:'NEXT 5 MINUTES' },
              ].map(({ tf, label }) => (
                <ForecastCard
                  key={tf} tf={tf} label={label}
                  pred={predictions[tf]}
                  livePrice={livePrice}
                  isActive={activeTF===tf}
                  onClick={() => setActiveTF(tf)}
                />
              ))}
            </div>
          </div>

          {/* ── SIGNALS + INDICATORS */}
          <div className="sig-grid" style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginBottom:12 }}>
            <div style={{ background:'rgba(0,9,20,.9)', border:'1px solid #091828', borderRadius:11, padding:17 }}>
              <div style={{ fontSize:8.5, color:'#162840', letterSpacing:'2.5px', marginBottom:14 }}>
                ◈ CNN-LSTM STREAMS · {activeTF.toUpperCase()}
              </div>
              {activePred
                ? Object.entries(activePred.signals).map(([k, s]) => (
                    <SignalBar key={k} label={s.label} score={s.score}/>
                  ))
                : <div style={{ color:'#162840', fontSize:11 }}>Loading…</div>}
            </div>

            <div style={{ background:'rgba(0,9,20,.9)', border:'1px solid #091828', borderRadius:11, padding:17 }}>
              <div style={{ fontSize:8.5, color:'#162840', letterSpacing:'2.5px', marginBottom:14 }}>
                ◈ TECHNICAL INDICATORS
              </div>
              {activePred ? (
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:7 }}>
                  {[
                    { l:'RSI', v: activePred.currentRSI?.toFixed(1),
                      c: activePred.currentRSI>70?'#ff4757':activePred.currentRSI<30?'#00ff9d':'#c9d1d9' },
                    { l:'ATR', v: fmt(activePred.currentATR) },
                    { l:'EMA 3', v: fmt(activePred.ema3),
                      c: activePred.ema3>activePred.ema8?'#00ff9d':'#ff4757' },
                    { l:'EMA 8', v: fmt(activePred.ema8),
                      c: activePred.ema8>activePred.ema21?'#00ff9d':'#ff4757' },
                    { l:'EMA 21', v: fmt(activePred.ema21) },
                    { l:'MACD', v: activePred.macdVal?.toFixed(2),
                      c: activePred.macdVal>0?'#00ff9d':'#ff4757' },
                    { l:'MACD HIST', v: activePred.macdHist?.toFixed(2),
                      c: activePred.macdHist>0?'#00ff9d':'#ff4757' },
                    { l:'BB MID', v: fmt(activePred.bbBands?.middle) },
                  ].map(({ l, v, c='#c9d1d9' }) => (
                    <div key={l} style={{ background:'#030710', borderRadius:6, padding:'7px 9px', border:'1px solid #08111e' }}>
                      <div style={{ fontSize:8, color:'#162535', letterSpacing:'1px', marginBottom:2 }}>{l}</div>
                      <div style={{ fontSize:12, color:c, fontWeight:600 }}>{v??'—'}</div>
                    </div>
                  ))}
                </div>
              ) : <div style={{ color:'#162840', fontSize:11 }}>Loading…</div>}
            </div>
          </div>

          {/* ── PREDICTION LOG */}
          <div style={{ background:'rgba(0,9,20,.9)', border:'1px solid #091828', borderRadius:11, padding:17, marginBottom:12 }}>
            <div style={{ fontSize:8.5, color:'#162840', letterSpacing:'2.5px', marginBottom:12 }}>
              ◈ PREDICTION LOG
            </div>
            {history.length===0
              ? <div style={{ color:'#162840', fontSize:11 }}>Awaiting first prediction…</div>
              : (
                <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
                  {history.map((h, i) => {
                    const c = h.direction==='UP'?'#00ff9d':h.direction==='DOWN'?'#ff4757':'#f1c40f';
                    const ic = h.direction==='UP'?'▲':h.direction==='DOWN'?'▼':'◆';
                    return (
                      <div key={i} className={i===0?'fade':''} style={{
                        display:'flex', alignItems:'center', gap:11, flexWrap:'wrap',
                        padding:'6px 10px',
                        background: i===0?`${c}07`:'#030710',
                        borderRadius:6, border:`1px solid ${i===0?c+'20':'#071018'}`,
                        fontSize:11,
                      }}>
                        <span style={{ color:'#1a3050', minWidth:68 }}>
                          {new Date(h.ts).toLocaleTimeString()}
                        </span>
                        <span style={{
                          fontSize:8, color:c,
                          background:`${c}14`, border:`1px solid ${c}28`,
                          padding:'1px 6px', borderRadius:3,
                        }}>+5MIN</span>
                        <span style={{ color:c, fontWeight:700 }}>{ic} {h.direction}</span>
                        <span style={{ color:'#1a3050' }}>
                          Conf: <span style={{ color:c }}>{h.confidence.toFixed(1)}%</span>
                        </span>
                        <span style={{ color:'#1a3050' }}>
                          @ <span style={{ color:'#6a8090' }}>${fmt(h.price)}</span>
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
          </div>

          {/* ── FOOTER */}
          <div style={{
            background:'rgba(0,5,12,.5)', border:'1px solid #060e1a',
            borderRadius:9, padding:'10px 14px',
            fontSize:9, color:'#162535', lineHeight:1.9,
          }}>
            <div>
              <span style={{ color:'#0d2235' }}>⚡ PRICE: </span>
              Binance WebSocket aggTrade stream (millisecond ticks) · Chainlink oracle anchor every 60s · No API keys required
            </div>
            <div>
              <span style={{ color:'#0d2235' }}>◈ MODEL: </span>
              Separate CNN-LSTM ensemble per timeframe (1m/3m/5m) · 5 signal streams · recalculates every 5s · injects live price into latest candle
            </div>
            <div style={{ color:'#091825' }}>
              ⚠ Not financial advice. Cryptocurrency trading involves substantial risk of loss.
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
