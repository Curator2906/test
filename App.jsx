import { useState, useCallback } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceDot,
} from "recharts";
import {
  Upload, TrendingUp, Layers, Trash2,
  CheckCircle, Loader2, ChevronRight, BarChart2
} from "lucide-react";

/* ═══════════════════════════════════════════════════════
   THEME — Bloomberg Quant Desk
═══════════════════════════════════════════════════════ */

const T = {
  bg: '#06080b',
  surface: '#0c1118',
  card: '#131f2b',
  border: '#1a2840',
  gold: '#e8a020',
  green: '#00c878',
  red: '#e03050',
  blue: '#2888d8',
  cyan: '#00d8d0',
  purple: '#8860e8',
  muted: '#506070',
  dim: '#1a2840',
  text: '#b8c8d8',
  textHi: '#e8f0f8',
};

const S = {
  card: { background: '#131f2b', border: '1px solid #1a2840', borderRadius: 10, padding: '18px 22px' },
  btn: (color = '#e8a020', solid = false) => ({
    background: solid ? color : 'transparent',
    border: `1px solid ${color}`,
    color: solid ? '#06080b' : color,
    borderRadius: 6, padding: '8px 18px', cursor: 'pointer',
    fontSize: 12, fontFamily: 'inherit', letterSpacing: '0.06em',
    fontWeight: 700, transition: 'all 0.15s',
    display: 'inline-flex', alignItems: 'center', gap: 7,
  }),
  input: {
    background: '#0c1118', border: '1px solid #1a2840', borderRadius: 6,
    color: '#b8c8d8', padding: '8px 12px', fontSize: 12, fontFamily: 'inherit',
    outline: 'none', width: '100%',
  },
  label: { fontSize: 9, color: '#506070', letterSpacing: '0.1em', fontWeight: 700, textTransform: 'uppercase' },
  mono: { fontFamily: "'IBM Plex Mono', 'Courier New', monospace" },
  tag: (color) => ({
    display: 'inline-flex', alignItems: 'center',
    background: `${color}20`, border: `1px solid ${color}50`,
    color: color, borderRadius: 4, padding: '2px 8px',
    fontSize: 9, fontWeight: 700, letterSpacing: '0.08em',
  }),
};

const RISK_FREE = 0.065;
const N_SIMS = 20000;
const PALETTE = [
  T.gold, T.green, T.blue,
  "#00d8d0", "#8860e8", "#e86820",
  "#a0e080", "#e080c0", "#80c0e0"
];

/* ═══════════════════════════════════════════════════════
   MATH ENGINE
═══════════════════════════════════════════════════════ */

const meanArr = a => a.reduce((s, v) => s + v, 0) / a.length;

const logRets = prices =>
  prices.slice(1).map((p, i) =>
    prices[i] > 0 ? Math.log(p / prices[i]) : 0
  );

function covMatrix(retArrays) {
  const n = retArrays.length;
  const mat = Array.from({ length: n }, () => Array(n).fill(0));
  const means = retArrays.map(meanArr);

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const len = Math.min(
        retArrays[i].length,
        retArrays[j].length
      );

      let cov = 0;
      for (let k = 0; k < len; k++) {
        cov +=
          (retArrays[i][retArrays[i].length - len + k] - means[i]) *
          (retArrays[j][retArrays[j].length - len + k] - means[j]);
      }

      mat[i][j] = mat[j][i] = cov / (len - 1);
    }
  }

  return mat;
}

function runMonteCarlo(annRets, annCov, maxWeight = null) {
  const k = annRets.length;
  const results = [];

  for (let s = 0; s < N_SIMS; s++) {

    const raw = Array.from(
      { length: k },
      () => -Math.log(Math.random() + 1e-10)
    );

    const sum = raw.reduce((a, b) => a + b, 0);
    const w = raw.map(x => x / sum);

    if (maxWeight !== null && Math.max(...w) > maxWeight)
      continue;

    let ret = 0;
    let vol2 = 0;

    for (let i = 0; i < k; i++)
      ret += w[i] * annRets[i];

    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++)
        vol2 += w[i] * w[j] * annCov[i][j];

    const vol = Math.sqrt(Math.max(vol2, 1e-10));

    results.push({
      w,
      ret,
      vol,
      sharpe: (ret - RISK_FREE) / vol
    });
  }

  return results;
}

function buildFrontier(portfolios) {
  const sorted = [...portfolios].sort((a, b) => a.ret - b.ret);
  const minR = sorted[0].ret;
  const maxR = sorted[sorted.length - 1].ret;
  const step = (maxR - minR) / 40;

  const frontier = [];

  for (let i = 0; i < 40; i++) {
    const lo = minR + i * step;
    const hi = lo + step;

    const bin = sorted.filter(
      p => p.ret >= lo && p.ret < hi
    );

    if (bin.length) {
      frontier.push(
        bin.reduce((b, p) =>
          p.vol < b.vol ? p : b
        )
      );
    }
  }

  return frontier;
}

function computeStats(prices, periods = 252) {
  if (!prices || prices.length < 2)
    throw new Error("Need at least 2 price points");

  const rets = logRets(prices);
  if (rets.length < 1)
    throw new Error("Could not compute returns");

  const meanRet   = meanArr(rets);
  const variance  = rets.reduce((a, r) => a + (r - meanRet) ** 2, 0) / Math.max(rets.length - 1, 1);

  return {
    annReturn : meanRet * periods,
    annVol    : Math.sqrt(variance * periods),
    daily     : rets,   // kept as "daily" for naming compatibility
    periods,
  };
}
/* ═══════════════════════════════════════════════════════
   CSV PARSER
═══════════════════════════════════════════════════════ */

function splitCSVLine(line) {
  // Handles quoted fields like "1,234.56" correctly
  const result = [];
  let cur = '', inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') { inQuotes = !inQuotes; }
    else if (ch === ',' && !inQuotes) { result.push(cur.trim()); cur = ''; }
    else { cur += ch; }
  }
  result.push(cur.trim());
  return result;
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 6) throw new Error("Need at least 6 data points");

  const headerCols = splitCSVLine(lines[0]).map(h => h.toLowerCase().trim().replace(/"/g, ''));

  // Try multiple possible column names used by NSE, Moneycontrol, Screener, brokers
  const PRICE_NAMES = ['close', 'closing price', 'close price', 'last price', 'ltp', 'price', 'adj close', 'adjclose'];
  const DATE_NAMES  = ['date', 'timestamp', 'time', 'datetime', 'traded date'];

  let priceIndex = headerCols.findIndex(h => PRICE_NAMES.some(n => h.includes(n)));
  let dateIndex  = headerCols.findIndex(h => DATE_NAMES.some(n => h.includes(n)));

  // Last fallback: assume col 0 = date, col 4 = close (standard OHLCV)
  if (priceIndex < 0 && headerCols.length >= 5) priceIndex = 4;
  if (dateIndex  < 0) dateIndex = 0;

  if (priceIndex < 0) throw new Error(`No price column found. Headers: ${headerCols.join(' | ')}`);

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = splitCSVLine(lines[i]);
    // Strip quotes, currency symbols, commas from price
    const rawPrice = (cols[priceIndex] || '').replace(/[",₹$\s]/g, '');
    const price = parseFloat(rawPrice);
    const date  = (cols[dateIndex] || '').replace(/"/g, '').trim();
    if (!isNaN(price) && price > 0 && date) {
      rows.push({ date, price });
    }
  }

  if (rows.length < 6) throw new Error(`Only ${rows.length} valid rows found after parsing`);

  rows.sort((a, b) => new Date(a.date) - new Date(b.date));
  return rows.map(r => r.price);
}
export default function App() {
  /* ═══════════════════════════════════════════════════════
     CORE STATE
  ═══════════════════════════════════════════════════════ */

  const [tab, setTab] = useState("upload");

  // Unlimited uploaded stocks
  const [stocks, setStocks] = useState({});

  // Asset classes for Level 1
  const [assetClasses, setAssetClasses] = useState([
    { id: "equity", label: "Direct Equity Basket", isBasket: true,  color: T.gold,   enabled: true },
    { id: "gold",   label: "Gold ETF",              isBasket: false, color: T.green,  enabled: true },
    { id: "debt",   label: "Debt Fund",             isBasket: false, color: T.blue,   enabled: true },
  ]);

  const PALETTE_ASSETS = [T.green, T.blue, '#00d8d0', '#8860e8', '#e86820', '#a0e080', '#e080c0', '#80c0e0'];

  const addAssetClass = () => {
    const id = `asset_${Date.now()}`;
    const colorIdx = assetClasses.filter(a => !a.isBasket).length % PALETTE_ASSETS.length;
    setAssetClasses(prev => [...prev, {
      id, label: 'New Asset', isBasket: false,
      color: PALETTE_ASSETS[colorIdx], enabled: true
    }]);
  };

  const removeAssetClass = (id) => {
    setAssetClasses(prev => prev.filter(a => a.id !== id));
    setAssetPrices(prev => { const copy = { ...prev }; delete copy[id]; return copy; });
  };

  const renameAssetClass = (id, label) => {
    setAssetClasses(prev => prev.map(a => a.id === id ? { ...a, label } : a));
  };

  const [assetPrices, setAssetPrices] = useState({});

  const [l2Result, setL2Result] = useState(null);
  const [l1Result, setL1Result] = useState(null);

  const [l2Running, setL2Running] = useState(false);
  const [l1Running, setL1Running] = useState(false);

  // Switchable L2 Objective
  const [l2Mode, setL2Mode] = useState("maxSharpe");
  const [dataFreq, setDataFreq] = useState("daily"); // "daily" | "monthly"


  /* ═══════════════════════════════════════════════════════
     STOCK UPLOAD (MULTI + REPLACE)
  ═══════════════════════════════════════════════════════ */

  const handleStockUpload = async (files) => {

    const fileArray = Array.from(files);
    const periods = dataFreq === "monthly" ? 12 : 252;

    for (const file of fileArray) {

      try {

        const text = await file.text();
        const prices = parseCSV(text);
        const stats = computeStats(prices, periods);

        const ticker = file.name
          .replace(".csv", "")
          .toUpperCase();

        setStocks(prev => ({
          ...prev,
          [ticker]: {
            prices,
            stats,
            enabled: true
          }
        }));

      } catch (err) {
        console.error("Upload error:", err.message);
      }
    }
  };


  const removeStock = (ticker) => {
    setStocks(prev => {
      const copy = { ...prev };
      delete copy[ticker];
      return copy;
    });
  };

  const toggleStock = (ticker) => {
    setStocks(prev => ({
      ...prev,
      [ticker]: {
        ...prev[ticker],
        enabled: !prev[ticker].enabled
      }
    }));
  };


  /* ═══════════════════════════════════════════════════════
     ASSET UPLOAD (LEVEL 1)
  ═══════════════════════════════════════════════════════ */

  const handleAssetUpload = async (assetId, file) => {

    const periods = dataFreq === "monthly" ? 12 : 252;

    try {

      const text = await file.text();
      const prices = parseCSV(text);
      const stats  = computeStats(prices, periods);

      setAssetPrices(prev => ({
        ...prev,
        [assetId]: { prices, stats }
      }));

    } catch (err) {
      console.error("Asset upload error:", err.message);
    }
  };

  const toggleAsset = (id) => {
    setAssetClasses(prev =>
      prev.map(a =>
        a.id === id ? { ...a, enabled: !a.enabled } : a
      )
    );
  };


  /* ═══════════════════════════════════════════════════════
     LEVEL 2 OPTIMISATION
  ═══════════════════════════════════════════════════════ */

  const runLevel2 = useCallback(() => {

    setL2Running(true);

    setTimeout(() => {

      try {

        const selected = Object.entries(stocks)
          .filter(([_, s]) => s.enabled)
          .map(([ticker, s]) => ({
            ticker,
            stats: s.stats
          }));

        if (selected.length < 2) return;

        const dailyRets = selected.map(s => s.stats.daily);
        const annRets = selected.map(s => s.stats.annReturn);

        const periods = selected[0].stats.periods ?? (dataFreq === "monthly" ? 12 : 252);
        const cov     = covMatrix(dailyRets);
        const annCov  = cov.map(r => r.map(v => v * periods));

        const portfolios = runMonteCarlo(annRets, annCov, 0.4);

        if (!portfolios.length) return;

        const maxSharpe = portfolios.reduce(
          (b, p) => p.sharpe > b.sharpe ? p : b,
          portfolios[0]
        );

        const minVol = portfolios.reduce(
          (b, p) => p.vol < b.vol ? p : b,
          portfolios[0]
        );

        // Aggressive (95th percentile by return)
        const sortedByReturn = [...portfolios]
          .sort((a, b) => a.ret - b.ret);

        const aggressive =
          sortedByReturn[
            Math.floor(sortedByReturn.length * 0.95)
          ];

        const chosen =
          l2Mode === "aggressive"
            ? aggressive
            : maxSharpe;

        // Build basket daily returns
        const basketDaily = dailyRets[0].map((_, i) =>
          dailyRets.reduce((sum, assetRets, j) =>
            sum + assetRets[i] * chosen.w[j], 0)
        );

        setL2Result({
          portfolios,
          maxSharpe,
          aggressive,
          minVol,
          selected: chosen,
          frontier: buildFrontier(portfolios),
          labels: selected.map(s => s.ticker),
          colors: selected.map((_, i) => PALETTE[i % PALETTE.length]),
          daily: basketDaily
        });

      } finally {
        setL2Running(false);
      }

    }, 60);

  }, [stocks, l2Mode]);


  /* ═══════════════════════════════════════════════════════
     LEVEL 1 OPTIMISATION (AGGRESSIVE BIAS)
  ═══════════════════════════════════════════════════════ */

  const runLevel1 = useCallback(() => {

    setL1Running(true);

    setTimeout(() => {

      try {

        const assets = assetClasses.filter(a => a.enabled);

        const assetData = assets.map(a => {

          if (a.isBasket) {
            if (!l2Result) return null;
            return {
              annReturn: l2Result.selected.ret,
              daily: l2Result.daily
            };
          }

          const entry = assetPrices[a.id];
          if (!entry) return null;

          // support both old format (raw array) and new format ({ prices, stats })
          const stats = entry.stats ?? computeStats(entry, dataFreq === "monthly" ? 12 : 252);

          return {
            annReturn : stats.annReturn,
            daily     : stats.daily,
            periods   : stats.periods,
          };
        });

        const valid = assetData.filter(d => d !== null);
        if (valid.length < 2) return;

       const annRets   = valid.map(d => d.annReturn);
        const dailyRets = valid.map(d => d.daily);
        const periods   = valid[0].periods ?? (dataFreq === "monthly" ? 12 : 252);

        const cov    = covMatrix(dailyRets);
        const annCov = cov.map(r => r.map(v => v * periods));

        const portfolios = runMonteCarlo(annRets, annCov);

        const sortedByReturn = [...portfolios]
          .sort((a, b) => a.ret - b.ret);

        const aggressive =
          sortedByReturn[
            Math.floor(sortedByReturn.length * 0.95)
          ];

        const minVol = portfolios.reduce(
          (b, p) => p.vol < b.vol ? p : b,
          portfolios[0]
        );

        setL1Result({
          portfolios,
          selected: aggressive,
          minVol,
          frontier: buildFrontier(portfolios),
          labels: assets.map(a => a.label),
          colors: assets.map(a => a.color)
        });

      } finally {
        setL1Running(false);
      }

    }, 60);

  }, [assetClasses, assetPrices, l2Result]);

  /* ═══════════════════════════════════════════════════════
     UI HELPERS
  ═══════════════════════════════════════════════════════ */

  const pct = v => `${(v * 100).toFixed(2)}%`;

  const AllocationBar = ({ weights, labels, colors }) => (
    <div>
      <div style={{ display: 'flex', height: 32, borderRadius: 5, overflow: 'hidden', gap: 1 }}>
        {weights.map((w, i) => (
          <div key={i} style={{ width: `${w * 100}%`, background: colors[i], display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {w > 0.07 && <span style={{ fontSize: 9, fontWeight: 700, color: '#000', fontFamily: "'IBM Plex Mono', monospace" }}>{pct(w)}</span>}
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 16px', marginTop: 10 }}>
        {weights.map((w, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: colors[i] }} />
            <span style={{ fontSize: 10, color: T.muted }}>{labels[i]}</span>
            <span style={{ fontSize: 10, color: T.text, fontFamily: "'IBM Plex Mono', monospace" }}>{pct(w)}</span>
          </div>
        ))}
      </div>
    </div>
  );

  const FrontierChart = ({ cloud, frontier, point, title }) => {
    const sample = cloud.filter((_, i) => i % 6 === 0);
    const fmt = v => `${(v * 100).toFixed(1)}%`;
    return (
      <div>
        {title && <div style={{ ...S.label, marginBottom: 12 }}>{title}</div>}
        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart margin={{ top: 10, right: 10, bottom: 24, left: 10 }}>
            <CartesianGrid stroke={T.border} strokeDasharray="3 3" />
            <XAxis dataKey="vol" name="Volatility" tickFormatter={fmt}
              tick={{ fill: T.muted, fontSize: 10 }}
              label={{ value: 'Volatility', fill: T.muted, fontSize: 10, position: 'insideBottom', offset: -10 }} />
            <YAxis dataKey="ret" name="Return" tickFormatter={fmt}
              tick={{ fill: T.muted, fontSize: 10 }}
              label={{ value: 'Return', fill: T.muted, fontSize: 10, angle: -90, position: 'insideLeft', offset: 12 }} />
            <Tooltip cursor={false}
              contentStyle={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 8, fontSize: 11 }}
              formatter={(v, n) => [fmt(v), n]} />
            <Scatter data={sample} fill={T.muted} opacity={0.2} r={2} />
            <Scatter data={frontier} fill={T.blue} opacity={0.9} r={3} />
            {point && <ReferenceDot x={point.vol} y={point.ret} r={9} fill={T.gold} stroke={T.bg} strokeWidth={2} label={{ value: '★', fill: T.gold, fontSize: 14, dy: -2 }} />}
          </ScatterChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', gap: 16, marginTop: 6, fontSize: 10, color: T.muted }}>
          <span><span style={{ color: T.gold }}>★</span> Selected portfolio</span>
          <span><span style={{ color: T.blue }}>●</span> Efficient frontier</span>
        </div>
      </div>
    );
  };

  /* ═══════════════════════════════════════════════════════
     TABS
  ═══════════════════════════════════════════════════════ */

  const TABS = [
    { key: "upload", label: "UPLOAD DATA", icon: <Upload size={11} /> },
    { key: "l2",     label: "LEVEL 2 · STOCKS", icon: <TrendingUp size={11} /> },
    { key: "l1",     label: "LEVEL 1 · PORTFOLIO", icon: <Layers size={11} /> }
  ];

  const tabStyle = active => ({
    padding: '11px 22px', cursor: 'pointer', fontSize: 11, fontWeight: 700,
    letterSpacing: '0.08em', color: active ? T.gold : T.muted,
    borderBottom: `2px solid ${active ? T.gold : 'transparent'}`,
    background: 'transparent', border: 'none', fontFamily: 'inherit',
    display: 'inline-flex', alignItems: 'center', gap: 7, transition: 'color 0.15s',
  });

  /* ═══════════════════════════════════════════════════════
     RENDER
  ═══════════════════════════════════════════════════════ */

  const statBox = (label, val, color) => (
    <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 8, padding: '14px 18px' }}>
      <div style={{ ...S.label, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color, ...S.mono }}>{val}</div>
    </div>
  );

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { width: 100%; min-height: 100vh; }
        #root { width: 100%; min-height: 100vh; display: flex; flex-direction: column; }
        ::-webkit-scrollbar { width: 5px; } ::-webkit-scrollbar-track { background: #0c1118; } ::-webkit-scrollbar-thumb { background: #1a2840; border-radius: 3px; }
        input[type=file] { display: none; }
        select { background: #0c1118; border: 1px solid #1a2840; color: #b8c8d8; border-radius: 6px; padding: 8px 12px; font-family: inherit; font-size: 12px; outline: none; cursor: pointer; }
        option { background: #0c1118; }
      `}</style>

      <div style={{ minHeight: '100vh', width: '100%', background: T.bg, color: T.text, fontFamily: "'IBM Plex Sans', sans-serif", minWidth: 900 }}>

        {/* ── HEADER ── */}
        <div style={{ background: T.surface, borderBottom: `1px solid ${T.border}`, padding: '14px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 700, color: T.textHi, letterSpacing: '0.05em' }}>
              <span style={{ color: T.gold }}>◈</span> MPT PORTFOLIO OPTIMIZER
            </div>
            <div style={{ fontSize: 9, color: T.muted, marginTop: 3, letterSpacing: '0.12em', ...S.mono }}>
              TWO-LEVEL MEAN-VARIANCE · INDIA MARKETS · RF {pct(RISK_FREE)} · {N_SIMS.toLocaleString()} MONTE CARLO SIMS
            </div>
          </div>
          <div style={{ display: 'flex', gap: 24, fontSize: 10, ...S.mono, color: T.muted, textAlign: 'right' }}>
            <div><div style={{ color: T.green }}>◎ {Object.keys(stocks).length} STOCKS LOADED</div><div>{Object.values(stocks).filter(s => s.enabled).length} active</div></div>
            <div><div style={{ color: T.blue }}>◎ {assetClasses.filter(a => a.enabled).length} ASSET CLASSES</div><div>Level 1 active</div></div>
          </div>
        </div>

        {/* ── TABS ── */}
        <div style={{ background: T.surface, borderBottom: `1px solid ${T.border}`, padding: '0 32px', display: 'flex' }}>
          {TABS.map(t => (
            <button key={t.key} onClick={() => setTab(t.key)} style={tabStyle(tab === t.key)}>
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        {/* ── CONTENT ── */}
        <div style={{ padding: '28px 32px' }}>

          {/* ─────────── UPLOAD TAB ─────────── */}
          {tab === 'upload' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

              {/* Stock upload zone */}
              <div style={S.card}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: T.green }}>Level 2 — Equity Basket</div>
                  {/* Data frequency toggle — applies to ALL uploads */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ ...S.label }}>Data Frequency</span>
                    <div style={{ display: 'flex', borderRadius: 6, overflow: 'hidden', border: `1px solid ${T.border}` }}>
                      {['daily', 'monthly'].map(f => (
                        <button key={f} onClick={() => setDataFreq(f)}
                          style={{ padding: '6px 14px', fontSize: 11, fontWeight: 700, fontFamily: 'inherit', cursor: 'pointer', letterSpacing: '0.06em', border: 'none', transition: 'all 0.15s',
                            background: dataFreq === f ? T.gold : T.surface,
                            color:      dataFreq === f ? T.bg   : T.muted,
                          }}>
                          {f.toUpperCase()}
                        </button>
                      ))}
                    </div>
                    <span style={{ fontSize: 10, color: T.muted }}>
                      ×{dataFreq === 'monthly' ? '12' : '252'} annualisation
                    </span>
                  </div>
                </div>
                <div style={{ fontSize: 11, color: T.muted, marginBottom: 16 }}>
                  Upload one or many CSVs. Each file must have a <span style={{ color: T.text }}>Date</span> and <span style={{ color: T.text }}>Close</span> column. Filename becomes the ticker label.
                </div>

                {/* Drop zone */}
                <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, height: 100, border: `2px dashed ${T.border}`, borderRadius: 8, cursor: 'pointer', background: T.surface, transition: 'border-color 0.2s' }}
                  onDragOver={e => e.preventDefault()}
                  onDrop={e => { e.preventDefault(); handleStockUpload(e.dataTransfer.files); }}>
                  <input type="file" multiple accept=".csv" onChange={e => handleStockUpload(e.target.files)} />
                  <Upload size={22} color={T.muted} />
                  <span style={{ fontSize: 11, color: T.muted }}>Drop CSVs here or <span style={{ color: T.gold }}>click to browse</span></span>
                  <span style={{ fontSize: 10, color: T.dim }}>Multiple files supported</span>
                </label>

                {/* Stock list */}
                {Object.keys(stocks).length > 0 && (
                  <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr 1fr 1fr 60px 36px', gap: 10, padding: '0 10px 8px', borderBottom: `1px solid ${T.border}` }}>
                      {['Ticker', 'Ann. Return', 'Ann. Vol', 'Sharpe', 'Active', ''].map(h => (
                        <div key={h} style={{ ...S.label }}>{h}</div>
                      ))}
                    </div>
                    {Object.entries(stocks).map(([ticker, s]) => {
                      const sharpe = (s.stats.annReturn - RISK_FREE) / s.stats.annVol;
                      return (
                        <div key={ticker} style={{ display: 'grid', gridTemplateColumns: '120px 1fr 1fr 1fr 60px 36px', gap: 10, alignItems: 'center', padding: '9px 10px', background: s.enabled ? `${T.green}08` : T.surface, borderRadius: 7, border: `1px solid ${s.enabled ? T.green + '30' : T.border}`, transition: 'all 0.15s' }}>
                          <div style={{ ...S.mono, fontSize: 12, color: T.gold, fontWeight: 700 }}>{ticker}</div>
                          <div style={{ ...S.mono, fontSize: 12, color: s.stats.annReturn >= 0 ? T.green : T.red }}>{pct(s.stats.annReturn)}</div>
                          <div style={{ ...S.mono, fontSize: 12, color: T.text }}>{pct(s.stats.annVol)}</div>
                          <div style={{ ...S.mono, fontSize: 12, color: sharpe > 1 ? T.gold : T.text }}>{sharpe.toFixed(3)}</div>
                          <div onClick={() => toggleStock(ticker)} style={{ width: 28, height: 16, borderRadius: 8, background: s.enabled ? T.green : T.dim, cursor: 'pointer', position: 'relative', transition: 'background 0.2s' }}>
                            <div style={{ position: 'absolute', width: 12, height: 12, borderRadius: '50%', background: '#fff', top: 2, left: s.enabled ? 14 : 2, transition: 'left 0.2s' }} />
                          </div>
                          <button onClick={() => removeStock(ticker)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: T.muted, display: 'flex', alignItems: 'center', padding: 0 }}>
                            <Trash2 size={14} />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Asset class uploads */}
              <div style={S.card}>
                <div style={{ fontSize: 13, fontWeight: 700, color: T.gold, marginBottom: 4 }}>Level 1 — Asset Classes</div>
                <div style={{ fontSize: 11, color: T.muted, marginBottom: 16 }}>Upload monthly price CSV for each asset class. Toggle to include/exclude from Level 1.</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {assetClasses.map(a => {
                    if (a.isBasket) return (
                      <div key={a.id} style={{ display: 'grid', gridTemplateColumns: '140px 1fr 160px 36px', gap: 12, alignItems: 'center', padding: '10px 14px', background: `${T.gold}08`, borderRadius: 7, border: `1px solid ${T.gold}30` }}>
                        <div style={{ ...S.mono, fontSize: 10, color: T.gold }}>EQUITY BASKET</div>
                        <div style={{ fontSize: 11, color: T.text }}>{a.label} <span style={{ color: T.muted }}>(computed from Level 2)</span></div>
                        <div />
                        <CheckCircle size={16} color={l2Result ? T.green : T.dim} />
                      </div>
                    );
                    const uploaded = !!assetPrices[a.id];
                    return (
                      <div key={a.id} style={{ display: 'grid', gridTemplateColumns: '140px 1fr 160px 36px 28px', gap: 12, alignItems: 'center', padding: '10px 14px', background: uploaded && a.enabled ? `${T.green}06` : T.surface, borderRadius: 7, border: `1px solid ${uploaded && a.enabled ? T.green + '40' : T.border}`, opacity: a.enabled ? 1 : 0.5, transition: 'all 0.15s' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div onClick={() => toggleAsset(a.id)} style={{ width: 14, height: 14, borderRadius: 3, border: `2px solid ${a.color}`, background: a.enabled ? a.color : 'transparent', cursor: 'pointer', flexShrink: 0, transition: 'all 0.15s' }} />
                          <span style={{ ...S.mono, fontSize: 10, color: T.muted }}>ASSET</span>
                        </div>
                        <input
                          value={a.label}
                          onChange={e => renameAssetClass(a.id, e.target.value)}
                          style={{ ...S.input, fontSize: 11, padding: '5px 10px' }}
                        />
                        <label style={{ cursor: a.enabled ? 'pointer' : 'default' }}>
                          <input type="file" accept=".csv" disabled={!a.enabled} onChange={e => handleAssetUpload(a.id, e.target.files[0])} />
                          <div style={{ ...S.btn(uploaded ? T.green : T.muted), justifyContent: 'center', fontSize: 10, opacity: a.enabled ? 1 : 0.4 }}>
                            <Upload size={11} /> {uploaded ? 'Replace CSV' : 'Upload CSV'}
                          </div>
                        </label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          {uploaded && a.enabled && <CheckCircle size={14} color={T.green} />}
                        </div>
                        <button onClick={() => removeAssetClass(a.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: T.muted, display: 'flex', alignItems: 'center', padding: 0 }}>
                          <Trash2 size={13} />
                        </button>
                      </div>
                    );
                  })}

                  {/* Add new asset class */}
                  <button onClick={addAssetClass} style={{ ...S.btn(T.muted), fontSize: 11, alignSelf: 'flex-start', marginTop: 4 }}>
                    <span style={{ fontSize: 16, lineHeight: 1 }}>+</span> Add Asset Class
                  </button>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button onClick={() => setTab('l2')} disabled={Object.values(stocks).filter(s => s.enabled).length < 2}
                  style={{ ...S.btn(T.gold, true), opacity: Object.values(stocks).filter(s => s.enabled).length < 2 ? 0.4 : 1 }}>
                  Go to Level 2 <ChevronRight size={14} />
                </button>
              </div>
            </div>
          )}

          {/* ─────────── LEVEL 2 TAB ─────────── */}
          {tab === 'l2' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div style={S.card}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: T.green, letterSpacing: '0.04em' }}>LEVEL 2 — STOCK BASKET MPT</div>
                    <div style={{ fontSize: 11, color: T.muted, marginTop: 4 }}>
                      {Object.values(stocks).filter(s => s.enabled).length} active stocks · {N_SIMS.toLocaleString()} simulations · 40% max weight
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div>
                      <div style={{ ...S.label, marginBottom: 5 }}>Objective</div>
                      <select value={l2Mode} onChange={e => setL2Mode(e.target.value)}>
                        <option value="maxSharpe">Max Sharpe</option>
                        <option value="aggressive">Aggressive (95th pct return)</option>
                      </select>
                    </div>
                    <button onClick={runLevel2} disabled={l2Running || Object.values(stocks).filter(s => s.enabled).length < 2}
                      style={{ ...S.btn(T.green, true), opacity: l2Running ? 0.7 : 1 }}>
                      {l2Running
                        ? <><Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} /> Running…</>
                        : <><TrendingUp size={13} /> Run Optimisation</>}
                    </button>
                  </div>
                </div>
                {Object.values(stocks).filter(s => s.enabled).length < 2 && (
                  <div style={{ marginTop: 12, padding: '10px 14px', background: `${T.gold}10`, border: `1px solid ${T.gold}40`, borderRadius: 6, fontSize: 11, color: T.gold }}>
                    ⚠ Enable at least 2 stocks in the Upload Data tab first.
                  </div>
                )}
              </div>

              {l2Result && (<>
                <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 20 }}>
                  <div style={S.card}><FrontierChart cloud={l2Result.portfolios} frontier={l2Result.frontier} point={l2Result.selected} title={`EFFICIENT FRONTIER — ${l2Result.labels.length} STOCKS`} /></div>
                  <div style={S.card}>
                    <div style={{ ...S.label, marginBottom: 14 }}>Weights — {l2Mode === 'aggressive' ? 'Aggressive' : 'Max Sharpe'}</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {l2Result.labels.map((lbl, i) => (
                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: `1px solid ${T.border}30` }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div style={{ width: 6, height: 6, borderRadius: 1, background: l2Result.colors[i] }} />
                            <span style={{ fontSize: 11, color: T.text, ...S.mono }}>{lbl}</span>
                          </div>
                          <span style={{ fontSize: 12, color: T.gold, ...S.mono }}>{pct(l2Result.selected.w[i])}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <div style={S.card}>
                  <div style={{ ...S.label, marginBottom: 14 }}>Allocation</div>
                  <AllocationBar weights={l2Result.selected.w} labels={l2Result.labels} colors={l2Result.colors} />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                  {statBox('Expected Return', pct(l2Result.selected.ret), T.green)}
                  {statBox('Volatility', pct(l2Result.selected.vol), T.text)}
                  {statBox('Sharpe Ratio', l2Result.selected.sharpe.toFixed(3), T.gold)}
                  {statBox('vs Max Sharpe', l2Mode === 'aggressive' ? `+${pct(l2Result.selected.ret - l2Result.maxSharpe.ret)}` : '—', T.cyan)}
                </div>
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <button onClick={() => setTab('l1')} style={{ ...S.btn(T.gold, true) }}>
                    Run Level 1 Portfolio <ChevronRight size={14} />
                  </button>
                </div>
              </>)}
            </div>
          )}

          {/* ─────────── LEVEL 1 TAB ─────────── */}
          {tab === 'l1' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div style={S.card}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: T.gold, letterSpacing: '0.04em' }}>LEVEL 1 — FULL PORTFOLIO MPT</div>
                    <div style={{ fontSize: 11, color: T.muted, marginTop: 4 }}>
                      {assetClasses.filter(a => a.enabled).length} asset classes · aggressive 95th pct bias
                    </div>
                  </div>
                  <button onClick={runLevel1} disabled={l1Running || !l2Result}
                    style={{ ...S.btn(T.gold, true), opacity: (!l2Result || l1Running) ? 0.5 : 1 }}>
                    {l1Running
                      ? <><Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} /> Running…</>
                      : <><Layers size={13} /> Run Optimisation</>}
                  </button>
                </div>
                {!l2Result && (
                  <div style={{ marginTop: 12, padding: '10px 14px', background: '#102030', border: `1px solid ${T.blue}40`, borderRadius: 6, fontSize: 11, color: T.blue, display: 'flex', gap: 8, alignItems: 'center' }}>
                    <ChevronRight size={13} /> Run Level 2 stock optimisation first.
                  </div>
                )}
                {l2Result && (
                  <div style={{ marginTop: 12, padding: '8px 14px', background: '#002818', border: `1px solid ${T.green}40`, borderRadius: 6, fontSize: 11, color: T.green }}>
                    ✓ Level 2 basket — return {pct(l2Result.selected.ret)} · vol {pct(l2Result.selected.vol)} · Sharpe {l2Result.selected.sharpe.toFixed(3)}
                  </div>
                )}
              </div>

              {l1Result && (<>
                <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 20 }}>
                  <div style={S.card}><FrontierChart cloud={l1Result.portfolios} frontier={l1Result.frontier} point={l1Result.selected} title={`EFFICIENT FRONTIER — ${l1Result.labels.length} ASSET CLASSES`} /></div>
                  <div style={S.card}>
                    <div style={{ ...S.label, marginBottom: 14 }}>Weights — Aggressive Portfolio</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {l1Result.labels.map((lbl, i) => (
                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: `1px solid ${T.border}30` }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div style={{ width: 6, height: 6, borderRadius: 1, background: l1Result.colors[i] }} />
                            <span style={{ fontSize: 11, color: T.text }}>{lbl}</span>
                          </div>
                          <span style={{ fontSize: 12, color: T.gold, ...S.mono }}>{pct(l1Result.selected.w[i])}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <div style={S.card}>
                  <div style={{ ...S.label, marginBottom: 14 }}>Final Allocation</div>
                  <AllocationBar weights={l1Result.selected.w} labels={l1Result.labels} colors={l1Result.colors} />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                  {statBox('Expected Return', pct(l1Result.selected.ret), T.green)}
                  {statBox('Volatility', pct(l1Result.selected.vol), T.text)}
                  {statBox('Sharpe Ratio', l1Result.selected.sharpe.toFixed(3), T.gold)}
                  {statBox('Min Vol Return', pct(l1Result.minVol.ret), T.muted)}
                </div>
              </>)}
            </div>
          )}

        </div>
      </div>
    </>
  );
}