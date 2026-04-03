import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import {
  Microscope, Scan, Droplets, Activity, Zap, Upload,
  AlertTriangle, CheckCircle, XCircle, Bot, Send,
  Loader2, BarChart2, FlaskConical, ChevronDown,
  ChevronUp, Cpu, Shield, Wind, Radio, ClipboardList
} from 'lucide-react'

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 120000
})

const callAPI = {
  breastImage:   (f) => { const d = new FormData(); d.append('file', f); return API.post('/predict/breast-image', d) },
  breastTabular: (features) => API.post('/predict/breast-tabular', { features }),
  skin:          (f) => { const d = new FormData(); d.append('file', f); return API.post('/predict/skin', d) },
  blood:         (f) => { const d = new FormData(); d.append('file', f); return API.post('/predict/blood', d) },
  lung:          (f) => { const d = new FormData(); d.append('file', f); return API.post('/predict/lung', d) },
  xray:          (f) => { const d = new FormData(); d.append('file', f); return API.post('/predict/xray', d) },
  ecg:           (f) => { const d = new FormData(); d.append('file', f); return API.post('/predict/ecg', d) },
  universal:     (f) => { const d = new FormData(); d.append('file', f); return API.post('/predict/universal', d) },
  chat: (ctx, msg, hist) => API.post('/chat/recommendation', { prediction_context: ctx, user_message: msg, chat_history: hist }),
}

const CANCER_TYPES = [
  { id: 'breast', label: 'Breast',      icon: Microscope, color: '#DB2777', accept: { 'image/*': ['.png','.jpg','.jpeg'] }, hint: 'Mammogram or histopathology image' },
  { id: 'skin',   label: 'Skin',        icon: Scan,       color: '#D97706', accept: { 'image/*': ['.png','.jpg','.jpeg'] }, hint: 'Dermoscopy or skin lesion image' },
  { id: 'blood',  label: 'Blood',       icon: Droplets,   color: '#DC2626', accept: { 'image/*': ['.png','.jpg','.jpeg'] }, hint: 'Blood smear microscopy image' },
  { id: 'lung',   label: 'Lung',        icon: Wind,       color: '#7C3AED', accept: { 'image/*': ['.png','.jpg','.jpeg'] }, hint: 'Lung CT scan image' },
  { id: 'xray',   label: 'Chest X-Ray', icon: Radio,      color: '#059669', accept: { 'image/*': ['.png','.jpg','.jpeg'] }, hint: 'Chest X-Ray image' },
  { id: 'ecg',    label: 'ECG',         icon: Activity,   color: '#0284C7', accept: { 'text/csv': ['.csv'] },              hint: 'ECG signal CSV file' },
]

const BREAST_FEATURES = [
  "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
  "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
  "radius error","texture error","perimeter error","area error","smoothness error",
  "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
  "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
  "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

const DEMO_BENIGN    = [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
const DEMO_MALIGNANT = [20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902]

const fmt = (v) => {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(3)
  return String(v)
}

const T = {
  bg:        '#F8FAFC',
  surface:   '#FFFFFF',
  surface2:  '#F1F5F9',
  border:    '#E2E8F0',
  border2:   '#CBD5E1',
  text:      '#0F172A',
  text2:     '#475569',
  text3:     '#94A3B8',
  primary:   '#0EA5E9',
  primaryDk: '#0284C7',
  success:   '#059669',
  danger:    '#DC2626',
}

function ConfBar({ pct, color }) {
  const safe = Math.max(0, Math.min(100, Number(pct) || 0))
  return (
    <div className="h-2 rounded-full overflow-hidden" style={{ background: T.surface2 }}>
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${safe}%` }}
        transition={{ duration: 1.1, ease: [0.34, 1.56, 0.64, 1] }}
        className="h-full rounded-full"
        style={{ background: `linear-gradient(90deg, ${color}99, ${color})` }}
      />
    </div>
  )
}

function FeaturePanel({ data, title = 'Extracted Features' }) {
  const [open, setOpen] = useState(false)
  if (!data || typeof data !== 'object') return null
  const rows = Object.entries(data).filter(([, v]) => ['number','string','boolean'].includes(typeof v))
  if (!rows.length) return null
  return (
    <div className="rounded-xl overflow-hidden" style={{ border: `1px solid ${T.border}` }}>
      <button onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-2.5 transition-colors hover:bg-slate-50"
        style={{ background: T.surface2 }}>
        <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: T.text3 }}>{title}</span>
        {open ? <ChevronUp size={12} style={{ color: T.text3 }} /> : <ChevronDown size={12} style={{ color: T.text3 }} />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden">
            <div className="px-3 pb-3 pt-2 grid grid-cols-2 gap-1.5" style={{ background: T.surface }}>
              {rows.map(([k, v]) => (
                <div key={k} className="flex items-center justify-between px-2.5 py-1.5 rounded-lg" style={{ background: T.surface2, border: `1px solid ${T.border}` }}>
                  <span className="text-[11px] capitalize truncate" style={{ color: T.text2 }}>{k.replace(/_/g, ' ')}</span>
                  <span className="text-[11px] font-mono font-bold ml-2 shrink-0" style={{ color: T.primaryDk }}>{fmt(v)}</span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function ResultCard({ result }) {
  if (!result || typeof result !== 'object') return null
  const pct      = Math.max(0, Math.min(100, Number(result.confidence ?? 0)))
  const positive = Boolean(result.is_malignant || result.is_positive || result.is_abnormal || result.is_pneumonia)
  const rc       = positive ? T.danger : T.success
  const rBg      = positive ? '#FEF2F2' : '#F0FDF4'
  const rBorder  = positive ? '#FECACA' : '#BBF7D0'

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
      <div className="rounded-2xl p-5" style={{ background: rBg, border: `1.5px solid ${rBorder}` }}>
        <div className="flex items-start gap-3">
          {positive
            ? <XCircle size={24} style={{ color: rc }} className="shrink-0 mt-0.5" />
            : <CheckCircle size={24} style={{ color: rc }} className="shrink-0 mt-0.5" />}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <p className="text-lg font-bold" style={{ color: rc }}>{result.prediction || 'No prediction'}</p>
              {result.risk_level && (
                <span className="text-[10px] font-bold px-2.5 py-1 rounded-full shrink-0"
                  style={{ background: positive ? '#FEE2E2' : '#DCFCE7', color: rc, border: `1px solid ${rBorder}` }}>
                  {result.risk_level} Risk
                </span>
              )}
            </div>
            {result.cancer_type   && <p className="text-xs mt-0.5" style={{ color: T.text2 }}>{result.cancer_type}</p>}
            {result.lesion_type   && <p className="text-xs mt-0.5" style={{ color: T.text2 }}>{result.lesion_type}</p>}
            {result.finding       && <p className="text-xs mt-0.5" style={{ color: T.text2 }}>{result.finding}</p>}
            {result.analysis_target && <p className="text-xs mt-1" style={{ color: T.text3 }}>{result.analysis_target}</p>}
          </div>
        </div>

        <div className="mt-4 space-y-1.5">
          <div className="flex justify-between">
            <span className="text-xs" style={{ color: T.text2 }}>Confidence Score</span>
            <span className="text-xs font-mono font-bold" style={{ color: rc }}>{pct.toFixed(1)}%</span>
          </div>
          <ConfBar pct={pct} color={rc} />
        </div>

        {result.model_type && (
          <div className="flex items-center gap-1.5 mt-3">
            <Cpu size={10} style={{ color: T.text3 }} />
            <span className="text-[10px]" style={{ color: T.text3 }}>{result.model_type}</span>
          </div>
        )}
      </div>

      {result.class_scores && (
        <div className="rounded-xl p-4 space-y-2" style={{ background: T.surface, border: `1px solid ${T.border}` }}>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: T.text3 }}>Class Probabilities</p>
          {Object.entries(result.class_scores).map(([cls, p]) => {
            const prob = Math.max(0, Math.min(100, Number(p ?? 0)))
            return (
              <div key={cls} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span style={{ color: T.text2 }}>{cls}</span>
                  <span className="font-mono font-semibold" style={{ color: T.text }}>{prob.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: T.surface2 }}>
                  <motion.div initial={{ width: 0 }} animate={{ width: `${prob}%` }} transition={{ duration: 0.8 }}
                    className="h-full rounded-full" style={{ background: '#7C3AED' }} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {result.abcde_flags && (
        <div className="rounded-xl p-4" style={{ background: T.surface, border: `1px solid ${T.border}` }}>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: T.text3 }}>ABCDE Dermoscopy</p>
          <div className="space-y-1.5">
            {Object.entries(result.abcde_flags).map(([k, v]) => (
              <div key={k} className="flex justify-between text-xs">
                <span style={{ color: T.text2 }}>{k.replace(/_/g, ' ')}</span>
                <span className="font-semibold" style={{ color: ['Present','Irregular','Variable'].includes(String(v)) ? '#D97706' : T.success }}>{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <FeaturePanel data={result.features} />
      {result.feature_summary && <FeaturePanel data={result.feature_summary} title="Clinical Feature Summary" />}

      {result.ai_recommendation && (
        <div className="rounded-xl p-4" style={{ background: '#EFF6FF', border: '1px solid #BFDBFE' }}>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-2" style={{ color: T.primaryDk }}>AI Recommendation</p>
          <p className="text-xs leading-relaxed" style={{ color: '#1E40AF' }}>{result.ai_recommendation}</p>
        </div>
      )}
    </motion.div>
  )
}

function AIChat({ context }) {
  const [msgs, setMsgs]       = useState([{ role: 'assistant', text: context ? 'Result received. What would you like to know?' : 'Run a scan above, then ask me anything.' }])
  const [input, setInput]     = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef             = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])
  useEffect(() => { if (context) setMsgs([{ role: 'assistant', text: 'Result received. Ask me anything about this finding.' }]) }, [context])

  const send = async () => {
    if (!input.trim() || loading) return
    const q = input.trim(); setInput(''); setLoading(true)
    setMsgs(prev => [...prev, { role: 'user', text: q }])
    try {
      const hist = msgs.slice(-6).map(m => ({ role: m.role, content: m.text }))
      const { data } = await callAPI.chat(context || 'General inquiry', q, hist)
      setMsgs(prev => [...prev, { role: 'assistant', text: data?.response || 'No response.' }])
    } catch {
      setMsgs(prev => [...prev, { role: 'assistant', text: 'Unable to connect. Please consult a healthcare professional.' }])
    } finally { setLoading(false) }
  }

  return (
    <div className="space-y-3">
      <div className="space-y-2.5 max-h-52 overflow-y-auto pr-0.5">
        {msgs.map((m, i) => (
          <div key={i} className={`flex gap-2 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
            <div className="w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[9px] font-bold"
              style={{ background: m.role === 'user' ? '#DBEAFE' : '#D1FAE5', color: m.role === 'user' ? T.primaryDk : T.success }}>
              {m.role === 'user' ? 'U' : <Bot size={11} />}
            </div>
            <div className="max-w-[88%] px-3 py-2 rounded-xl text-xs leading-relaxed shadow-sm"
              style={{
                background: m.role === 'user' ? '#EFF6FF' : T.surface,
                border: `1px solid ${m.role === 'user' ? '#BFDBFE' : T.border}`,
                color: T.text
              }}>
              {m.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex gap-2">
            <div className="w-6 h-6 rounded-full flex items-center justify-center shrink-0" style={{ background: '#D1FAE5' }}>
              <Bot size={11} style={{ color: T.success }} />
            </div>
            <div className="px-3 py-2 rounded-xl flex gap-1 items-center shadow-sm" style={{ background: T.surface, border: `1px solid ${T.border}` }}>
              <span className="typing-dot w-1.5 h-1.5 rounded-full" style={{ background: T.success }} />
              <span className="typing-dot w-1.5 h-1.5 rounded-full" style={{ background: T.success }} />
              <span className="typing-dot w-1.5 h-1.5 rounded-full" style={{ background: T.success }} />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="flex gap-2">
        <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder="Ask about your results..."
          className="flex-1 px-3 py-2 text-xs rounded-xl focus:outline-none transition-all"
          style={{ background: T.surface2, border: `1px solid ${T.border}`, color: T.text }}
        />
        <button onClick={send} disabled={!input.trim() || loading}
          className="p-2 rounded-xl transition-all disabled:opacity-40"
          style={{ background: T.primaryDk, color: '#fff' }}>
          <Send size={13} />
        </button>
      </div>
    </div>
  )
}

function DropZone({ onFile, accept, hint, file, color, isCSV }) {
  const onDrop = useCallback(f => f[0] && onFile(f[0]), [onFile])
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept, maxFiles: 1 })
  const preview = useMemo(() => { if (!file || isCSV) return null; return URL.createObjectURL(file) }, [file, isCSV])
  useEffect(() => { return () => { if (preview) URL.revokeObjectURL(preview) } }, [preview])

  return (
    <div {...getRootProps()} className="relative rounded-2xl overflow-hidden cursor-pointer transition-all duration-200"
      style={{ border: `2px dashed ${isDragActive ? color : T.border2}`, background: isDragActive ? `${color}08` : T.surface2 }}>
      <input {...getInputProps()} />
      {preview ? (
        <div className="relative">
          <img src={preview} alt="preview" className="w-full h-40 object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
          <div className="absolute bottom-3 left-3 flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-white font-medium truncate max-w-[200px]">{file.name}</span>
          </div>
        </div>
      ) : file ? (
        <div className="py-7 flex flex-col items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: `${color}15` }}>
            <Activity size={16} style={{ color }} />
          </div>
          <p className="text-xs font-medium truncate max-w-[200px]" style={{ color: T.text2 }}>{file.name}</p>
        </div>
      ) : (
        <div className="py-8 flex flex-col items-center gap-3">
          <motion.div animate={isDragActive ? { scale: 1.08 } : { scale: 1 }}
            className="w-12 h-12 rounded-2xl flex items-center justify-center shadow-sm"
            style={{ background: `${color}12`, border: `1.5px solid ${color}30` }}>
            <Upload size={20} style={{ color }} />
          </motion.div>
          <div className="text-center">
            <p className="text-sm font-semibold" style={{ color: T.text }}>{isDragActive ? 'Release to upload' : 'Drop here or click to browse'}</p>
            <p className="text-xs mt-1" style={{ color: T.text3 }}>{hint}</p>
          </div>
        </div>
      )}
    </div>
  )
}

function ChatBox({ context }) {
  return (
    <div className="rounded-2xl p-4 space-y-3 shadow-sm" style={{ background: T.surface, border: `1px solid ${T.border}` }}>
      <div className="flex items-center gap-2">
        <div className="w-6 h-6 rounded-lg flex items-center justify-center" style={{ background: '#D1FAE5' }}>
          <Bot size={12} style={{ color: T.success }} />
        </div>
        <p className="text-xs font-semibold" style={{ color: T.text2 }}>AI Medical Assistant</p>
      </div>
      <AIChat context={context} />
    </div>
  )
}

function LoadingCard({ color, icon: Icon }) {
  return (
    <div className="h-full min-h-[200px] rounded-2xl flex flex-col items-center justify-center gap-4 shadow-sm"
      style={{ background: T.surface, border: `1px solid ${T.border}` }}>
      <div className="relative w-12 h-12">
        <div className="absolute inset-0 rounded-full" style={{ border: `2px solid ${T.border}` }} />
        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-current animate-spin" style={{ borderTopColor: color }} />
        <Icon size={16} className="absolute inset-0 m-auto" style={{ color }} />
      </div>
      <p className="text-xs" style={{ color: T.text2 }}>Running AI inference...</p>
    </div>
  )
}

function EmptyCard() {
  return (
    <div className="h-full min-h-[200px] rounded-2xl flex flex-col items-center justify-center gap-3 shadow-sm"
      style={{ background: T.surface, border: `1.5px dashed ${T.border2}` }}>
      <BarChart2 size={22} style={{ color: T.text3 }} />
      <p className="text-xs" style={{ color: T.text3 }}>Upload and run analysis to see results</p>
    </div>
  )
}

function ErrorBox({ msg }) {
  return (
    <div className="flex items-start gap-2 p-3 rounded-xl" style={{ background: '#FEF2F2', border: '1px solid #FECACA' }}>
      <AlertTriangle size={12} style={{ color: T.danger }} className="mt-0.5 shrink-0" />
      <p className="text-xs" style={{ color: '#991B1B' }}>{msg}</p>
    </div>
  )
}

function BreastSection({ color }) {
  const [mode, setMode]         = useState('image')
  const [file, setFile]         = useState(null)
  const [features, setFeatures] = useState(Array(30).fill(0))
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const reset = () => { setResult(null); setError(null) }

  const runImage = async () => {
    if (!file) return; setLoading(true); setResult(null); setError(null)
    try { const { data } = await callAPI.breastImage(file); setResult(data) }
    catch (e) { setError(e.response?.data?.error || e.message || 'Analysis failed.') }
    finally { setLoading(false) }
  }

  const runTabular = async () => {
    setLoading(true); setResult(null); setError(null)
    try { const { data } = await callAPI.breastTabular(features.map(Number)); setResult(data) }
    catch (e) { setError(e.response?.data?.error || e.message || 'Analysis failed.') }
    finally { setLoading(false) }
  }

  const chatCtx = result ? `Breast ${mode} analysis: ${result.prediction} with ${Number(result.confidence ?? 0).toFixed(1)}% confidence.` : null

  return (
    <div className="space-y-4">
      <div className="flex gap-1.5 p-1 rounded-xl w-fit shadow-sm" style={{ background: T.surface2, border: `1px solid ${T.border}` }}>
        {[['image', Microscope, 'Image Detection'], ['tabular', ClipboardList, 'Clinical Risk Estimator']].map(([m, Icon, lbl]) => (
          <button key={m} onClick={() => { setMode(m); reset() }}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all"
            style={{
              background: mode === m ? T.surface : 'transparent',
              color: mode === m ? color : T.text3,
              border: mode === m ? `1px solid ${color}30` : '1px solid transparent',
              boxShadow: mode === m ? '0 1px 3px rgba(0,0,0,0.08)' : 'none'
            }}>
            <Icon size={12} />{lbl}
          </button>
        ))}
      </div>

      {mode === 'image' ? (
        <div className="grid lg:grid-cols-2 gap-4">
          <div className="space-y-3">
            <p className="text-xs" style={{ color: T.text2 }}>Upload mammogram or histopathology image for ResNet50V2 classification.</p>
            <DropZone onFile={setFile} accept={{ 'image/*': ['.png','.jpg','.jpeg'] }} hint="Mammogram or histopathology image" file={file} color={color} isCSV={false} />
            <button onClick={runImage} disabled={!file || loading}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-bold transition-all disabled:cursor-not-allowed disabled:opacity-50 shadow-sm"
              style={{ background: file && !loading ? color : T.surface2, color: file && !loading ? '#fff' : T.text3, border: `1.5px solid ${file && !loading ? color : T.border}` }}>
              {loading ? <><Loader2 size={14} className="animate-spin" />Analyzing...</> : <><Microscope size={14} />Analyze Mammogram</>}
            </button>
            {error && <ErrorBox msg={error} />}
          </div>
          <div>
            {loading && <LoadingCard color={color} icon={Microscope} />}
            {result && !loading && <><ResultCard result={result} /><div className="mt-3"><ChatBox context={chatCtx} /></div></>}
            {!result && !loading && <EmptyCard />}
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <p className="text-xs" style={{ color: T.text2 }}>Enter 30 clinical biopsy features for neural network risk estimation.</p>
              <div className="flex gap-2">
                <button onClick={() => setFeatures(DEMO_BENIGN)}
                  className="text-[10px] px-3 py-1.5 rounded-lg font-semibold shadow-sm"
                  style={{ background: '#F0FDF4', color: T.success, border: '1px solid #BBF7D0' }}>Demo Benign</button>
                <button onClick={() => setFeatures(DEMO_MALIGNANT)}
                  className="text-[10px] px-3 py-1.5 rounded-lg font-semibold shadow-sm"
                  style={{ background: '#FEF2F2', color: T.danger, border: '1px solid #FECACA' }}>Demo Malignant</button>
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-72 overflow-y-auto pr-1 rounded-xl p-3"
              style={{ background: T.surface2, border: `1px solid ${T.border}` }}>
              {BREAST_FEATURES.map((name, i) => (
                <div key={i}>
                  <label className="text-[9px] block mb-1 capitalize truncate font-medium" style={{ color: T.text3 }}>{name}</label>
                  <input type="number" step="any" value={features[i]}
                    onChange={e => { const c = [...features]; c[i] = e.target.value; setFeatures(c) }}
                    className="w-full px-2.5 py-1.5 text-xs rounded-lg focus:outline-none transition-all"
                    style={{ background: T.surface, border: `1px solid ${T.border}`, color: T.text }}
                  />
                </div>
              ))}
            </div>
            <button onClick={runTabular} disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-bold transition-all shadow-sm"
              style={{ background: color, color: '#fff' }}>
              {loading ? <><Loader2 size={14} className="animate-spin" />Estimating...</> : <><ClipboardList size={14} />Generate Risk Report</>}
            </button>
            {error && <ErrorBox msg={error} />}
          </div>
          {result && !loading && (
            <div className="grid lg:grid-cols-2 gap-4">
              <ResultCard result={result} />
              <ChatBox context={chatCtx} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ImageSection({ type, apiKey }) {
  const [file, setFile]       = useState(null)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const handleRun = async () => {
    if (!file) return; setLoading(true); setResult(null); setError(null)
    try { const { data } = await callAPI[apiKey](file); setResult(data) }
    catch (e) { setError(e.response?.data?.error || e.message || 'Analysis failed.') }
    finally { setLoading(false) }
  }

  const chatCtx = result ? `${type.label} analysis: ${result.prediction} with ${Number(result.confidence ?? 0).toFixed(1)}% confidence.` : null

  return (
    <div className="grid lg:grid-cols-2 gap-4">
      <div className="space-y-3">
        <p className="text-xs" style={{ color: T.text2 }}>{type.hint}</p>
        <DropZone onFile={setFile} accept={type.accept} hint={type.hint} file={file} color={type.color} isCSV={type.id === 'ecg'} />
        <button onClick={handleRun} disabled={!file || loading}
          className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-bold transition-all disabled:cursor-not-allowed disabled:opacity-50 shadow-sm"
          style={{ background: file && !loading ? type.color : T.surface2, color: file && !loading ? '#fff' : T.text3, border: `1.5px solid ${file && !loading ? type.color : T.border}` }}>
          {loading ? <><Loader2 size={14} className="animate-spin" />Analyzing...</> : <><Cpu size={14} />Run Analysis</>}
        </button>
        {error && <ErrorBox msg={error} />}
      </div>
      <div>
        {loading && <LoadingCard color={type.color} icon={type.icon} />}
        {result && !loading && <><ResultCard result={result} /><div className="mt-3"><ChatBox context={chatCtx} /></div></>}
        {!result && !loading && <EmptyCard />}
      </div>
    </div>
  )
}

function ChooseSection() {
  const [selected, setSelected] = useState('breast')
  const type = CANCER_TYPES.find(t => t.id === selected)
  const API_KEY_MAP = { skin: 'skin', blood: 'blood', lung: 'lung', xray: 'xray', ecg: 'ecg' }

  return (
    <section className="space-y-5">
      <div className="flex items-end justify-between">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-1" style={{ color: T.primaryDk }}>Section 01</p>
          <h2 className="text-xl font-bold" style={{ color: T.text }}>Choose Detection Type</h2>
        </div>
        <span className="text-[10px] font-mono" style={{ color: T.text3 }}>{CANCER_TYPES.length} modules</span>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-6 gap-1.5 p-1.5 rounded-2xl shadow-sm"
        style={{ background: T.surface2, border: `1px solid ${T.border}` }}>
        {CANCER_TYPES.map(({ id, label, icon: Icon, color }) => (
          <button key={id} onClick={() => setSelected(id)}
            className="flex flex-col items-center gap-1.5 py-3 px-1 rounded-xl transition-all duration-200"
            style={{
              background: selected === id ? T.surface : 'transparent',
              border: selected === id ? `1.5px solid ${color}40` : '1.5px solid transparent',
              boxShadow: selected === id ? '0 2px 8px rgba(0,0,0,0.08)' : 'none'
            }}>
            <div className="w-9 h-9 rounded-xl flex items-center justify-center transition-all"
              style={{ background: selected === id ? `${color}15` : 'transparent' }}>
              <Icon size={16} style={{ color: selected === id ? color : T.text3 }} />
            </div>
            <span className="text-[10px] font-semibold text-center leading-tight"
              style={{ color: selected === id ? T.text : T.text3 }}>{label}</span>
          </button>
        ))}
      </div>

      <div className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl"
        style={{ background: `${type.color}08`, border: `1.5px solid ${type.color}25` }}>
        <type.icon size={13} style={{ color: type.color }} />
        <span className="text-xs font-semibold" style={{ color: type.color }}>{type.label} Detection</span>
        <span className="text-xs ml-auto" style={{ color: T.text3 }}>
          {selected === 'breast' ? 'Image + Clinical Risk Estimator' : type.hint}
        </span>
      </div>

      <AnimatePresence mode="wait">
        <motion.div key={selected} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
          {selected === 'breast' ? <BreastSection color={type.color} /> : <ImageSection type={type} apiKey={API_KEY_MAP[selected]} />}
        </motion.div>
      </AnimatePresence>

      <div className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ background: '#FFFBEB', border: '1px solid #FDE68A' }}>
        <Shield size={10} style={{ color: '#D97706' }} className="shrink-0" />
        <p className="text-[10px]" style={{ color: '#92400E' }}>Screening support only — not a medical diagnosis. Always consult a healthcare professional.</p>
      </div>
    </section>
  )
}

function UniversalSection() {
  const [file, setFile]       = useState(null)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const handleRun = async () => {
    if (!file) return
    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const { data } = await callAPI.universal(file)
      console.log("UNIVERSAL API RESPONSE:", data)
      setResult(data)
    } catch (e) {
      setError(e.response?.data?.error || e.message || 'Detection failed.')
    } finally {
      setLoading(false)
    }
  }

  const chatCtx = result
    ? `Auto-detected image type: ${result.category_display}. Final prediction: ${result.prediction} with ${Number(result.confidence ?? 0).toFixed(1)}% confidence.`
    : null

  return (
    <section className="space-y-5">
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-widest mb-1" style={{ color: T.primaryDk }}>Section 02</p>
          <h2 className="text-xl font-bold" style={{ color: T.text }}>Universal Detector</h2>
          <p className="text-xs mt-1" style={{ color: T.text2 }}>
            Upload any medical image — AI identifies the image type and analyzes it automatically.
          </p>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full shadow-sm"
          style={{ border: `1.5px solid ${T.primaryDk}30`, background: '#EFF6FF' }}>
          <Zap size={11} style={{ color: T.primaryDk }} />
          <span className="text-[10px] font-bold" style={{ color: T.primaryDk }}>AUTO ROUTE</span>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <div className="space-y-3">
          <p className="text-xs" style={{ color: T.text2 }}>
            Supports breast, skin, blood, lung and chest X-ray images.
          </p>

          <DropZone
            onFile={setFile}
            accept={{ 'image/*': ['.png','.jpg','.jpeg','.bmp','.webp'] }}
            hint="Any medical image — breast, skin, blood, lung, chest"
            file={file}
            color={T.primaryDk}
            isCSV={false}
          />

          <button
            onClick={handleRun}
            disabled={!file || loading}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-bold transition-all disabled:cursor-not-allowed disabled:opacity-50 shadow-sm"
            style={{
              background: file && !loading ? T.primaryDk : T.surface2,
              color: file && !loading ? '#fff' : T.text3,
              border: `1.5px solid ${file && !loading ? T.primaryDk : T.border}`
            }}>
            {loading ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Detecting...
              </>
            ) : (
              <>
                <Zap size={14} />
                Auto-Detect & Analyze
              </>
            )}
          </button>

          {error && <ErrorBox msg={error} />}
        </div>

        <div>
          {loading && (
            <div className="h-full min-h-[200px] rounded-2xl flex flex-col items-center justify-center gap-4 shadow-sm"
              style={{ background: T.surface, border: `1px solid ${T.border}` }}>
              <div className="relative w-12 h-12">
                <div className="absolute inset-0 rounded-full" style={{ border: `2px solid ${T.border}` }} />
                <div className="absolute inset-0 rounded-full border-2 border-t-sky-500 animate-spin" />
                <Zap size={16} className="absolute inset-0 m-auto" style={{ color: T.primaryDk }} />
              </div>
              <p className="text-xs" style={{ color: T.text2 }}>Identifying & routing...</p>
            </div>
          )}

          {result && !loading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">

              <div className="rounded-xl p-4 flex items-start gap-3 shadow-sm"
                style={{ background: '#EFF6FF', border: `1.5px solid #BFDBFE` }}>
                <span className="text-2xl">{result.category_icon || '🧠'}</span>
                <div className="flex-1">
                  <p className="text-[10px] font-bold uppercase tracking-widest mb-1" style={{ color: T.primaryDk }}>
                    Detected Image Type
                  </p>
                  <p className="text-sm font-bold" style={{ color: T.text }}>
                    {result.category_display || 'Unknown Type'}
                  </p>
                  <p className="text-xs mt-1" style={{ color: T.text2 }}>
                    {result.category_description || 'No description available'}
                  </p>
                  <p className="text-[10px] mt-2" style={{ color: T.text3 }}>
                    Router Confidence: {Number(result.router_confidence ?? 0).toFixed(1)}%
                  </p>
                </div>
              </div>

              <ResultCard result={result} />

              {result.routing_note && (
                <div className="rounded-xl p-3" style={{ background: '#F8FAFC', border: `1px solid ${T.border}` }}>
                  <p className="text-[10px] font-bold uppercase tracking-widest mb-1" style={{ color: T.text3 }}>
                    AI Routing Info
                  </p>
                  <p className="text-xs" style={{ color: T.text2 }}>
                    {result.routing_note}
                  </p>
                  {result.pipeline && (
                    <p className="text-[10px] mt-2 font-mono" style={{ color: T.text3 }}>
                      {result.pipeline}
                    </p>
                  )}
                </div>
              )}

              {result.uncertainty_note && (
                <div className="rounded-xl p-3" style={{ background: '#FFFBEB', border: '1px solid #FDE68A' }}>
                  <p className="text-[10px] font-bold uppercase tracking-widest mb-1" style={{ color: '#D97706' }}>
                    Verification Recommended
                  </p>
                  <p className="text-xs" style={{ color: '#92400E' }}>
                    {result.uncertainty_note}
                  </p>
                </div>
              )}

              <ChatBox context={chatCtx} />
            </motion.div>
          )}

          {!result && !loading && <EmptyCard />}
        </div>
      </div>
    </section>
  )
}

export default function App() {
  return (
    <div className="min-h-screen" style={{ background: T.bg }}>
      <div className="fixed top-0 inset-x-0 h-1 pointer-events-none" style={{ background: `linear-gradient(90deg, ${T.primaryDk}, #7C3AED, ${T.success})` }} />

      <header className="sticky top-0 z-50 shadow-sm" style={{ background: 'rgba(255,255,255,0.92)', backdropFilter: 'blur(20px)', borderBottom: `1px solid ${T.border}` }}>
        <div className="max-w-5xl mx-auto px-6 flex items-center justify-between" style={{ height: 56 }}>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl flex items-center justify-center shadow-sm" style={{ background: `linear-gradient(135deg, ${T.primaryDk}, #0369A1)` }}>
              <FlaskConical size={15} className="text-white" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="font-bold text-sm tracking-tight" style={{ color: T.text }}>BioVision AI</span>
              <span className="text-[10px] hidden sm:inline" style={{ color: T.text3 }}>Biomedical Intelligence</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-1">
              {['Breast','Skin','Blood','Lung','X-Ray','ECG'].map(m => (
                <span key={m} className="text-[9px] font-semibold px-2 py-0.5 rounded-md"
                  style={{ background: T.surface2, color: T.text2, border: `1px solid ${T.border}` }}>{m}</span>
              ))}
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full"
              style={{ background: '#F0FDF4', border: '1px solid #BBF7D0' }}>
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[10px] font-semibold" style={{ color: T.success }}>ONLINE</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-6 py-12 space-y-16">
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="text-center space-y-5">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full shadow-sm"
            style={{ background: '#EFF6FF', border: `1.5px solid #BFDBFE` }}>
            <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: T.primaryDk }} />
            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: T.primaryDk }}>Biomedical Image Detection & Feature Extraction</span>
          </div>

          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight leading-tight" style={{ color: T.text }}>
            AI-Powered{' '}
            <span style={{ background: `linear-gradient(135deg, ${T.primaryDk}, #7C3AED)`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Medical</span>
            <br />Image Analysis
          </h1>

          <p className="max-w-md mx-auto text-sm leading-relaxed" style={{ color: T.text2 }}>
            Seven specialized AI models — breast image & clinical risk, skin, blood, lung, chest X-ray and ECG arrhythmia detection.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4 pt-1 text-xs" style={{ color: T.text2 }}>
            {[['#DB2777','ResNet50V2 + Fine-tuning'],['#0284C7','1D CNN ECG'],['#059669','Groq LLaMA3'],['#7C3AED','7 AI Models']].map(([c,l]) => (
              <span key={l} className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ background: c }} />{l}
              </span>
            ))}
          </div>
        </motion.div>

        <div className="h-px" style={{ background: `linear-gradient(90deg, transparent, ${T.border2}, transparent)` }} />
        <ChooseSection />

        <div className="relative">
          <div className="h-px" style={{ background: `linear-gradient(90deg, transparent, ${T.border2}, transparent)` }} />
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="flex items-center gap-2 px-4" style={{ background: T.bg }}>
              <div className="w-1.5 h-1.5 rounded-full" style={{ background: T.primaryDk, opacity: 0.5 }} />
              <span className="text-[10px] font-mono font-bold" style={{ color: T.text3 }}>OR</span>
              <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#7C3AED', opacity: 0.5 }} />
            </div>
          </div>
        </div>

        <UniversalSection />

        <footer className="pb-4 pt-8 text-center space-y-3" style={{ borderTop: `1px solid ${T.border}` }}>
          <div className="flex items-center justify-center gap-2 text-xs" style={{ color: T.text2 }}>
            <Shield size={12} style={{ color: T.text3 }} />
            <span>For educational and research purposes only — not a substitute for professional medical diagnosis</span>
          </div>
          <p className="text-xs font-semibold" style={{ color: T.text3 }}>
            Made by{' '}
            <span style={{ background: `linear-gradient(135deg, ${T.primaryDk}, #7C3AED)`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              DKS
            </span>
          </p>
        </footer>
      </div>
    </div>
  )
}