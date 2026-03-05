# Quantum BTC AI — Chainlink Oracle Predictor

Real-time Bitcoin price predictor powered by **Chainlink on-chain price feeds** and a **CNN-LSTM ensemble model**.

## Price Source

Live BTC/USD price is fetched directly from the Chainlink oracle on Ethereum Mainnet:
- Contract: `0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88`
- Feed: [data.chain.link/streams/btc-usd-cexprice-streams](https://data.chain.link/streams/btc-usd-cexprice-streams)
- Data is read via public Ethereum RPC endpoints (no API key needed)

## Model Architecture

CNN-LSTM hybrid ensemble with 5 signal streams:
1. **EMA Crossover** — EMA(3/8/21) momentum
2. **RSI Momentum** — RSI(14) overbought/oversold
3. **MACD Histogram** — trend acceleration
4. **Bollinger Position** — mean reversion signal
5. **Volume Anomaly** — directional volume divergence

Each stream passes through a calibrated LSTM cell → stacked with learned weights → UP/DOWN/NEUTRAL signal + confidence %.

## Deploy to Vercel

### Option A — Vercel CLI (fastest)
```bash
npm i -g vercel
npm install
vercel
```

### Option B — GitHub + Vercel UI
1. Push this folder to a GitHub repo
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import the repo — Vercel auto-detects Next.js
4. Click **Deploy** — done!

### Option C — One-click (after pushing to GitHub)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

## Run Locally
```bash
npm install
npm run dev
# Open http://localhost:3000
```

## No API Keys Required
The app uses:
- **Chainlink** → public Ethereum RPC endpoints (no key needed)
- **Binance REST API** → public endpoints for 5m OHLCV (no key needed)

## Tech Stack
- **Next.js 14** — React framework
- **ethers.js 5** — Ethereum / Chainlink interaction
- **Binance REST API** — historical 5-min candles for model
- Deployed on **Vercel Edge Network**
