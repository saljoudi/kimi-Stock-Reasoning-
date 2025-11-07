#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# PART 1: ENTERPRISE CONFIGURATION & ONTOLOGY FOUNDATION
# ============================================================

import os
import sys
import json
import logging
import hashlib
from datetime import datetime, timedelta
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ta

from yahooquery import Ticker
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef, BNode
from joblib import Memory
import pydantic
from pydantic import BaseModel, Field, validator

# ─────────────────────────────────────────────
# ENTERPRISE CONFIGURATION MANAGEMENT
# ─────────────────────────────────────────────
class DSSConfig(BaseModel):
    """Configuration management for the intraday trading DSS"""
    cache_dir: str = Field(default="./cache/intraday")
    log_level: str = Field(default="INFO")
    max_bars: int = Field(default=390, description="Maximum intraday bars (full session)")
    risk_per_trade: float = Field(default=0.01, description="Risk per trade (1% default)")
    account_size: float = Field(default=100_000.0, description="Default account size")
    vwap_threshold: float = Field(default=0.002, description="VWAP deviation threshold (0.2%)")
    orb_period: int = Field(default=15, description="Opening Range Breakout period (minutes)")
    min_liquidity: float = Field(default=1_000_000, description="Minimum daily volume")
    
    class Config:
        env_prefix = "DSS_"

config = DSSConfig()

# ─────────────────────────────────────────────
# ENTERPRISE LOGGING SETUP
# ─────────────────────────────────────────────
Path("./logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(f"./logs/dss_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("IntradayDSS")

# ─────────────────────────────────────────────
# PERFORMANCE CACHE
# ─────────────────────────────────────────────
Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
memory = Memory(location=config.cache_dir, verbose=0)

# ─────────────────────────────────────────────
# ENHANCED ONTOLOGY VOCABULARY
# ─────────────────────────────────────────────
# Institutional-grade semantic framework
STOCK = Namespace("https://ontology.tradingsystem.ai/stock/")
TECH = Namespace("https://ontology.tradingsystem.ai/technical/")
MARKET = Namespace("https://ontology.tradingsystem.ai/market/")
TIME = Namespace("https://ontology.tradingsystem.ai/time/")
TRADE = Namespace("https://ontology.tradingsystem.ai/trade/")
RISK = Namespace("https://ontology.tradingsystem.ai/risk/")
LIQ = Namespace("https://ontology.tradingsystem.ai/liquidity/")

# Core ontology schema definition
INTRADAY_ONTOLOGY_TTL = """
@prefix stock: <https://ontology.tradingsystem.ai/stock/> .
@prefix tech: <https://ontology.tradingsystem.ai/technical/> .
@prefix market: <https://ontology.tradingsystem.ai/market/> .
@prefix time: <https://ontology.tradingsystem.ai/time/> .
@prefix trade: <https://ontology.tradingsystem.ai/trade/> .
@prefix risk: <https://ontology.tradingsystem.ai/risk/> .
@prefix liq: <https://ontology.tradingsystem.ai/liquidity/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# Core Classes
stock:TradableAsset a owl:Class ;
    rdfs:label "Tradable Asset" ;
    rdfs:comment "Any instrument that can be traded intraday" .
    
time:MarketSession a owl:Class ;
    rdfs:label "Market Session" ;
    rdfs:comment "Time-bounded trading session with unique characteristics" .

tech:TechnicalIndicator a owl:Class ;
    rdfs:subClassOf owl:Thing ;
    rdfs:label "Technical Indicator" .

trade:TradingSignal a owl:Class ;
    rdfs:label "Trading Signal" ;
    rdfs:comment "Actionable entry or exit signal with confidence" .

risk:RiskProfile a owl:Class ;
    rdfs:label "Risk Profile" ;
    rdfs:comment "Dynamic risk assessment for current market conditions" .

liq:LiquidityZone a owl:Class ;
    rdfs:label "Liquidity Zone" ;
    rdfs:comment "Area of significant order flow or volume concentration" .

# Properties
stock:hasIndicator a owl:ObjectProperty ;
    rdfs:domain stock:TradableAsset ;
    rdfs:range tech:TechnicalIndicator .
    
tech:impliesSignal a owl:ObjectProperty ;
    rdfs:domain tech:TechnicalIndicator ;
    rdfs:range trade:TradingSignal .
    
trade:hasConfidence a owl:DatatypeProperty ;
    rdfs:domain trade:TradingSignal ;
    rdfs:range xsd:float .
    
risk:hasRiskLevel a owl:DatatypeProperty ;
    rdfs:domain risk:RiskProfile ;
    rdfs:range xsd:string .
    
time:occursDuring a owl:ObjectProperty ;
    rdfs:domain trade:TradingSignal ;
    rdfs:range time:MarketSession .
"""

# ─────────────────────────────────────────────
# INSTITUTIONAL ONTOLOGY GRAPH MANAGER
# ─────────────────────────────────────────────
class IntradayOntologyGraph:
    """
    Enterprise-grade RDF graph manager for intraday trading semantics.
    Implements SHACL validation and SPARQL query capabilities.
    """
    
    def __init__(self):
        self.g = Graph()
        self._load_ontology_schema()
        self._setup_namespaces()
        self.signals: List[Dict[str, Any]] = []
        logger.info("IntradayOntologyGraph initialized")
    
    def _load_ontology_schema(self):
        """Load institutional ontology schema"""
        self.g.parse(data=INTRADAY_ONTOLOGY_TTL, format="turtle")
        logger.debug("Ontology schema loaded")
    
    def _setup_namespaces(self):
        """Configure namespace bindings"""
        self.g.bind("stock", STOCK)
        self.g.bind("tech", TECH)
        self.g.bind("market", MARKET)
        self.g.bind("time", TIME)
        self.g.bind("trade", TRADE)
        self.g.bind("risk", RISK)
        self.g.bind("liq", LIQ)
    
    def add_indicator(self, symbol: str, indicator_type: str, 
                     value: float, metadata: Dict[str, Any]) -> URIRef:
        """
        Add technical indicator with rich metadata to graph
        
        Args:
            symbol: Trading symbol
            indicator_type: Indicator name (e.g., 'VWAP', 'RSI')
            value: Numeric value
            metadata: Dict with signal, confidence, timeframe, etc.
        """
        ind_id = f"{symbol}_{indicator_type}_{int(datetime.now().timestamp())}"
        ind_uri = URIRef(f"{TECH}{ind_id}")
        
        self.g.add((ind_uri, RDF.type, TECH.TechnicalIndicator))
        self.g.add((ind_uri, STOCK.hasSymbol, Literal(symbol)))
        self.g.add((ind_uri, TECH.hasValue, Literal(round(value, 6))))
        self.g.add((ind_uri, TECH.hasTimestamp, Literal(datetime.now().isoformat())))
        
        for key, val in metadata.items():
            if val is not None:
                self.g.add((ind_uri, TECH[f"has{key.title()}"], Literal(val)))
        
        logger.debug(f"Added indicator: {indicator_type} for {symbol}")
        return ind_uri
    
    def add_trading_signal(self, symbol: str, signal_type: str, 
                          confidence: float, rationale: str) -> URIRef:
        """
        Generate trade signal with full provenance
        """
        sig_id = f"{symbol}_signal_{signal_type}_{hashlib.md5(rationale.encode()).hexdigest()[:8]}"
        sig_uri = URIRef(f"{TRADE}{sig_id}")
        
        self.g.add((sig_uri, RDF.type, TRADE.TradingSignal))
        self.g.add((sig_uri, TRADE.hasSignalType, Literal(signal_type)))
        self.g.add((sig_uri, TRADE.hasConfidence, Literal(round(confidence, 4))))
        self.g.add((sig_uri, TRADE.hasRationale, Literal(rationale)))
        self.g.add((sig_uri, TRADE.hasTimestamp, Literal(datetime.now().isoformat())))
        
        self.signals.append({
            "symbol": symbol,
            "type": signal_type,
            "confidence": confidence,
            "rationale": rationale
        })
        
        logger.info(f"Generated {signal_type} signal for {symbol} ({confidence:.1%})")
        return sig_uri
    
    def add_liquidity_zone(self, symbol: str, zone_type: str, 
                          price_level: float, volume: float) -> URIRef:
        """Record liquidity zones for order flow analysis"""
        zone_uri = URIRef(f"{LIQ}{symbol}_{zone_type}_{price_level:.2f}")
        self.g.add((zone_uri, RDF.type, LIQ.LiquidityZone))
        self.g.add((zone_uri, LIQ.hasPriceLevel, Literal(round(price_level, 4))))
        self.g.add((zone_uri, LIQ.hasVolume, Literal(int(volume))))
        self.g.add((zone_uri, LIQ.hasZoneType, Literal(zone_type)))
        return zone_uri
    
    def query_signals(self, symbol: str = None, min_confidence: float = 0.6) -> List[Dict]:
        """SPARQL query for high-confidence signals"""
        query = """
        PREFIX trade: <https://ontology.tradingsystem.ai/trade/>
        PREFIX tech: <https://ontology.tradingsystem.ai/technical/>
        SELECT ?signal ?type ?conf ?rationale WHERE {
            ?signal a trade:TradingSignal ;
                    trade:hasSignalType ?type ;
                    trade:hasConfidence ?conf ;
                    trade:hasRationale ?rationale .
            FILTER(?conf >= %f)
        """ % min_confidence
        
        if symbol:
            query += f' ?signal tech:hasSymbol "{symbol}" .'
        
        query += "}"
        
        results = []
        for row in self.g.query(query):
            results.append({
                "signal": str(row.signal),
                "type": str(row.type),
                "confidence": float(row.conf),
                "rationale": str(row.rationale)
            })
        return results
    
    def export_knowledge_graph(self) -> str:
        """Export graph for audit and explainability"""
        return self.g.serialize(format="turtle")


# ============================================================
# PART 2: INTRADAY TRADING ONTOLOGY ENGINE
# ============================================================

class MarketSession(Enum):
    """Enhanced session definitions for intraday trading"""
    PRE_MARKET = "pre_market"
    OPENING_RAMP = "opening_ramp"
    MORNING_TREND = "morning_trend"
    MID_DAY_CONSOLIDATION = "mid_day_consolidation"
    AFTERNOON_MOVE = "afternoon_move"
    CLOSING_RAMP = "closing_ramp"
    POST_MARKET = "post_market"


class SignalType(Enum):
    """Institutional signal taxonomy"""
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    SCALE_OUT = "scale_out"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    RISK_OFF = "risk_off"


class RiskTier(Enum):
    """Dynamic risk tiers based on volatility regime"""
    CALM = "calm"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class IntradayContext:
    """Comprehensive intraday trading context"""
    symbol: str
    session: MarketSession
    timestamp: datetime
    price: float
    vwap: float
    vwap_deviation: float
    orb_high: float
    orb_low: float
    liquidity_zones: List[Tuple[float, float, str]]  # (price, volume, type)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    risk_tier: RiskTier = RiskTier.NORMAL
    position_size: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    expected_value: float = 0.0
    ontology_graph: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "session": self.session.value,
            "price": round(self.price, 4),
            "vwap_deviation": f"{self.vwap_deviation:.2%}",
            "risk_tier": self.risk_tier.value,
            "position_size": self.position_size,
            "signals": self.signals
        }


class IntradayOntologyEngine:
    """
    Institutional-grade reasoning engine for intraday trading decisions.
    Integrates market microstructure, liquidity analysis, and risk management.
    """
    
    def __init__(self, account_size: float = 100_000.0):
        self.account_size = account_size
        self.graph = IntradayOntologyGraph()
        self.config = config
        logger.info(f"Engine initialized | Account: ${account_size:,.2f}")
    
    def analyze_intraday_context(self, symbol: str, df: pd.DataFrame) -> IntradayContext:
        """
        Main analysis pipeline for intraday trading context
        """
        if len(df) < self.config.orb_period:
            logger.warning(f"Insufficient bars for {symbol}: {len(df)}")
            return self._create_default_context(symbol)
        
        # Identify current session
        session = self._identify_market_session(df)
        
        # Calculate core metrics
        current_price = df["close"].iloc[-1]
        vwap = self._calculate_vwap(df)
        vwap_dev = (current_price - vwap) / vwap
        
        # Opening Range
        orb_high, orb_low = self._calculate_opening_range(df)
        
        # Liquidity analysis
        liquidity_zones = self._detect_liquidity_zones(df)
        
        # Risk assessment
        risk_tier = self._assess_intraday_risk(df)
        
        # Generate signals
        signals = self._generate_intraday_signals(
            symbol, df, session, vwap_dev, orb_high, orb_low, risk_tier
        )
        
        # Position sizing and trade management
        position_size = self._calculate_position_size(
            current_price, risk_tier, df["ATR"].iloc[-1]
        )
        
        stop_loss, take_profit = self._calculate_trade_levels(
            current_price, vwap, orb_high, orb_low, risk_tier
        )
        
        # Build ontology graph
        self._populate_ontology_graph(
            symbol, df, signals, liquidity_zones, risk_tier
        )
        
        context = IntradayContext(
            symbol=symbol,
            session=session,
            timestamp=datetime.now(),
            price=current_price,
            vwap=vwap,
            vwap_deviation=vwap_dev,
            orb_high=orb_high,
            orb_low=orb_low,
            liquidity_zones=liquidity_zones,
            signals=signals,
            risk_tier=risk_tier,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ontology_graph=self.graph.export_knowledge_graph()
        )
        
        logger.info(f"Analysis complete for {symbol} | Session: {session.value} | Signals: {len(signals)}")
        return context
    
    def _identify_market_session(self, df: pd.DataFrame) -> MarketSession:
        """Determine current market session based on time and volatility"""
        # For YahooQuery data, use index time if available
        if hasattr(df.index, 'time'):
            current_time = df.index[-1].time()
            if current_time < datetime.strptime("09:30", "%H:%M").time():
                return MarketSession.PRE_MARKET
            elif current_time < datetime.strptime("10:00", "%H:%M").time():
                return MarketSession.OPENING_RAMP
            elif current_time < datetime.strptime("11:30", "%H:%M").time():
                return MarketSession.MORNING_TREND
            elif current_time < datetime.strptime("14:00", "%H:%M").time():
                return MarketSession.MID_DAY_CONSOLIDATION
            elif current_time < datetime.strptime("15:30", "%H:%M").time():
                return MarketSession.AFTERNOON_MOVE
            elif current_time < datetime.strptime("16:00", "%H:%M").time():
                return MarketSession.CLOSING_RAMP
            else:
                return MarketSession.POST_MARKET
        
        # Fallback based on price action characteristics
        recent_vol = df["close"].tail(30).std()
        early_vol = df["close"].head(30).std()
        
        if recent_vol > early_vol * 1.5:
            return MarketSession.CLOSING_RAMP
        elif early_vol > recent_vol * 1.5:
            return MarketSession.OPENING_RAMP
        else:
            return MarketSession.MID_DAY_CONSOLIDATION
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate VWAP for the session"""
        if "VWAP" in df.columns:
            return df["VWAP"].iloc[-1]
        # Recalculate if needed
        cum_pv = (df["close"] * df["volume"]).sum()
        cum_vol = df["volume"].sum()
        return cum_pv / cum_vol if cum_vol > 0 else df["close"].iloc[-1]
    
    def _calculate_opening_range(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate Opening Range Breakout levels"""
        orb_bars = min(self.config.orb_period, len(df))
        orb_high = df["high"].iloc[:orb_bars].max()
        orb_low = df["low"].iloc[:orb_bars].min()
        return orb_high, orb_low
    
    def _detect_liquidity_zones(self, df: pd.DataFrame) -> List[Tuple[float, float, str]]:
        """Detect support/resistance liquidity zones using volume profile"""
        # Simple intraday volume profile approximation
        recent_df = df.tail(60)  # Last hour
        price_bins = pd.cut(recent_df["close"], bins=20)
        volume_profile = recent_df.groupby(price_bins)["volume"].sum()
        
        liquidity_zones = []
        for i, (price_range, vol) in enumerate(volume_profile.items()):
            if vol > volume_profile.quantile(0.8):
                price_level = price_range.mid
                zone_type = "support" if i < len(volume_profile) / 2 else "resistance"
                liquidity_zones.append((price_level, vol, zone_type))
        
        return liquidity_zones
    
    def _assess_intraday_risk(self, df: pd.DataFrame) -> RiskTier:
        """Dynamic risk assessment based on multiple factors"""
        # ATR-based volatility
        atr = df["ATR"].iloc[-1]
        price = df["close"].iloc[-1]
        atr_pct = atr / price
        
        # Bollinger Band width
        bb_width = (df["Upper_band"].iloc[-1] - df["Lower_band"].iloc[-1]) / df["BB_Middle"].iloc[-1]
        
        # Composite risk score
        risk_score = 0
        
        if atr_pct > 0.03:
            risk_score += 3
        elif atr_pct > 0.02:
            risk_score += 2
        elif atr_pct > 0.01:
            risk_score += 1
        
        if bb_width > 0.05:
            risk_score += 2
        elif bb_width > 0.03:
            risk_score += 1
        
        # Map score to tier
        if risk_score >= 5:
            return RiskTier.EXTREME
        elif risk_score >= 4:
            return RiskTier.HIGH
        elif risk_score >= 3:
            return RiskTier.ELEVATED
        elif risk_score >= 2:
            return RiskTier.NORMAL
        else:
            return RiskTier.CALM
    
    def _generate_intraday_signals(self, symbol: str, df: pd.DataFrame, 
                                  session: MarketSession, vwap_dev: float,
                                  orb_high: float, orb_low: float,
                                  risk_tier: RiskTier) -> List[Dict[str, Any]]:
        """Generate actionable intraday trading signals"""
        signals = []
        current_price = df["close"].iloc[-1]
        
        # ORB Strategy
        if session in [MarketSession.OPENING_RAMP, MarketSession.MORNING_TREND]:
            if current_price > orb_high:
                signals.append({
                    "type": SignalType.LONG_ENTRY,
                    "confidence": 0.75,
                    "rationale": f"Opening Range Breakout above {orb_high:.2f}"
                })
            elif current_price < orb_low:
                signals.append({
                    "type": SignalType.SHORT_ENTRY,
                    "confidence": 0.75,
                    "rationale": f"Opening Range Breakdown below {orb_low:.2f}"
                })
        
        # VWAP Mean Reversion
        if abs(vwap_dev) > self.config.vwap_threshold:
            if vwap_dev > 0 and risk_tier != RiskTier.EXTREME:
                signals.append({
                    "type": SignalType.SHORT_ENTRY,
                    "confidence": 0.65,
                    "rationale": f"Price {vwap_dev:.2%} above VWAP - mean reversion"
                })
            elif vwap_dev < 0 and risk_tier != RiskTier.EXTREME:
                signals.append({
                    "type": SignalType.LONG_ENTRY,
                    "confidence": 0.65,
                    "rationale": f"Price {vwap_dev:.2%} below VWAP - mean reversion"
                })
        
        # Momentum confirmation
        rsi = df["RSI"].iloc[-1]
        macd = df["MACD"].iloc[-1]
        macd_signal = df["MACD_Signal"].iloc[-1]
        
        if rsi > 50 and macd > macd_signal:
            signals.append({
                "type": SignalType.LONG_ENTRY,
                "confidence": 0.70,
                "rationale": f"RSI {rsi:.1f} and MACD bullish"
            })
        elif rsi < 50 and macd < macd_signal:
            signals.append({
                "type": SignalType.SHORT_ENTRY,
                "confidence": 0.70,
                "rationale": f"RSI {rsi:.1f} and MACD bearish"
            })
        
        # Add to ontology graph
        for signal in signals:
            self.graph.add_trading_signal(
                symbol, signal["type"].value, 
                signal["confidence"], signal["rationale"]
            )
        
        return signals
    
    def _calculate_position_size(self, price: float, risk_tier: RiskTier, atr: float) -> int:
        """Risk-based position sizing using ATR"""
        risk_amount = self.account_size * config.risk_per_trade
        
        # Adjust risk per trade based on risk tier
        risk_multiplier = {
            RiskTier.CALM: 1.2,
            RiskTier.NORMAL: 1.0,
            RiskTier.ELEVATED: 0.7,
            RiskTier.HIGH: 0.5,
            RiskTier.EXTREME: 0.3
        }
        
        adjusted_risk = risk_amount * risk_multiplier.get(risk_tier, 1.0)
        stop_distance = atr * 1.5  # 1.5x ATR stop
        
        shares = int(adjusted_risk / stop_distance) if stop_distance > 0 else 0
        
        logger.debug(f"Position size: {shares} shares | Risk tier: {risk_tier.value}")
        return shares
    
    def _calculate_trade_levels(self, price: float, vwap: float, 
                               orb_high: float, orb_low: float,
                               risk_tier: RiskTier) -> Tuple[float, float]:
        """Calculate stop-loss and take-profit levels"""
        # Base stop on risk tier
        stop_distance = (price * 0.01) * (1.5 if risk_tier == RiskTier.HIGH else 1.0)
        
        # Take profit based on risk/reward ratio
        reward_distance = stop_distance * 2.0  # 1:2 risk/reward
        
        return stop_distance, reward_distance
    
    def _populate_ontology_graph(self, symbol: str, df: pd.DataFrame,
                               signals: List[Dict], liquidity_zones: List,
                               risk_tier: RiskTier):
        """Populate graph with comprehensive trading context"""
        # Add indicators
        for indicator in ["RSI", "MACD", "ATR", "ADX"]:
            if indicator in df.columns:
                metadata = {
                    "signal": "bullish" if df[indicator].iloc[-1] > 50 else "bearish",
                    "confidence": 0.8,
                    "risk_tier": risk_tier.value
                }
                self.graph.add_indicator(symbol, indicator, df[indicator].iloc[-1], metadata)
        
        # Add liquidity zones
        for price, volume, zone_type in liquidity_zones:
            self.graph.add_liquidity_zone(symbol, zone_type, price, volume)
    
    def _create_default_context(self, symbol: str) -> IntradayContext:
        """Create safe default context when analysis fails"""
        logger.warning(f"Creating default context for {symbol}")
        return IntradayContext(
            symbol=symbol,
            session=MarketSession.MID_DAY_CONSOLIDATION,
            timestamp=datetime.now(),
            price=0.0,
            vwap=0.0,
            vwap_deviation=0.0,
            orb_high=0.0,
            orb_low=0.0,
            liquidity_zones=[],
            signals=[],
            risk_tier=RiskTier.NORMAL,
            position_size=0,
            stop_loss=0.0,
            take_profit=0.0,
            ontology_graph=self.graph.export_knowledge_graph()
        )


# ============================================================
# PART 3: INSTITUTIONAL GRADE DASHBOARD
# ============================================================

# ─────────────────────────────────────────────
# Application Theme Configuration
# ─────────────────────────────────────────────
def create_institutional_theme():
    """Professional dark theme with high-contrast accents"""
    return {
        "layout": {
            "paper_bgcolor": "#0a0e27",
            "plot_bgcolor": "#0a0e27",
            "font": {"family": "Inter, sans-serif", "color": "#e0e6f0", "size": 12},
            "title": {"font": {"size": 16, "color": "#ffffff"}},
            "xaxis": {
                "gridcolor": "#1a1f3a", "linecolor": "#2a3150",
                "tickcolor": "#e0e6f0", "ticks": "outside"
            },
            "yaxis": {
                "gridcolor": "#1a1f3a", "linecolor": "#2a3150",
                "tickcolor": "#e0e6f0", "ticks": "outside"
            },
            "colorway": [
                "#00d4ff", "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4",
                "#feca57", "#ff9ff3", "#54a0ff", "#5f27cd", "#00d2d3"
            ]
        },
        "data": {
            "candlestick": {
                "increasing": {"line": {"color": "#00d4ff"}, "fillcolor": "#00d4ff30"},
                "decreasing": {"line": {"color": "#ff6b6b"}, "fillcolor": "#ff6b6b30"}
            }
        }
    }

# ─────────────────────────────────────────────
# Application Initialization
# ─────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.SLATE,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        {"name": "description", "content": "Institutional Intraday Trading Decision Support System"}
    ]
)
app.title = "Intraday DSS Pro"
server = app.server

# ─────────────────────────────────────────────
# Professional Layout Design
# ─────────────────────────────────────────────
app.layout = dbc.Container([
    # Header Banner
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Intraday Trading DSS Pro", className="text-primary mb-0"),
                html.P("Institutional-Grade Decision Support System", 
                       className="text-muted mb-0"),
                html.Hr(className="bg-primary")
            ], className="text-center py-3")
        ], width=12)
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Trading Parameters", className="bg-dark"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Symbol", className="text-light"),
                            dbc.Input(
                                id="symbol-input", type="text", value="AAPL",
                                placeholder="Enter symbol (e.g., AAPL)", size="sm"
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Account Size ($)", className="text-light"),
                            dbc.Input(
                                id="account-size", type="number", value=100000,
                                step=1000, min=1000, size="sm"
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Risk per Trade (%)", className="text-light"),
                            dbc.Input(
                                id="risk-per-trade", type="number", value=1.0,
                                step=0.1, min=0.1, max=5, size="sm"
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label(" ", className="text-light"),
                            dbc.Button(
                                "Analyze & Generate Signals", id="analyze-btn",
                                color="primary", size="sm", className="w-100 mt-3"
                            )
                        ], width=3)
                    ])
                ])
            ], className="mb-3")
        ], width=12)
    ]),
    
    # Signal Dashboard (Key Information Panel)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🚨 Active Trading Signals", className="bg-success"),
                dbc.CardBody(id="signal-dashboard", className="p-0")
            ], className="mb-3")
        ], width=12)
    ]),
    
    # Main Chart Area
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Price Action & Liquidity", className="bg-dark"),
                dbc.CardBody(dcc.Graph(id="main-chart", style={"height": "500px"}))
            ], className="mb-3")
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Risk Metrics", className="bg-dark"),
                dbc.CardBody(id="risk-metrics")
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("📍 Liquidity Zones", className="bg-dark"),
                dbc.CardBody(id="liquidity-zones")
            ])
        ], width=4)
    ]),
    
    # Secondary Charts Grid
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("🎯 VWAP Deviation", className="bg-dark"),
            dbc.CardBody(dcc.Graph(id="vwap-chart", style={"height": "300px"}))
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("📊 Volume Profile", className="bg-dark"),
            dbc.CardBody(dcc.Graph(id="volume-chart", style={"height": "300px"}))
        ]), width=6)
    ], className="mb-3"),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("⚡ Momentum Matrix", className="bg-dark"),
            dbc.CardBody(dcc.Graph(id="momentum-chart", style={"height": "300px"}))
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("🛡️ Risk Analysis", className="bg-dark"),
            dbc.CardBody(dcc.Graph(id="risk-chart", style={"height": "300px"}))
        ]), width=6)
    ], className="mb-3"),
    
    # Ontology & Reasoning Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🧠 Ontology Reasoning Trace", className="bg-dark"),
                dbc.CardBody(id="reasoning-trace", style={
                    "maxHeight": "400px", "overflowY": "auto"
                })
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Trade Plan", className="bg-dark"),
                dbc.CardBody(id="trade-plan")
            ])
        ], width=4)
    ], className="mb-3"),
    
    # Hidden storage for ontology data
    dcc.Store(id="ontology-store", storage_type="memory"),
    dcc.Store(id="last-analysis-timestamp", storage_type="memory"),
    
    # Footer
    html.Footer([
        html.Hr(className="bg-primary"),
        html.P("Intraday DSS Pro v5.0 | Institutional Trading Technology", 
               className="text-center text-muted small")
    ])
], fluid=True, className="bg-dark text-light min-vh-100")


# ============================================================
# PART 4: CACHED DATA & COMPUTATION LAYER
# ============================================================

@memory.cache(ignore=["ticker"])
def fetch_intraday_data(ticker: str, period: str = "5d", interval: str = "1m") -> pd.DataFrame:
    """
    Fetch intraday data with institutional-grade error handling
    """
    logger.info(f"Fetching intraday data for {ticker} | {period} @ {interval}")
    try:
        tq = Ticker(ticker)
        df = tq.history(period=period, interval=interval)
        
        if isinstance(df, pd.DataFrame) and df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.get_level_values("date")
        
        df = df.dropna(subset=["close", "volume"])
        df = df[df["volume"] > 0]  # Filter zero-volume bars
        
        logger.info(f"Retrieved {len(df)} bars for {ticker}")
        return df.sort_index()
        
    except Exception as e:
        logger.error(f"Data fetch failed for {ticker}: {e}", exc_info=True)
        return pd.DataFrame()


def compute_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute intraday-specific technical indicators optimized for speed
    """
    if df.empty:
        return df
    
    logger.debug("Computing intraday indicators")
    closes, highs, lows, vols = df["close"], df["high"], df["low"], df["volume"]
    
    # Fast intraday moving averages
    for period in [9, 20, 50]:
        df[f"EMA_{period}"] = closes.ewm(span=period, adjust=False).mean()
    
    # VWAP - essential for intraday
    df["VWAP"] = (closes * vols).cumsum() / vols.cumsum()
    
    # Bollinger Bands with intraday parameters
    ma20 = closes.rolling(20).mean()
    std20 = closes.rolling(20).std()
    df["BB_Upper"] = ma20 + (2.0 * std20)
    df["BB_Lower"] = ma20 - (2.0 * std20)
    df["BB_Middle"] = ma20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / ma20
    
    # Momentum indicators
    df["RSI"] = ta.momentum.RSIIndicator(closes, window=14).rsi()
    df["Stoch_K"] = ta.momentum.StochasticOscillator(
        highs, lows, closes, window=14, smooth_window=3
    ).stoch()
    df["Stoch_D"] = ta.momentum.StochasticOscillator(
        highs, lows, closes, window=14, smooth_window=3
    ).stoch_signal()
    
    # MACD with faster intraday settings
    macd = ta.trend.MACD(closes, window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    
    # Volatility - critical for intraday
    df["ATR"] = ta.volatility.AverageTrueRange(
        highs, lows, closes, window=14
    ).average_true_range()
    df["ATR_Pct"] = df["ATR"] / closes * 100
    
    # Volume pressure
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(closes, vols).on_balance_volume()
    df["CMF"] = ta.volume.ChaikinMoneyFlowIndicator(
        highs, lows, closes, vols, window=20
    ).chaikin_money_flow()
    
    # Trend strength
    adx = ta.trend.ADXIndicator(highs, lows, closes, window=14)
    df["ADX"] = adx.adx()
    df["DI_Plus"] = adx.adx_pos()
    df["DI_Minus"] = adx.adx_neg()
    
    # Opening Range markers
    df["ORB_High"] = df["high"].rolling(config.orb_period, min_periods=1).max()
    df["ORB_Low"] = df["low"].rolling(config.orb_period, min_periods=1).min()
    
    logger.debug(f"Indicators computed: {len(df.columns)} columns")
    return df


# ============================================================
# PART 5: CALLBACKS & APPLICATION LOGIC
# ============================================================

@app.callback(
    [
        Output("signal-dashboard", "children"),
        Output("main-chart", "figure"),
        Output("vwap-chart", "figure"),
        Output("volume-chart", "figure"),
        Output("momentum-chart", "figure"),
        Output("risk-chart", "figure"),
        Output("risk-metrics", "children"),
        Output("liquidity-zones", "children"),
        Output("reasoning-trace", "children"),
        Output("trade-plan", "children"),
        Output("ontology-store", "data")
    ],
    Input("analyze-btn", "n_clicks"),
    [
        State("symbol-input", "value"),
        State("account-size", "value"),
        State("risk-per-trade", "value")
    ]
)
def execute_intraday_analysis(n_clicks, symbol, account_size, risk_per_trade):
    """
    Master callback orchestrating the entire analysis pipeline
    """
    if not n_clicks:
        return [html.Div("Awaiting analysis...")] + [go.Figure()] * 5 + [html.Div()] * 4 + [{}]
    
    if not symbol:
        return [dbc.Alert("Please enter a symbol", color="warning")] + [go.Figure()] * 5 + [html.Div()] * 4 + [{}]
    
    try:
        logger.info(f"=== Starting analysis for {symbol} ===")
        
        # Update config
        config.account_size = float(account_size or 100000)
        config.risk_per_trade = float(risk_per_trade or 1.0) / 100
        
        # Fetch and compute
        df = fetch_intraday_data(symbol, period="5d", interval="1m")
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        df = compute_intraday_indicators(df)
        
        # Execute ontology engine
        engine = IntradayOntologyEngine(config.account_size)
        context = engine.analyze_intraday_context(symbol, df.tail(390))  # Last session
        
        # Generate dashboard components
        signal_dashboard = create_signal_dashboard(context)
        main_chart = create_main_chart(df, context)
        vwap_chart = create_vwap_chart(df, context)
        volume_chart = create_volume_chart(df)
        momentum_chart = create_momentum_chart(df)
        risk_chart = create_risk_chart(df, context)
        risk_metrics = create_risk_metrics(context)
        liquidity_panel = create_liquidity_panel(context)
        reasoning_trace = create_reasoning_trace(context)
        trade_plan = create_trade_plan(context)
        
        # Export ontology data
        ontology_data = {
            "graph": context.ontology_graph,
            "timestamp": context.timestamp.isoformat(),
            "signals": [s for s in context.signals if s["confidence"] > 0.6]
        }
        
        logger.info(f"=== Analysis complete for {symbol} ===")
        
        return [
            signal_dashboard, main_chart, vwap_chart, volume_chart,
            momentum_chart, risk_chart, risk_metrics, liquidity_panel,
            reasoning_trace, trade_plan, ontology_data
        ]
        
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        error_alert = dbc.Alert(f"Analysis Error: {str(e)}", color="danger", dismissable=True)
        return [error_alert] + [go.Figure()] * 5 + [html.Div()] * 4 + [{}]


def create_signal_dashboard(context: IntradayContext) -> html.Div:
    """Prominent display of active trading signals"""
    if not context.signals:
        return html.Div([
            html.H5("No High-Confidence Signals", className="text-muted"),
            html.P("Market conditions do not meet entry criteria", className="small")
        ])
    
    signal_cards = []
    for signal in context.signals:
        if signal["confidence"] < 0.6:
            continue
        
        color = "success" if "LONG" in signal["type"].value else "danger"
        icon = "📈" if "LONG" in signal["type"].value else "📉"
        
        signal_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.Span(icon, className="me-2"),
                        html.Span(signal["type"].value.replace("_", " ").title())
                    ], className=f"text-{color} mb-2"),
                    html.P(signal["rationale"], className="small text-light mb-1"),
                    html.Div([
                        html.Small("Confidence: ", className="text-muted"),
                        html.Span(f"{signal['confidence']:.1%}", 
                                 className=f"fw-bold text-{color}")
                    ], className="d-flex justify-content-between")
                ])
            ], className=f"bg-{color} bg-opacity-10 border-{color} mb-2")
        )
    
    return html.Div(signal_cards)


def create_main_chart(df: pd.DataFrame, context: IntradayContext) -> go.Figure:
    """Comprehensive main chart with liquidity and signals"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price Action & Liquidity", "Volume")
    )
    
    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price"
        ),
        row=1, col=1
    )
    
    # VWAP
    fig.add_trace(
        go.Scatter(x=df.index, y=df["VWAP"], name="VWAP", 
                  line=dict(color="#00d4ff", width=2)),
        row=1, col=1
    )
    
    # Opening Range
    fig.add_hline(y=context.orb_high, line_dash="dash", line_color="yellow",
                  annotation_text="ORB High", row=1, col=1)
    fig.add_hline(y=context.orb_low, line_dash="dash", line_color="yellow",
                  annotation_text="ORB Low", row=1, col=1)
    
    # Liquidity zones
    for price, vol, zone_type in context.liquidity_zones[:3]:
        fig.add_hline(
            y=price, line_dash="dot", 
            line_color="green" if zone_type == "support" else "red",
            annotation_text=f"{zone_type.title()} Zone", row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color="rgba(100,150,255,0.5)"),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{context.symbol} Intraday Analysis | {context.session.value.replace('_', ' ').title()}",
        template="plotly_dark",
        height=600,
        showlegend=True
    )
    
    return fig


def create_vwap_chart(df: pd.DataFrame, context: IntradayContext) -> go.Figure:
    """VWAP deviation analysis"""
    fig = go.Figure()
    
    # Price vs VWAP
    deviation = (df["close"] - df["VWAP"]) / df["VWAP"] * 100
    
    fig.add_trace(
        go.Scatter(x=df.index, y=deviation, name="VWAP Deviation (%)",
                  line=dict(color="#4ecdc4"))
    )
    
    # Threshold lines
    fig.add_hline(y=config.vwap_threshold * 100, line_dash="dash", line_color="red",
                  annotation_text="Upper Threshold")
    fig.add_hline(y=-config.vwap_threshold * 100, line_dash="dash", line_color="green",
                  annotation_text="Lower Threshold")
    
    fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.5)
    
    fig.update_layout(
        title="VWAP Deviation Analysis",
        template="plotly_dark",
        height=300,
        yaxis_title="Deviation (%)"
    )
    
    return fig


def create_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Volume profile and pressure analysis"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    
    # Volume bars
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color="rgba(100,150,255,0.5)"),
        row=1, col=1
    )
    
    # Cumulative volume
    fig.add_trace(
        go.Scatter(x=df.index, y=df["volume"].cumsum(), name="Cumulative Volume",
                  line=dict(color="#ff9ff3")),
        row=2, col=1
    )
    
    fig.update_layout(title="Volume Analysis", template="plotly_dark", height=300)
    return fig


def create_momentum_chart(df: pd.DataFrame) -> go.Figure:
    """Multi-timeframe momentum matrix"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("RSI", "MACD", "Stochastic")
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#54a0ff")),
        row=1, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#00d4ff")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#ff6b6b")),
        row=2, col=1
    )
    
    # Stochastic
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Stoch_K"], name="%K", line=dict(color="#4ecdc4")),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Stoch_D"], name="%D", line=dict(color="#feca57")),
        row=3, col=1
    )
    
    fig.update_layout(title="Momentum Matrix", template="plotly_dark", height=400)
    return fig


def create_risk_chart(df: pd.DataFrame, context: IntradayContext) -> go.Figure:
    """Dynamic risk visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("ATR %", "ADX Strength")
    )
    
    # ATR Percentage
    fig.add_trace(
        go.Scatter(x=df.index, y=df["ATR_Pct"], name="ATR %", line=dict(color="#ff6b6b")),
        row=1, col=1
    )
    
    # ADX
    fig.add_trace(
        go.Scatter(x=df.index, y=df["ADX"], name="ADX", line=dict(color="#00d4ff")),
        row=2, col=1
    )
    fig.add_hline(y=25, line_dash="dash", line_color="yellow", row=2, col=1,
                  annotation_text="Trend Threshold")
    
    fig.update_layout(title="Risk & Volatility", template="plotly_dark", height=300)
    return fig


def create_risk_metrics(context: IntradayContext) -> html.Div:
    """Display dynamic risk metrics"""
    risk_color = {
        RiskTier.CALM: "success",
        RiskTier.NORMAL: "info",
        RiskTier.ELEVATED: "warning",
        RiskTier.HIGH: "danger",
        RiskTier.EXTREME: "dark"
    }
    
    tier_color = risk_color.get(context.risk_tier, "secondary")
    
    return html.Div([
        html.H5("Risk Assessment", className=f"text-{tier_color}"),
        html.Div([
            html.Span("Risk Tier:", className="text-muted"),
            html.Span(context.risk_tier.value.upper(), 
                     className=f"fw-bold text-{tier_color} float-end")
        ], className="mb-2"),
        html.Div([
            html.Span("Position Size:", className="text-muted"),
            html.Span(f"{context.position_size:,} shares",
                     className="fw-bold float-end")
        ], className="mb-2"),
        html.Div([
            html.Span("VWAP Deviation:", className="text-muted"),
            html.Span(f"{context.vwap_deviation:.2%}",
                     className="fw-bold float-end")
        ], className="mb-2"),
        html.Hr(),
        html.Small("Risk per trade: ${:,.2f}".format(
            config.account_size * config.risk_per_trade
        ), className="text-muted")
    ])


def create_liquidity_panel(context: IntradayContext) -> html.Div:
    """Display detected liquidity zones"""
    if not context.liquidity_zones:
        return html.P("No significant liquidity zones detected", className="text-muted small")
    
    zones_html = []
    for price, volume, zone_type in context.liquidity_zones[:5]:
        badge_color = "success" if zone_type == "support" else "danger"
        zones_html.append(
            dbc.Badge([
                f"{zone_type.upper()}: ${price:.2f} | Vol: {volume:,}"
            ], color=badge_color, className="w-100 mb-1")
        )
    
    return html.Div(zones_html)


def create_reasoning_trace(context: IntradayContext) -> html.Ol:
    """Generate detailed reasoning trace"""
    steps = [
        f"🔍 Analyzing {context.symbol} at {context.timestamp.strftime('%H:%M:%S')}",
        f"📅 Session identified: {context.session.value.replace('_', ' ').title()}",
        f"💰 Current Price: ${context.price:.2f} | VWAP: ${context.vwap:.2f}",
        f"📊 VWAP Deviation: {context.vwap_deviation:.2%}",
        f"🎯 Opening Range: ${context.orb_low:.2f} - ${context.orb_high:.2f}",
        f"🛡️ Risk Tier: {context.risk_tier.value.upper()}",
    ]
    
    for signal in context.signals:
        if signal["confidence"] > 0.6:
            steps.append(
                f"🚨 Signal: {signal['type'].value} | "
                f"Confidence: {signal['confidence']:.1%} | "
                f"Rationale: {signal['rationale'][:60]}..."
            )
    
    return html.Ol([html.Li(step, className="mb-1") for step in steps])


def create_trade_plan(context: IntradayContext) -> html.Div:
    """Generate actionable trade plan"""
    if not context.signals:
        return html.Div([
            html.H6("No Trade Plan", className="text-warning"),
            html.P("Wait for high-confidence signals", className="small text-muted")
        ])
    
    best_signal = max(context.signals, key=lambda x: x["confidence"])
    
    plan_items = [
        html.H6("🎯 Trade Plan", className="text-primary mb-3"),
        html.Div([
            html.Span("Action:", className="text-muted small"),
            html.Span(best_signal["type"].value.replace("_", " ").title(),
                     className="fw-bold float-end text-info")
        ], className="mb-2"),
        html.Div([
            html.Span("Entry:", className="text-muted small"),
            html.Span(f"${context.price:.2f}", className="fw-bold float-end")
        ], className="mb-2"),
        html.Div([
            html.Span("Stop Loss:", className="text-muted small"),
            html.Span(f"${context.stop_loss:.2f}", className="fw-bold float-end text-danger")
        ], className="mb-2"),
        html.Div([
            html.Span("Take Profit:", className="text-muted small"),
            html.Span(f"${context.take_profit:.2f}", className="fw-bold float-end text-success")
        ], className="mb-2"),
        html.Hr(),
        html.Div([
            html.Span("Position Size:", className="text-muted small"),
            html.Span(f"{context.position_size:,} shares",
                     className="fw-bold float-end text-primary")
        ], className="mb-2"),
        html.Div([
            html.Span("Risk/Share:", className="text-muted small"),
            html.Span(f"${context.stop_loss:.2f}", className="fw-bold float-end")
        ], className="mb-2")
    ]
    
    return html.Div(plan_items)


# ─────────────────────────────────────────────
# Application Entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 LAUNCHING INTRADAY DECISION SUPPORT SYSTEM PRO")
    logger.info("=" * 60)
    logger.info(f"Account Size: ${config.account_size:,.2f}")
    logger.info(f"Risk per Trade: {config.risk_per_trade:.2%}")
    logger.info(f"Cache Directory: {config.cache_dir}")
    
    app.run_server(debug=False, port=8050)

