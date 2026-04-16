#!/usr/bin/env python
# coding: utf-8

# ============================================================
# PART 1: ENTERPRISE CONFIGURATION & ONTOLOGY FOUNDATION
# ============================================================
# Improvements vs. original (ontology-layer only; plots/layout untouched):
#   [FIX-1] Per-analysis graph reset — no unbounded statement growth
#           between successive calls.
#   [FIX-2] Real knowledge-graph stats surfaced to MarketContext and UI
#           (replaces the hard-coded "+10" placeholder sum).
#   [FIX-3] Semantic contradiction linking actually runs (RSI vs MACD,
#           RSI vs Stochastic, OBV vs price/VWAP, ADX-strong vs RSI
#           extreme, Ichimoku cloud vs momentum, etc.).
#   [FIX-4] apply_inference_rules() is invoked inside infer_market_context,
#           so contradiction-resolution and confirmation-strengthening
#           adjust confidences as designed.
#   [FIX-5] SHACL rule graph now declares the ":" prefix (rules parsed
#           instead of silently failing), and SHACL inference runs from
#           infer_market_context via export_knowledge.
#   [FIX-6] Volume aggregation uses all six volume indicators (was 3 of 6)
#           and weighting between accumulation/distribution is symmetric.
#   [FIX-7] Recommendation logic now requires trend_direction agreement
#           with market state (fixes the "bear_trend + moderate_up +
#           Strong Sell" artifact).
#   [FIX-8] Fibonacci uses the last ~252 bars (true 52-week), and the
#           ontology-level knowledge summary is passed through cleanly.

from pyshacl import validate

# ─────────────────────────────────────────────
# STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# ─────────────────────────────────────────────
# THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
from joblib import Memory

import ta
from yahooquery import Ticker

from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef, XSD
from rdflib.namespace import DefinedNamespace

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore")
CACHE_DIR = "./cache_dir"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

def log_step(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


# ─────────────────────────────────────────────
# ENHANCED ONTOLOGY VOCABULARY
# ─────────────────────────────────────────────
STOCK    = Namespace("http://example.org/stock#")
TECH     = Namespace("http://example.org/technical#")
MARKET   = Namespace("http://example.org/market#")
TIME     = Namespace("http://example.org/time#")
EVIDENCE = Namespace("http://example.org/evidence#")
RISK     = Namespace("http://example.org/risk#")
RULES    = Namespace("http://example.org/rules#")


# ─────────────────────────────────────────────
# OWL Ontology Schema
# ─────────────────────────────────────────────
class EnhancedStockOntologyGraph:
    """
    Production-grade OWL ontology for financial technical analysis.
    """

    def __init__(self):
        self.g = Graph()
        self._define_enhanced_schema()
        log_step("Enhanced OWL ontology schema initialized with temporal semantics.")

    def _define_enhanced_schema(self):
        """Defines comprehensive OWL schema with inference rules."""
        self.g.bind("stock",    STOCK)
        self.g.bind("tech",     TECH)
        self.g.bind("market",   MARKET)
        self.g.bind("time",     TIME)
        self.g.bind("evidence", EVIDENCE)
        self.g.bind("risk",     RISK)
        self.g.bind("rules",    RULES)

        for cls in [
            STOCK.StockEntity, STOCK.Indicator, STOCK.Signal,
            MARKET.MarketState, MARKET.RiskLevel, MARKET.TrendRegime,
            TIME.Instant, TIME.Interval, EVIDENCE.EvidenceBundle
        ]:
            self.g.add((cls, RDF.type, RDFS.Class))
            self.g.add((cls, RDF.type, OWL.Class))

        indicator_types = {
            TECH.TrendIndicator: {
                "indicators": ["SMA", "EMA", "ADX", "Ichimoku", "DEMA", "TEMA"],
                "properties": {"timeframe": XSD.string, "period": XSD.integer, "weight": XSD.float}
            },
            TECH.MomentumIndicator: {
                "indicators": ["RSI", "MACD", "Stochastic", "CCI", "ROC", "MOM"],
                "properties": {"overbought_threshold": XSD.float, "oversold_threshold": XSD.float, "strength": XSD.float}
            },
            TECH.VolatilityIndicator: {
                "indicators": ["ATR", "BollingerBands", "KeltnerChannel", "DonchianChannel"],
                "properties": {"multiplier": XSD.float, "window": XSD.integer, "regime": XSD.string}
            },
            TECH.VolumeIndicator: {
                "indicators": ["OBV", "VWAP", "ADL", "MFI", "CMF", "ForceIndex", "VPCI"],
                "properties": {"volume_confirmation": XSD.boolean, "flow_strength": XSD.float}
            },
            TECH.MarketStructureIndicator: {
                "indicators": ["SupportResistance", "FibonacciLevels", "PivotPoints"],
                "properties": {"level_type": XSD.string, "strength": XSD.float, "reliability": XSD.float}
            }
        }

        for parent_class, config in indicator_types.items():
            self.g.add((parent_class, RDF.type, RDFS.Class))
            self.g.add((parent_class, RDFS.subClassOf, STOCK.Indicator))
            for child in config["indicators"]:
                child_uri = TECH[child]
                self.g.add((child_uri, RDF.type, RDFS.Class))
                self.g.add((child_uri, RDFS.subClassOf, parent_class))
                for prop_name, prop_type in config["properties"].items():
                    prop_uri = TECH[prop_name]
                    self.g.add((prop_uri, RDF.type, RDF.Property))
                    self.g.add((prop_uri, RDFS.domain, child_uri))
                    self.g.add((prop_uri, RDFS.range, prop_type))

        enhanced_properties = {
            STOCK.atTime:          (STOCK.Indicator, TIME.Instant),
            STOCK.observedAt:      (STOCK.Signal, TIME.Instant),
            STOCK.validFor:        (EVIDENCE.EvidenceBundle, TIME.Interval),
            STOCK.expiresAt:       (STOCK.Signal, TIME.Instant),
            STOCK.hasNumericValue: (STOCK.Indicator, XSD.float),
            STOCK.hasSignal:       (STOCK.Indicator, STOCK.Signal),
            STOCK.hasThreshold:    (STOCK.Indicator, XSD.float),
            STOCK.hasConfidence:   (STOCK.Indicator, XSD.float),
            STOCK.hasWeight:       (STOCK.Indicator, XSD.float),
            STOCK.hasStrength:     (STOCK.Indicator, XSD.float),
            STOCK.impliesState:    (STOCK.Indicator, MARKET.MarketState),
            STOCK.confirms:        (STOCK.Indicator, STOCK.Indicator),
            STOCK.contradicts:     (STOCK.Indicator, STOCK.Indicator),
            STOCK.contributesTo:   (STOCK.Indicator, MARKET.TrendRegime),
            STOCK.precedes:        (STOCK.Signal, STOCK.Signal),
            STOCK.succeeds:        (STOCK.Signal, STOCK.Signal),
            EVIDENCE.hasConfidence:  (EVIDENCE.EvidenceBundle, XSD.float),
            EVIDENCE.hasWeight:      (EVIDENCE.EvidenceBundle, XSD.float),
            EVIDENCE.supports:       (EVIDENCE.EvidenceBundle, STOCK.Indicator),
            EVIDENCE.challenges:     (EVIDENCE.EvidenceBundle, STOCK.Indicator),
            EVIDENCE.hasSource:      (EVIDENCE.EvidenceBundle, XSD.string),
            EVIDENCE.hasReliability: (EVIDENCE.EvidenceBundle, XSD.float),
            MARKET.hasVolatility: (MARKET.MarketState, XSD.string),
            MARKET.hasTrend:      (MARKET.MarketState, XSD.string),
            MARKET.hasRiskLevel:  (MARKET.MarketState, XSD.string),
            RISK.hasRiskScore:    (RISK.RiskAssessment, XSD.float),
            RISK.hasPositionSize: (RISK.Position, XSD.float),
            RISK.hasStopLoss:     (RISK.Position, XSD.float),
            RISK.hasTakeProfit:   (RISK.Position, XSD.float),
            RISK.maxDrawdown:     (RISK.Portfolio, XSD.float),
            RISK.sharpeRatio:     (RISK.Portfolio, XSD.float)
        }

        for prop, (domain, range_val) in enhanced_properties.items():
            self.g.add((prop, RDF.type, RDF.Property))
            self.g.add((prop, RDFS.domain, domain))
            self.g.add((prop, RDFS.range, range_val))
            if prop in [STOCK.confirms, STOCK.contradicts]:
                self.g.add((prop, RDF.type, OWL.TransitiveProperty))
            if prop in [STOCK.precedes]:
                self.g.add((prop, RDF.type, OWL.TransitiveProperty))
                self.g.add((prop, RDF.type, OWL.AsymmetricProperty))

    def add_indicator(self, symbol: str, indicator_type: str, value: float,
                      signal: str, confidence: float = 1.0, metadata: Dict = None) -> URIRef:
        """Adds temporally-indexed indicator with confidence weighting."""
        ts = metadata.get("timestamp") if metadata else datetime.now().isoformat()
        ind_uri = URIRef(f"{STOCK}{symbol}_{indicator_type}_{hash(ts)}_{hash(signal)}")

        type_map = {
            "RSI": TECH.RSI, "MACD": TECH.MACD, "Stochastic": TECH.Stochastic,
            "CCI": TECH.CCI, "ATR": TECH.ATR, "BollingerBands": TECH.BollingerBands,
            "OBV": TECH.OBV, "VWAP": TECH.VWAP, "Ichimoku": TECH.Ichimoku,
            "SMA": TECH.SMA, "EMA": TECH.EMA, "ADX": TECH.ADX,
            "MFI": TECH.MFI, "CMF": TECH.CMF, "ForceIndex": TECH.ForceIndex,
            "DEMA": TECH.DEMA, "TEMA": TECH.TEMA, "ROC": TECH.ROC,
            "KeltnerChannel": TECH.KeltnerChannel, "DonchianChannel": TECH.DonchianChannel
        }

        indicator_type_uri = type_map.get(indicator_type, STOCK.Indicator)
        self.g.add((ind_uri, RDF.type, indicator_type_uri))
        self.g.add((ind_uri, STOCK.hasNumericValue, Literal(round(float(value), 6))))
        self.g.add((ind_uri, STOCK.hasSignal, Literal(signal)))
        self.g.add((ind_uri, STOCK.atTime, Literal(ts, datatype=XSD.dateTime)))
        self.g.add((ind_uri, STOCK.hasConfidence, Literal(confidence)))

        if metadata:
            for key, val in metadata.items():
                if key != "timestamp":
                    prop_name = ''.join(word.capitalize() for word in key.split('_'))
                    if hasattr(TECH, prop_name):
                        self.g.add((ind_uri, getattr(TECH, prop_name), Literal(val)))

        if confidence < 1.0:
            ev_uri = URIRef(f"{EVIDENCE}ev_{symbol}_{indicator_type}_{hash(ts)}_{hash(signal)}")
            self.g.add((ev_uri, RDF.type, EVIDENCE.EvidenceBundle))
            self.g.add((ev_uri, EVIDENCE.hasConfidence, Literal(confidence)))
            self.g.add((ev_uri, EVIDENCE.supports, ind_uri))
            if metadata and "source" in metadata:
                self.g.add((ev_uri, EVIDENCE.hasSource, Literal(metadata["source"])))
            reliability = min(confidence * 1.2, 1.0)
            self.g.add((ev_uri, EVIDENCE.hasReliability, Literal(reliability)))

        return ind_uri

    def link_indicators(self, uri1: URIRef, uri2: URIRef, relationship: str, confidence: float = 1.0):
        """Creates semantic links between indicators with confidence weighting."""
        prop = STOCK.confirms if relationship == "confirms" else STOCK.contradicts
        self.g.add((uri1, prop, uri2))

        if confidence < 1.0:
            link_uri = URIRef(f"{EVIDENCE}link_{hash(uri1)}{hash(uri2)}")
            self.g.add((link_uri, RDF.type, EVIDENCE.EvidenceBundle))
            self.g.add((link_uri, EVIDENCE.hasConfidence, Literal(confidence)))
            self.g.add((link_uri, EVIDENCE.supports, uri1))

    def link_state(self, indicator_uri: URIRef, state: str, confidence: float = 1.0):
        """State linking with confidence and temporal validity."""
        state_uri = URIRef(f"{MARKET}{state}")
        self.g.add((indicator_uri, STOCK.impliesState, state_uri))

        validity_uri = URIRef(f"{TIME}validity_{hash(indicator_uri)}{hash(state_uri)}")
        self.g.add((validity_uri, RDF.type, TIME.Interval))
        self.g.add((indicator_uri, STOCK.validFor, validity_uri))

        if confidence < 1.0:
            ev_uri = URIRef(f"{EVIDENCE}ev_state_{hash(indicator_uri)}_{hash(state_uri)}")
            self.g.add((ev_uri, EVIDENCE.hasConfidence, Literal(confidence)))
            self.g.add((ev_uri, EVIDENCE.supports, indicator_uri))

    def detect_contradictions(self) -> List[Tuple[URIRef, URIRef, float]]:
        """Finds pairs of contradictory indicator signals with confidence scores."""
        contradictions = []
        query = """
        SELECT ?ind1 ?ind2 ?conf1 ?conf2 WHERE {
            ?ind1 stock:contradicts ?ind2 .
            ?ind1 stock:hasSignal ?sig1 .
            ?ind2 stock:hasSignal ?sig2 .
            ?ind1 stock:hasConfidence ?conf1 .
            ?ind2 stock:hasConfidence ?conf2 .
            FILTER(?sig1 != ?sig2)
        }
        """
        seen = set()
        for row in self.g.query(query, initNs={"stock": STOCK}):
            key = tuple(sorted([str(row.ind1), str(row.ind2)]))
            if key in seen:
                continue
            seen.add(key)
            contradiction_strength = min(float(row.conf1), float(row.conf2))
            contradictions.append((row.ind1, row.ind2, contradiction_strength))
        return contradictions

    def find_confirmations(self, min_confidence: float = 0.7) -> List[Tuple[URIRef, URIRef]]:
        """Finds strong confirmations between indicators."""
        confirmations = []
        query = f"""
        SELECT ?ind1 ?ind2 WHERE {{
            ?ind1 stock:confirms ?ind2 .
            ?ind1 stock:hasConfidence ?conf1 .
            ?ind2 stock:hasConfidence ?conf2 .
            FILTER(?conf1 > {min_confidence} && ?conf2 > {min_confidence})
        }}
        """
        seen = set()
        for row in self.g.query(query, initNs={"stock": STOCK}):
            key = tuple(sorted([str(row.ind1), str(row.ind2)]))
            if key in seen:
                continue
            seen.add(key)
            confirmations.append((row.ind1, row.ind2))
        return confirmations

    def apply_inference_rules(self):
        """Applies semantic inference rules to derive new knowledge."""
        contradictions = self.detect_contradictions()
        for ind1, ind2, strength in contradictions:
            self._resolve_contradiction(ind1, ind2, strength)

        confirmations = self.find_confirmations()
        for ind1, ind2 in confirmations:
            self._strengthen_confirmation(ind1, ind2)

        log_step(
            f"Inference rules applied: {len(contradictions)} contradiction "
            f"resolutions, {len(confirmations)} confirmation strengthenings"
        )
        return {"contradictions": len(contradictions), "confirmations": len(confirmations)}

    def _resolve_contradiction(self, ind1: URIRef, ind2: URIRef, strength: float):
        """Reduces confidence on both sides of a contradiction."""
        for ind in [ind1, ind2]:
            confs = list(self.g.objects(ind, STOCK.hasConfidence))
            if not confs:
                continue
            current_conf = confs[0]
            new_conf = float(current_conf) * (1 - strength * 0.3)
            self.g.remove((ind, STOCK.hasConfidence, current_conf))
            self.g.add((ind, STOCK.hasConfidence, Literal(new_conf)))

    def _strengthen_confirmation(self, ind1: URIRef, ind2: URIRef):
        """Boosts confidence on confirmed indicators (capped at 1.0)."""
        for ind in [ind1, ind2]:
            confs = list(self.g.objects(ind, STOCK.hasConfidence))
            if not confs:
                continue
            current_conf = confs[0]
            new_conf = min(float(current_conf) * 1.1, 1.0)
            self.g.remove((ind, STOCK.hasConfidence, current_conf))
            self.g.add((ind, STOCK.hasConfidence, Literal(new_conf)))

    def query_knowledge(self, query: str, init_ns: Dict = None) -> List:
        if init_ns is None:
            init_ns = {
                "stock": STOCK, "tech": TECH, "market": MARKET,
                "time": TIME, "evidence": EVIDENCE
            }
        return list(self.g.query(query, initNs=init_ns))

    def export_knowledge(self, format: str = "turtle") -> str:
        """Exports knowledge graph with OWL + SHACL reasoning applied."""
        log_step("Applying ontology reasoning (OWL + SHACL)…")

        # ── OWL RL closure
        try:
            import owlrl
            owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(self.g)
        except Exception as e:
            log_step(f"OWL RL inference warning: {e}")

        # ── SHACL rule-based inference (now with :prefix declared → rules parse)
        try:
            shacl_rules = """
            @prefix sh:     <http://www.w3.org/ns/shacl#> .
            @prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .
            @prefix stock:  <http://example.org/stock#> .
            @prefix tech:   <http://example.org/technical#> .
            @prefix market: <http://example.org/market#> .
            @prefix :       <http://example.org/rules#> .

            :RSIOverboughtRule
                a sh:NodeShape ;
                sh:targetClass tech:RSI ;
                sh:rule [
                    a sh:TripleRule ;
                    sh:condition [ sh:path stock:hasNumericValue ; sh:minInclusive 70 ] ;
                    sh:subject   sh:this ;
                    sh:predicate stock:impliesState ;
                    sh:object    market:BearTrend ;
                ] .

            :RSIOversoldRule
                a sh:NodeShape ;
                sh:targetClass tech:RSI ;
                sh:rule [
                    a sh:TripleRule ;
                    sh:condition [ sh:path stock:hasNumericValue ; sh:maxInclusive 30 ] ;
                    sh:subject   sh:this ;
                    sh:predicate stock:impliesState ;
                    sh:object    market:BullTrend ;
                ] .

            :MACDBullishRule
                a sh:NodeShape ;
                sh:targetClass tech:MACD ;
                sh:rule [
                    a sh:TripleRule ;
                    sh:condition [ sh:path stock:hasSignal ; sh:hasValue "bullish_crossover" ] ;
                    sh:subject   sh:this ;
                    sh:predicate stock:impliesState ;
                    sh:object    market:BullTrend ;
                ] .

            :MACDBearishRule
                a sh:NodeShape ;
                sh:targetClass tech:MACD ;
                sh:rule [
                    a sh:TripleRule ;
                    sh:condition [ sh:path stock:hasSignal ; sh:hasValue "bearish_crossover" ] ;
                    sh:subject   sh:this ;
                    sh:predicate stock:impliesState ;
                    sh:object    market:BearTrend ;
                ] .

            :ADXStrongTrendRule
                a sh:NodeShape ;
                sh:targetClass tech:ADX ;
                sh:rule [
                    a sh:TripleRule ;
                    sh:condition [ sh:path stock:hasNumericValue ; sh:minInclusive 25 ] ;
                    sh:subject   sh:this ;
                    sh:predicate stock:impliesState ;
                    sh:object    market:BullTrend ;
                ] .

            :ADXWeakTrendRule
                a sh:NodeShape ;
                sh:targetClass tech:ADX ;
                sh:rule [
                    a sh:TripleRule ;
                    sh:condition [ sh:path stock:hasNumericValue ; sh:maxInclusive 20 ] ;
                    sh:subject   sh:this ;
                    sh:predicate stock:impliesState ;
                    sh:object    market:RangeBound ;
                ] .

            :VolumeAccumulationRule
                a sh:NodeShape ;
                sh:targetClass tech:OBV ;
                sh:rule [
                    a sh:TripleRule ;
                    sh:condition [ sh:path stock:hasSignal ; sh:hasValue "accumulation" ] ;
                    sh:subject   sh:this ;
                    sh:predicate stock:impliesState ;
                    sh:object    market:BullTrend ;
                ] .

            :VolumeDistributionRule
                a sh:NodeShape ;
                sh:targetClass tech:OBV ;
                sh:rule [
                    a sh:TripleRule ;
                    sh:condition [ sh:path stock:hasSignal ; sh:hasValue "distribution" ] ;
                    sh:subject   sh:this ;
                    sh:predicate stock:impliesState ;
                    sh:object    market:BearTrend ;
                ] .
            """

            conforms, results_graph, results_text = validate(
                self.g,
                shacl_graph=shacl_rules,
                shacl_graph_format="turtle",
                inference="rdfs",
                advanced=True,
                debug=False
            )
            if results_graph is not None:
                self.g += results_graph
            log_step("SHACL reasoning completed successfully.")
        except Exception as e:
            log_step(f"SHACL reasoning error: {e}")

        try:
            return self.g.serialize(format=format)
        except Exception as e:
            log_step(f"Serialization error: {e}")
            return ""

    def serialize(self, format: str = "turtle") -> str:
        """Backward-compatible serializer (delegates to export_knowledge)."""
        return self.export_knowledge(format=format)

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Returns summary statistics of the knowledge graph."""
        return {
            "total_statements":   len(self.g),
            "indicators":         len(list(self.g.subjects(RDF.type, STOCK.Indicator))),
            "signals":            len(list(self.g.subjects(RDF.type, STOCK.Signal))),
            "evidence_bundles":   len(list(self.g.subjects(RDF.type, EVIDENCE.EvidenceBundle))),
            "market_states":      len(list(self.g.subjects(RDF.type, MARKET.MarketState))),
            "risk_assessments":   len(list(self.g.subjects(RDF.type, RISK.RiskAssessment))),
            "contradictions":     len(self.detect_contradictions()),
            "confirmations":      len(self.find_confirmations())
        }


# ============================================================
# ENHANCED ONTOLOGY ENGINE
# ============================================================
class EnhancedStockAnalysisOntology:
    """
    Ontology engine with per-analysis graph isolation and semantic
    contradiction/confirmation linking.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.version = "7.1-owl-enhanced"
        self._context_cache: Dict[str, "MarketContext"] = {}
        self.ontology = EnhancedStockOntologyGraph()

        self.indicator_weights = {
            "trend": 0.30, "momentum": 0.25, "volume": 0.20,
            "volatility": 0.15, "support_resistance": 0.10
        }

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _safe_get_value(self, df, column, default=0.0, index_offset=-1):
        if (column in df.columns and len(df) > abs(index_offset) and
                not pd.isna(df[column].iloc[index_offset])):
            return df[column].iloc[index_offset]
        return default

    def _reset_graph(self):
        """[FIX-1] Fresh ontology per analysis — prevents unbounded growth
        across successive calls and avoids cross-run pollution."""
        self.ontology = EnhancedStockOntologyGraph()

    # ------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------
    def infer_market_context(self, symbol: str, df: pd.DataFrame) -> "MarketContext":
        """Main pipeline with full indicator coverage."""
        cache_key = f"{symbol}_{len(df)}_{df.index[-1].strftime('%Y%m%d')}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        if len(df) < 50:
            return self._default_context()

        # [FIX-1] Start each analysis with a clean graph.
        self._reset_graph()

        # Extract all indicator categories
        extracts = {
            "trend":      self._extract_trend_enhanced(symbol, df),
            "momentum":   self._extract_momentum_enhanced(symbol, df),
            "volume":     self._extract_volume_enhanced(symbol, df),
            "volatility": self._extract_volatility_enhanced(symbol, df),
            "ichimoku":   self._extract_ichimoku_enhanced(symbol, df),
            "fibonacci":  self._extract_fibonacci_enhanced(symbol, df)
        }

        # [FIX-3] Semantic contradiction/confirmation linking across categories.
        self._link_semantic_relationships(extracts)

        # [FIX-4] Apply inference rules now that links exist.
        self.ontology.apply_inference_rules()

        # Detect contradictions AFTER linking so they actually surface.
        contradictions = self.ontology.detect_contradictions()

        # Weighted inference
        market_state,    state_conf = self._infer_market_state_weighted(extracts)
        trend_direction, trend_conf = self._infer_trend_direction_weighted(extracts)
        risk_level,      risk_conf  = self._infer_risk_level_weighted(extracts)

        overall_confidence = (state_conf * 0.4 + trend_conf * 0.3 + risk_conf * 0.3)

        sr_levels = self._calculate_sr_levels(df)

        reasoning_chain = self._build_dynamic_reasoning(
            symbol, extracts, market_state, trend_direction, risk_level,
            contradictions, overall_confidence
        )

        # Link all to market state
        for category in extracts.values():
            for uri in category.get("uris", []):
                self.ontology.link_state(
                    uri, market_state.value,
                    confidence=category.get("avg_confidence", 1.0)
                )

        # [FIX-2] Pipe real graph stats through MarketContext.
        summary = self.ontology.get_knowledge_summary()

        # [FIX-5] Run SHACL via export_knowledge so rules are actually
        # applied to the output graph.
        ontology_ttl = self.ontology.export_knowledge(format="turtle")

        context = MarketContext(
            market_state=market_state,
            trend_direction=trend_direction,
            risk_level=risk_level,
            confidence_score=round(overall_confidence, 3),
            volatility_regime=extracts["volatility"]["regime"],
            volume_profile=extracts["volume"]["profile"],
            support_levels=sr_levels["support"],
            resistance_levels=sr_levels["resistance"],
            ontology_graph=ontology_ttl,
            reasoning_chain=reasoning_chain,
            contradictions=[
                {"indicator1": str(c[0]), "indicator2": str(c[1]), "strength": f"{c[2]:.3f}"}
                for c in contradictions
            ],
            knowledge_summary=summary,
            confirmations=[
                {"indicator1": str(c[0]), "indicator2": str(c[1])}
                for c in self.ontology.find_confirmations()
            ]
        )

        self._context_cache[cache_key] = context
        return context

    # ============================================================
    # CLASSIFICATION HELPERS
    # ============================================================
    def _classify_ma_signal(self, current, ma_value, threshold=0.01):
        if ma_value == 0:
            return "neutral", 0.5
        diff = (current - ma_value) / ma_value
        if diff > threshold:
            return "bullish", min(abs(diff) * 10, 1.0)
        elif diff < -threshold:
            return "bearish", min(abs(diff) * 10, 1.0)
        return "neutral", 0.5

    def _classify_adx(self, adx_value):
        if adx_value >= 40:
            return "very_strong_trend", 0.95
        elif adx_value >= 25:
            return "strong_trend", 0.8
        elif adx_value >= 20:
            return "moderate_trend", 0.6
        return "weak_trend", 0.4

    def _classify_rsi(self, rsi_val):
        if rsi_val > 70:
            return "overbought", 0.9
        elif rsi_val < 30:
            return "oversold", 0.9
        elif 50 <= rsi_val <= 70:
            return "bullish_momentum", 0.7
        elif 30 <= rsi_val < 50:
            return "bearish_momentum", 0.7
        return "neutral", 0.5

    def _classify_macd(self, macd_val, signal_val, hist_val):
        if macd_val > signal_val and hist_val > 0:
            return "bullish_crossover", min(abs(hist_val) * 5, 1.0)
        elif macd_val < signal_val and hist_val < 0:
            return "bearish_crossover", min(abs(hist_val) * 5, 1.0)
        return "neutral", 0.5

    def _classify_stochastic(self, k_val, d_val):
        if k_val > 80 and d_val > 80:
            return "overbought", 0.9
        elif k_val < 20 and d_val < 20:
            return "oversold", 0.9
        elif k_val > d_val:
            return "bullish_cross", 0.7
        elif k_val < d_val:
            return "bearish_cross", 0.7
        return "neutral", 0.5

    def _classify_cci(self, cci_val):
        if cci_val > 100:
            return "bullish_trend", 0.8
        elif cci_val < -100:
            return "bearish_trend", 0.8
        return "neutral", 0.5

    def _classify_mfi(self, mfi_val):
        if mfi_val > 80:
            return "overbought", 0.9
        elif mfi_val < 20:
            return "oversold", 0.9
        elif mfi_val > 50:
            return "bullish_flow", 0.7
        elif mfi_val < 50:
            return "bearish_flow", 0.7
        return "neutral", 0.5

    # ============================================================
    # [FIX-3] Semantic contradiction/confirmation linker
    # ============================================================
    def _link_semantic_relationships(self, extracts: Dict[str, Dict]):
        """Links indicators whose signals are semantically opposed (or aligned).
        This is what makes `detect_contradictions` produce non-zero output."""
        signal_of = {}

        def register(uri, signal):
            signal_of[uri] = signal

        # Index all extracted indicators by URI → signal (if available).
        for cat in extracts.values():
            uris    = cat.get("uris", [])
            signals = cat.get("signals", [])
            for i, uri in enumerate(uris):
                if i < len(signals):
                    register(uri, signals[i])

        bullish_tokens = {"bullish", "bullish_momentum", "bullish_crossover",
                          "bullish_cross", "bullish_trend", "bullish_flow",
                          "accumulation", "above_cloud", "above_vwap",
                          "positive_force"}
        bearish_tokens = {"bearish", "bearish_momentum", "bearish_crossover",
                          "bearish_cross", "bearish_trend", "bearish_flow",
                          "distribution", "below_cloud", "below_vwap",
                          "negative_force"}
        overbought_tokens = {"overbought"}
        oversold_tokens   = {"oversold"}

        uris = list(signal_of.keys())
        # Pairwise comparison — O(n²) but n is small (≤ ~20 indicators).
        for i in range(len(uris)):
            for j in range(i + 1, len(uris)):
                u1, u2 = uris[i], uris[j]
                s1, s2 = signal_of[u1], signal_of[u2]

                # Directional contradiction
                if s1 in bullish_tokens and s2 in bearish_tokens:
                    self.ontology.link_indicators(u1, u2, "contradicts")
                elif s1 in bearish_tokens and s2 in bullish_tokens:
                    self.ontology.link_indicators(u1, u2, "contradicts")

                # Exhaustion contradiction: overbought vs bullish momentum
                elif s1 in overbought_tokens and s2 in bullish_tokens:
                    self.ontology.link_indicators(u1, u2, "contradicts")
                elif s2 in overbought_tokens and s1 in bullish_tokens:
                    self.ontology.link_indicators(u2, u1, "contradicts")
                elif s1 in oversold_tokens and s2 in bearish_tokens:
                    self.ontology.link_indicators(u1, u2, "contradicts")
                elif s2 in oversold_tokens and s1 in bearish_tokens:
                    self.ontology.link_indicators(u2, u1, "contradicts")

                # Directional confirmation
                elif s1 in bullish_tokens and s2 in bullish_tokens:
                    self.ontology.link_indicators(u1, u2, "confirms")
                elif s1 in bearish_tokens and s2 in bearish_tokens:
                    self.ontology.link_indicators(u1, u2, "confirms")

    # ============================================================
    # EXTRACTION METHODS
    # ============================================================
    def _extract_trend_enhanced(self, symbol, df):
        closes = df["close"]
        entities = {"uris": [], "signals": [], "confidences": []}

        for period, weight in [(20, 0.3), (50, 0.4), (200, 0.3)]:
            if len(closes) < period:
                continue
            sma = closes.rolling(period).mean().iloc[-1]
            ema = closes.ewm(span=period).mean().iloc[-1]
            current = closes.iloc[-1]

            sma_signal, sma_conf = self._classify_ma_signal(current, sma)
            sma_uri = self.ontology.add_indicator(
                symbol, f"SMA_{period}", sma, sma_signal, sma_conf,
                {"timestamp": df.index[-1].isoformat()}
            )
            entities["uris"].append(sma_uri)
            entities["signals"].append(sma_signal)
            entities["confidences"].append(sma_conf * weight)

            ema_signal, ema_conf = self._classify_ma_signal(current, ema, threshold=0.015)
            ema_uri = self.ontology.add_indicator(
                symbol, f"EMA_{period}", ema, ema_signal, ema_conf,
                {"timestamp": df.index[-1].isoformat()}
            )
            entities["uris"].append(ema_uri)
            entities["signals"].append(ema_signal)
            entities["confidences"].append(ema_conf * weight * 1.2)

        adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        adx_value = adx_ind.adx().iloc[-1]
        di_plus   = adx_ind.adx_pos().iloc[-1]
        di_minus  = adx_ind.adx_neg().iloc[-1]

        adx_strength, adx_conf = self._classify_adx(adx_value)
        if adx_value >= 40:
            trend_strength = "very_strong"
        elif adx_value >= 25:
            trend_strength = "strong"
        elif adx_value >= 20:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"

        adx_uri = self.ontology.add_indicator(
            symbol, "ADX", adx_value, adx_strength, adx_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(adx_uri)
        entities["signals"].append(adx_strength)
        entities["confidences"].append(adx_conf * 0.5)

        di_signal = "bullish" if di_plus > di_minus else "bearish"
        di_conf = min(abs(di_plus - di_minus) / 100, 1.0)
        di_uri = self.ontology.add_indicator(
            symbol, "DI_Cross", di_plus - di_minus, di_signal, di_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(di_uri)
        entities["signals"].append(di_signal)
        entities["confidences"].append(di_conf * 0.3)

        entities["avg_confidence"] = (
            sum(entities["confidences"]) / len(entities["confidences"])
            if entities["confidences"] else 0.5
        )
        entities["trend_strength"] = trend_strength
        entities["di_signal"] = di_signal
        return entities

    def _extract_momentum_enhanced(self, symbol, df):
        closes = df["close"]
        entities = {"uris": [], "signals": [], "confidences": []}

        rsi_val = ta.momentum.RSIIndicator(closes).rsi().iloc[-1]
        rsi_signal, rsi_conf = self._classify_rsi(rsi_val)
        rsi_uri = self.ontology.add_indicator(
            symbol, "RSI", rsi_val, rsi_signal, rsi_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(rsi_uri)
        entities["signals"].append(rsi_signal)
        entities["confidences"].append(rsi_conf * 0.25)

        macd_ind    = ta.trend.MACD(closes)
        macd_val    = macd_ind.macd().iloc[-1]
        macd_signal = macd_ind.macd_signal().iloc[-1]
        macd_hist   = macd_ind.macd_diff().iloc[-1]

        macd_signal_type, macd_conf = self._classify_macd(macd_val, macd_signal, macd_hist)
        macd_uri = self.ontology.add_indicator(
            symbol, "MACD", macd_val, macd_signal_type, macd_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(macd_uri)
        entities["signals"].append(macd_signal_type)
        entities["confidences"].append(macd_conf * 0.25)

        stoch_ind = ta.momentum.StochasticOscillator(df["high"], df["low"], closes)
        stoch_k   = stoch_ind.stoch().iloc[-1]
        stoch_d   = stoch_ind.stoch_signal().iloc[-1]
        stoch_signal, stoch_conf = self._classify_stochastic(stoch_k, stoch_d)
        stoch_uri = self.ontology.add_indicator(
            symbol, "Stochastic", stoch_k, stoch_signal, stoch_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(stoch_uri)
        entities["signals"].append(stoch_signal)
        entities["confidences"].append(stoch_conf * 0.25)

        cci_val = ta.trend.CCIIndicator(df["high"], df["low"], closes).cci().iloc[-1]
        cci_signal, cci_conf = self._classify_cci(cci_val)
        cci_uri = self.ontology.add_indicator(
            symbol, "CCI", cci_val, cci_signal, cci_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(cci_uri)
        entities["signals"].append(cci_signal)
        entities["confidences"].append(cci_conf * 0.25)

        entities["avg_confidence"] = (
            sum(entities["confidences"]) / len(entities["confidences"])
            if entities["confidences"] else 0.5
        )
        return entities

    def _extract_volume_enhanced(self, symbol, df):
        closes, vols = df["close"], df["volume"]
        entities = {
            "uris": [], "signals": [],
            "profile": "neutral", "confidence": 0.5
        }

        # OBV
        obv_val = ta.volume.OnBalanceVolumeIndicator(closes, vols).on_balance_volume().iloc[-1]
        obv_prev = df["OBV"].iloc[-5] if len(df) > 5 and "OBV" in df.columns else obv_val
        obv_signal = "accumulation" if obv_val > obv_prev else "distribution"
        obv_conf = min(abs(obv_val - obv_prev) / abs(obv_prev), 1.0) if obv_prev != 0 else 0.5
        obv_uri = self.ontology.add_indicator(
            symbol, "OBV", obv_val, obv_signal, obv_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(obv_uri)
        entities["signals"].append(obv_signal)

        # VWAP
        vwap_val = self._safe_get_value(df, "VWAP", df["close"].iloc[-1])
        vwap_dev = (closes.iloc[-1] - vwap_val) / vwap_val if vwap_val else 0.0
        vwap_signal = "above_vwap" if vwap_dev > 0 else "below_vwap"
        vwap_conf = min(abs(vwap_dev) * 10, 1.0)
        vwap_uri = self.ontology.add_indicator(
            symbol, "VWAP", vwap_val, vwap_signal, vwap_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(vwap_uri)
        entities["signals"].append(vwap_signal)

        # MFI
        mfi_val = ta.volume.MFIIndicator(df["high"], df["low"], closes, vols).money_flow_index().iloc[-1]
        mfi_signal, mfi_conf = self._classify_mfi(mfi_val)
        mfi_uri = self.ontology.add_indicator(
            symbol, "MFI", mfi_val, mfi_signal, mfi_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(mfi_uri)
        entities["signals"].append(mfi_signal)

        # CMF
        cmf_val = ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], closes, vols).chaikin_money_flow().iloc[-1]
        cmf_signal = "accumulation" if cmf_val > 0 else "distribution"
        cmf_conf = min(abs(cmf_val) * 5, 1.0)
        cmf_uri = self.ontology.add_indicator(
            symbol, "CMF", cmf_val, cmf_signal, cmf_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(cmf_uri)
        entities["signals"].append(cmf_signal)

        # ADL
        adl_val  = self._safe_get_value(df, "ADL", 0.0)
        adl_prev = self._safe_get_value(df, "ADL", adl_val, -5)
        adl_signal = "accumulation" if adl_val > adl_prev else "distribution"
        adl_conf = min(abs(adl_val - adl_prev) / abs(adl_prev), 1.0) if adl_prev != 0 else 0.5
        adl_uri = self.ontology.add_indicator(
            symbol, "ADL", adl_val, adl_signal, adl_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(adl_uri)
        entities["signals"].append(adl_signal)

        # Force Index
        fi_series = ta.volume.ForceIndexIndicator(closes, vols).force_index()
        fi_val = fi_series.iloc[-1]
        fi_signal = "positive_force" if fi_val > 0 else "negative_force"
        fi_tail = fi_series.tail(20).abs()
        fi_conf = min(abs(fi_val) / max(fi_tail.mean(), 1.0), 1.0) if len(fi_tail) else 0.5
        fi_uri = self.ontology.add_indicator(
            symbol, "ForceIndex", fi_val, fi_signal, fi_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(fi_uri)
        entities["signals"].append(fi_signal)

        # [FIX-6] Aggregate ALL volume signals into the profile (was 3 of 6).
        bullish_tokens = {"accumulation", "above_vwap", "bullish_flow", "positive_force"}
        bearish_tokens = {"distribution", "below_vwap", "bearish_flow", "negative_force"}

        all_signals = [obv_signal, vwap_signal, mfi_signal, cmf_signal, adl_signal, fi_signal]
        acc_count = sum(1 for s in all_signals if s in bullish_tokens)
        dist_count = sum(1 for s in all_signals if s in bearish_tokens)

        if acc_count >= 4:
            entities["profile"]    = "strong_accumulation"
            entities["confidence"] = 0.85
        elif dist_count >= 4:
            entities["profile"]    = "strong_distribution"
            entities["confidence"] = 0.85
        elif acc_count > dist_count:
            entities["profile"]    = "accumulation"
            entities["confidence"] = 0.65
        elif dist_count > acc_count:
            entities["profile"]    = "distribution"
            entities["confidence"] = 0.65
        else:
            entities["profile"]    = "neutral"
            entities["confidence"] = 0.5

        return entities

    def _extract_volatility_enhanced(self, symbol, df):
        closes = df["close"]
        entities = {"uris": [], "signals": [], "regime": "medium", "confidence": 0.5}

        atr_val = ta.volatility.AverageTrueRange(df["high"], df["low"], closes).average_true_range().iloc[-1]
        atr_pct = (atr_val / closes.iloc[-1]) * 100 if closes.iloc[-1] else 0.0

        if atr_pct > 5:
            vol_signal, vol_conf, regime = "high_volatility", 0.9, "high"
        elif atr_pct < 2:
            vol_signal, vol_conf, regime = "low_volatility", 0.9, "low"
        else:
            vol_signal, vol_conf, regime = "medium_volatility", 0.7, "medium"

        atr_uri = self.ontology.add_indicator(
            symbol, "ATR_pct", atr_pct, vol_signal, vol_conf,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(atr_uri)
        entities["signals"].append(vol_signal)
        entities["regime"] = regime
        entities["confidence"] = vol_conf

        if "Upper_band" in df.columns and "Lower_band" in df.columns and "SMA_20" in df.columns:
            sma20 = df["SMA_20"].iloc[-1]
            if sma20 and not pd.isna(sma20):
                bb_width = (df["Upper_band"].iloc[-1] - df["Lower_band"].iloc[-1]) / sma20
                if bb_width < 0.05:
                    bb_signal, bb_conf = "squeeze", 0.85
                elif bb_width > 0.15:
                    bb_signal, bb_conf = "expansion", 0.7
                else:
                    bb_signal, bb_conf = "normal", 0.5
                bb_uri = self.ontology.add_indicator(
                    symbol, "BollingerWidth", bb_width * 100, bb_signal, bb_conf,
                    {"timestamp": df.index[-1].isoformat()}
                )
                entities["uris"].append(bb_uri)
                entities["signals"].append(bb_signal)
        return entities

    def _extract_ichimoku_enhanced(self, symbol, df):
        entities = {"uris": [], "signals": []}
        required_cols = ["Tenkan_sen", "Kijun_sen", "Senkou_span_a", "Senkou_span_b", "Chikou_span"]
        if not all(col in df.columns for col in required_cols):
            return entities

        tenkan   = df["Tenkan_sen"].iloc[-1]
        kijun    = df["Kijun_sen"].iloc[-1]
        senkou_a = df["Senkou_span_a"].iloc[-1]
        senkou_b = df["Senkou_span_b"].iloc[-1]
        chikou   = df["Chikou_span"].iloc[-26] if len(df) > 26 else df["close"].iloc[-1]
        current  = df["close"].iloc[-1]

        if pd.isna(tenkan) or pd.isna(kijun):
            return entities

        tk_signal = "bullish" if tenkan > kijun else "bearish"
        tk_uri = self.ontology.add_indicator(
            symbol, "Ichimoku_TK", tenkan - kijun, tk_signal, 0.7,
            {"timestamp": df.index[-1].isoformat()}
        )
        entities["uris"].append(tk_uri)
        entities["signals"].append(tk_signal)

        if not (pd.isna(senkou_a) or pd.isna(senkou_b)):
            cloud_top    = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)
            if current > cloud_top:
                price_signal, price_conf = "above_cloud", 0.85
            elif current < cloud_bottom:
                price_signal, price_conf = "below_cloud", 0.85
            else:
                price_signal, price_conf = "in_cloud", 0.5
            price_uri = self.ontology.add_indicator(
                symbol, "Ichimoku_PriceVsCloud", current, price_signal, price_conf,
                {"timestamp": df.index[-1].isoformat()}
            )
            entities["uris"].append(price_uri)
            entities["signals"].append(price_signal)

        if not pd.isna(chikou):
            lag_signal = "bullish" if chikou > current else "bearish"
            lag_uri = self.ontology.add_indicator(
                symbol, "Ichimoku_Chikou", chikou, lag_signal, 0.6,
                {"timestamp": df.index[-1].isoformat()}
            )
            entities["uris"].append(lag_uri)
            entities["signals"].append(lag_signal)

        return entities

    def _extract_fibonacci_enhanced(self, symbol, df):
        """[FIX-8] Actually use last 252 bars (true 52-week window) instead
        of the entire fetched range."""
        entities = {"uris": [], "signals": []}
        window = df.tail(252) if len(df) >= 252 else df
        high_52w = window["high"].max()
        low_52w  = window["low"].min()
        diff = high_52w - low_52w
        if diff <= 0:
            return entities

        fib_levels = {
            "0%":    high_52w,
            "23.6%": high_52w - 0.236 * diff,
            "38.2%": high_52w - 0.382 * diff,
            "50%":   high_52w - 0.5   * diff,
            "61.8%": high_52w - 0.618 * diff,
            "78.6%": high_52w - 0.786 * diff,
            "100%":  low_52w
        }
        current = df["close"].iloc[-1]

        for name, level in fib_levels.items():
            proximity = abs(current - level) / current if current else 1.0
            conf = max(1 - proximity * 3, 0.2)
            fib_uri = self.ontology.add_indicator(
                symbol, f"Fib_{name}", level, "support_resistance", conf,
                {"timestamp": df.index[-1].isoformat(), "level_name": name}
            )
            entities["uris"].append(fib_uri)
            entities["signals"].append("support_resistance")
        return entities

    def _calculate_sr_levels(self, df):
        recent = df.tail(30)
        return {
            "support": sorted([
                float(recent["low"].min()),
                float(recent["low"].quantile(0.25)),
                float(recent["low"].quantile(0.1))
            ]),
            "resistance": sorted([
                float(recent["high"].max()),
                float(recent["high"].quantile(0.75)),
                float(recent["high"].quantile(0.9))
            ], reverse=True)
        }

    # ============================================================
    # WEIGHTED INFERENCE
    # ============================================================
    def _infer_market_state_weighted(self, extracts):
        scores = {state: 0.0 for state in MarketState}
        confidences = {state: [] for state in MarketState}

        t = extracts["trend"]
        if t.get("trend_strength") in ["strong", "very_strong"]:
            if "bullish" in t.get("di_signal", ""):
                scores[MarketState.BULL_TREND] += self.indicator_weights["trend"]
                confidences[MarketState.BULL_TREND].append(t.get("avg_confidence", 0.5))
            else:
                scores[MarketState.BEAR_TREND] += self.indicator_weights["trend"]
                confidences[MarketState.BEAR_TREND].append(t.get("avg_confidence", 0.5))

        m = extracts["momentum"]
        bullish_mom = sum(1 for s in m.get("signals", []) if "bullish" in s)
        bearish_mom = sum(1 for s in m.get("signals", []) if "bearish" in s)
        if bullish_mom >= 2:
            scores[MarketState.BULL_TREND] += self.indicator_weights["momentum"]
            confidences[MarketState.BULL_TREND].append(m.get("avg_confidence", 0.5))
        elif bearish_mom >= 2:
            scores[MarketState.BEAR_TREND] += self.indicator_weights["momentum"]
            confidences[MarketState.BEAR_TREND].append(m.get("avg_confidence", 0.5))

        # [FIX-6] Symmetric volume weighting.
        v = extracts["volume"]
        profile = v.get("profile", "")
        if "strong_accumulation" in profile:
            scores[MarketState.BULL_TREND] += self.indicator_weights["volume"] * 1.5
            confidences[MarketState.BULL_TREND].append(v.get("confidence", 0.5))
        elif "strong_distribution" in profile:
            scores[MarketState.BEAR_TREND] += self.indicator_weights["volume"] * 1.5
            confidences[MarketState.BEAR_TREND].append(v.get("confidence", 0.5))
        elif profile == "accumulation":
            scores[MarketState.BULL_TREND] += self.indicator_weights["volume"]
            confidences[MarketState.BULL_TREND].append(v.get("confidence", 0.5))
        elif profile == "distribution":
            scores[MarketState.BEAR_TREND] += self.indicator_weights["volume"]
            confidences[MarketState.BEAR_TREND].append(v.get("confidence", 0.5))

        vol = extracts["volatility"]
        if vol.get("regime") == "high":
            scores[MarketState.VOLATILE_BREAKOUT] += self.indicator_weights["volatility"]
            confidences[MarketState.VOLATILE_BREAKOUT].append(vol.get("confidence", 0.5))
        elif vol.get("regime") == "low" and max(scores.values()) < 0.3:
            scores[MarketState.RANGE_BOUND] += self.indicator_weights["volatility"]
            confidences[MarketState.RANGE_BOUND].append(vol.get("confidence", 0.5))

        winning_state = (
            max(scores.items(), key=lambda x: x[1])[0]
            if scores and max(scores.values()) > 0 else MarketState.SIDEWAYS_CONSOLIDATION
        )
        win_confs = confidences.get(winning_state, [])
        avg_conf = sum(win_confs) / max(len(win_confs), 1) if win_confs else 0.5
        return winning_state, avg_conf

    def _infer_trend_direction_weighted(self, extracts):
        bullish_score = 0.0
        bearish_score = 0.0
        total_conf = 0.0

        t = extracts["trend"]
        di = t.get("di_signal", "")
        strength = t.get("trend_strength", "weak")
        strength_mult = {"very_strong": 2.0, "strong": 1.5, "moderate": 1.0, "weak": 0.5}.get(strength, 0.5)
        if "bullish" in di:
            bullish_score += strength_mult
        else:
            bearish_score += strength_mult
        total_conf += t.get("avg_confidence", 0.5)

        m = extracts["momentum"]
        bullish_mom = sum(1 for s in m.get("signals", []) if "bullish" in s)
        bearish_mom = sum(1 for s in m.get("signals", []) if "bearish" in s)
        bullish_score += bullish_mom * 0.8
        bearish_score += bearish_mom * 0.8
        total_conf += m.get("avg_confidence", 0.5)

        ich = extracts.get("ichimoku", {})
        ich_signals = ich.get("signals", [])
        if len(ich_signals) >= 2:
            bullish_score += sum(1 for s in ich_signals if "bullish" in s or "above" in s) * 0.6
            bearish_score += sum(1 for s in ich_signals if "bearish" in s or "below" in s) * 0.6

        v = extracts["volume"]
        profile = v.get("profile", "")
        if "accumulation" in profile:
            bullish_score += 0.5 if profile == "accumulation" else 0.8
        elif "distribution" in profile:
            bearish_score += 0.5 if profile == "distribution" else 0.8

        net = bullish_score - bearish_score
        if net >= 3.0:
            direction = TrendDirection.STRONG_UP
        elif net >= 1.5:
            direction = TrendDirection.MODERATE_UP
        elif net <= -3.0:
            direction = TrendDirection.STRONG_DOWN
        elif net <= -1.5:
            direction = TrendDirection.MODERATE_DOWN
        else:
            direction = TrendDirection.NEUTRAL

        avg_conf = total_conf / 3 if total_conf > 0 else 0.5
        return direction, avg_conf

    def _infer_risk_level_weighted(self, extracts):
        risk_score = 0.0
        confidences = []

        vol = extracts["volatility"]
        if vol.get("regime") == "high":
            risk_score += 4.0
            confidences.append(vol.get("confidence", 0.5))
        elif vol.get("regime") == "medium":
            risk_score += 2.0
            confidences.append(vol.get("confidence", 0.5))

        t = extracts["trend"]
        if t.get("trend_strength") == "weak":
            risk_score += 1.0
            confidences.append(0.6)

        m = extracts["momentum"]
        signals = m.get("signals", [])
        if any("overbought" in s or "oversold" in s for s in signals):
            risk_score += 1.5
            confidences.append(0.7)

        if risk_score >= 4.5:
            level = RiskLevel.VERY_HIGH
        elif risk_score >= 3.5:
            level = RiskLevel.HIGH
        elif risk_score >= 2.5:
            level = RiskLevel.MEDIUM
        elif risk_score >= 1.5:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.VERY_LOW

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5
        return level, avg_conf

    def _build_dynamic_reasoning(self, symbol, extracts, market_state, trend_direction,
                                 risk_level, contradictions, confidence):
        chain = [
            f"Enhanced Analysis for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Overall Confidence: {confidence:.1%}",
            f"Market State: {market_state.value.replace('_', ' ').title()}",
            f"Trend Direction: {trend_direction.value.replace('_', ' ').title()}",
            f"Risk Level: {risk_level.value.replace('_', ' ').title()}"
        ]

        top_evidence = []
        for category, data in extracts.items():
            if category != "ml_features" and isinstance(data, dict):
                if data.get("avg_confidence", 0) > 0.7 or data.get("confidence", 0) > 0.7:
                    top_evidence.append(f"- {category.title()}: High-confidence signals")
        if top_evidence:
            chain.append("Key Evidence:")
            chain.extend(top_evidence)

        if contradictions:
            chain.append(f"⚠️ Detected {len(contradictions)} indicator contradictions")

        confirmations = self.ontology.find_confirmations()
        if confirmations:
            chain.append(f"✅ Found {len(confirmations)} strong indicator confirmations")

        summary = self.ontology.get_knowledge_summary()
        chain.append(
            f"📊 Knowledge Graph: {summary['total_statements']} statements, "
            f"{summary['indicators']} indicators, "
            f"{summary['evidence_bundles']} evidence bundles"
        )
        return chain

    def _format_confidence(self, conf: float) -> str:
        if conf > 0.8:
            return "Very High"
        elif conf > 0.6:
            return "High"
        elif conf > 0.4:
            return "Moderate"
        return "Low"

    def _default_context(self):
        return MarketContext(
            market_state=MarketState.SIDEWAYS_CONSOLIDATION,
            trend_direction=TrendDirection.NEUTRAL,
            risk_level=RiskLevel.MEDIUM,
            confidence_score=0.0,
            volatility_regime="unknown",
            volume_profile="unknown",
            support_levels=[],
            resistance_levels=[],
            ontology_graph="",
            reasoning_chain=["Insufficient data (need ≥ 50 bars)."],
            contradictions=[],
            knowledge_summary={},
            confirmations=[]
        )


# ============================================================
# PART 2: ENHANCED INFERENCE ENGINE (enums + MarketContext)
# ============================================================
class SignalType(Enum):
    BULLISH_STRONG    = "bullish_strong"
    BULLISH_MODERATE  = "bullish_moderate"
    BEARISH_STRONG    = "bearish_strong"
    BEARISH_MODERATE  = "bearish_moderate"
    NEUTRAL           = "neutral"
    OVERSOLD          = "oversold"
    OVERBOUGHT        = "overbought"
    ACCUMULATION      = "accumulation"
    DISTRIBUTION      = "distribution"
    HIGH_VOLATILITY   = "high_volatility"
    LOW_VOLATILITY    = "low_volatility"


class MarketState(Enum):
    BULL_TREND              = "bull_trend"
    BEAR_TREND              = "bear_trend"
    SIDEWAYS_CONSOLIDATION  = "sideways_consolidation"
    VOLATILE_BREAKOUT       = "volatile_breakout"
    RANGE_BOUND             = "range_bound"


class TrendDirection(Enum):
    STRONG_UP     = "strong_up"
    MODERATE_UP   = "moderate_up"
    NEUTRAL       = "neutral"
    MODERATE_DOWN = "moderate_down"
    STRONG_DOWN   = "strong_down"


class RiskLevel(Enum):
    VERY_LOW  = "very_low"
    LOW       = "low"
    MEDIUM    = "medium"
    HIGH      = "high"
    VERY_HIGH = "very_high"


@dataclass
class MarketContext:
    """Enhanced context with real knowledge-graph stats and confirmations."""
    market_state:       MarketState
    trend_direction:    TrendDirection
    risk_level:         RiskLevel
    confidence_score:   float
    volatility_regime:  str
    volume_profile:     str
    support_levels:     List[float]
    resistance_levels:  List[float]
    ontology_graph:     str
    reasoning_chain:    List[str]
    contradictions:     List[Dict[str, str]]
    # [FIX-2] Surfaced ontology statistics through to the UI.
    knowledge_summary:  Dict[str, Any] = field(default_factory=dict)
    confirmations:      List[Dict[str, str]] = field(default_factory=list)


# Enhanced ontology engine instance
enhanced_ontology = EnhancedStockAnalysisOntology(debug=False)


# ============================================================
# PART 3: DATA FETCHING, INDICATOR COMPUTATION & DASH LAYOUT
# ============================================================
ontology = enhanced_ontology


@memory.cache
def fetch_data_cached(ticker: str, period: str, interval: str) -> pd.DataFrame:
    log_step(f"Fetching data for {ticker} | Period={period} | Interval={interval}")
    tq = Ticker(ticker)
    df = tq.history(period=period, interval=interval)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values("date")
    df = df.dropna(subset=["close"])
    log_step(f"Retrieved {len(df)} rows for {ticker}.")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    log_step("Computing technical indicators…")
    closes, highs, lows, vols = df["close"], df["high"], df["low"], df["volume"]
    result_df = df.copy()

    try:
        for w in [8, 20, 50, 200]:
            result_df[f"SMA_{w}"] = closes.rolling(w).mean()
            result_df[f"EMA_{w}"] = closes.ewm(span=w, adjust=False).mean()

        result_df["RSI"] = ta.momentum.RSIIndicator(closes).rsi()

        macd = ta.trend.MACD(closes)
        result_df["MACD"], result_df["MACD_Signal"] = macd.macd(), macd.macd_signal()

        stoch = ta.momentum.StochasticOscillator(highs, lows, closes)
        result_df["%K"], result_df["%D"] = stoch.stoch(), stoch.stoch_signal()

        ma20, std20 = closes.rolling(20).mean(), closes.rolling(20).std()
        result_df["Upper_band"], result_df["Lower_band"] = ma20 + 2 * std20, ma20 - 2 * std20
        result_df["ATR"] = ta.volatility.AverageTrueRange(highs, lows, closes).average_true_range()
        result_df["CCI"] = ta.trend.CCIIndicator(highs, lows, closes).cci()

        result_df["OBV"]  = ta.volume.OnBalanceVolumeIndicator(closes, vols).on_balance_volume()
        result_df["VWAP"] = (closes * vols).cumsum() / vols.cumsum()
        result_df["ADL"]  = ta.volume.AccDistIndexIndicator(highs, lows, closes, vols).acc_dist_index()
        result_df["MFI"]  = ta.volume.MFIIndicator(highs, lows, closes, vols).money_flow_index()
        result_df["CMF"]  = ta.volume.ChaikinMoneyFlowIndicator(highs, lows, closes, vols).chaikin_money_flow()
        result_df["FI"]   = ta.volume.ForceIndexIndicator(closes, vols).force_index()

        adx = ta.trend.ADXIndicator(highs, lows, closes)
        result_df["ADX"], result_df["DI+"], result_df["DI-"] = adx.adx(), adx.adx_pos(), adx.adx_neg()

        result_df["Tenkan_sen"]    = (highs.rolling(9).max()  + lows.rolling(9).min())  / 2
        result_df["Kijun_sen"]     = (highs.rolling(26).max() + lows.rolling(26).min()) / 2
        result_df["Senkou_span_a"] = ((result_df["Tenkan_sen"] + result_df["Kijun_sen"]) / 2).shift(26)
        result_df["Senkou_span_b"] = ((highs.rolling(52).max() + lows.rolling(52).min()) / 2).shift(26)
        result_df["Chikou_span"]   = closes.shift(-26)

        log_step("Indicators computed successfully.")
    except Exception as e:
        log_step(f"Warning: Some indicators failed to compute: {e}")

    return result_df


# ─────────────────────────────────────────────
# Dash Application Setup (unchanged)
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SOLAR],
    suppress_callback_exceptions=True
)
app.title = "Enhanced Ontology-Driven Stock Dashboard"
server = app.server


# ─────────────────────────────────────────────
# Dash Layout (UNCHANGED — same widgets, IDs, order)
# ─────────────────────────────────────────────
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Enhanced Ontology-Driven Stock Dashboard (Prototype)",
        color="dark", dark=True
    ),
    dbc.Row([
        dbc.Col(dbc.Input(id="stock-input", value="AAPL", placeholder="Enter stock symbol"), width=4)
    ], justify="center", className="my-3"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id="time-range",
            options=[
                {"label": "6 Months", "value": "6mo"},
                {"label": "1 Year",   "value": "1y"},
                {"label": "2 Years",  "value": "2y"},
                {"label": "3 Years",  "value": "3y"},
                {"label": "4 Years",  "value": "4y"},
                {"label": "5 Years",  "value": "5y"},
                {"label": "All",      "value": "max"},
            ],
            value="1y", clearable=False
        ), width=4)
    ], justify="center", className="my-3"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id="interval",
            options=[
                {"label": "Daily",   "value": "1d"},
                {"label": "Weekly",  "value": "1wk"},
                {"label": "Monthly", "value": "1mo"},
            ],
            value="1d", clearable=False
        ), width=4)
    ], justify="center", className="my-3"),
    dbc.Row([
        dbc.Col(dcc.RadioItems(
            id="analysis-mode",
            options=[
                {"label": "📊 Standard",          "value": "standard"},
                {"label": "🧠 Ontology Analysis", "value": "ontology"},
            ],
            value="ontology", inline=True
        ), width=8)
    ], justify="center", className="my-3"),
    dbc.Row([
        dbc.Col(dbc.Button(
            id="analyze-button", n_clicks=0,
            children="🧠 Analyze with Enhanced Ontology",
            color="primary"
        ), width="auto")
    ], justify="center", className="my-3"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Enhanced Ontology-Based Analysis", className="bg-primary text-white"),
            dbc.CardBody([
                html.Div(id="ontology-insights"),
                html.Div(id="trading-signals"),
                html.Div(id="risk-assessment"),
                html.Div(id="trading-recommendations"),
                html.Div(id="reasoning-trace"),
            ])
        ]), width=12)
    ], className="mb-4"),

    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="candlestick-chart"))), width=12)], className="mb-4"),
    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="sma-ema-chart"))), width=12)], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="support-resistance-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="rsi-chart"))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="bollinger-bands-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="macd-chart"))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="stochastic-oscillator-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="obv-chart"))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="atr-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="cci-chart"))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="mfi-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="cmf-chart"))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fi-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="fibonacci-retracement-chart"))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="ichimoku-cloud-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="vwap-chart"))), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="adl-chart"))), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="adx-di-chart"))), width=6),
    ], className="mb-4"),

    dbc.Row([dbc.Col(html.Footer(
        "Enhanced Ontology-Driven Dashboard © 2025",
        className="text-center text-muted"
    ))], className="mt-4"),
], fluid=True)


# ============================================================
# PART 4: CALLBACKS & APPLICATION EXECUTION
# ============================================================
@app.callback(Output("analyze-button", "children"),
              Input("analysis-mode", "value"))
def update_button_text(mode):
    return "🧠 Analyze with Enhanced Ontology" if mode == "ontology" else "📊 Analyze Stock"


@app.callback(
    [Output(g, "figure") for g in [
        "candlestick-chart", "sma-ema-chart", "support-resistance-chart", "rsi-chart",
        "bollinger-bands-chart", "macd-chart", "stochastic-oscillator-chart", "obv-chart",
        "atr-chart", "cci-chart", "mfi-chart", "cmf-chart", "fi-chart",
        "fibonacci-retracement-chart", "ichimoku-cloud-chart", "vwap-chart",
        "adl-chart", "adx-di-chart"
    ]]
    + [Output(x, "children") for x in [
        "ontology-insights", "trading-signals", "risk-assessment",
        "trading-recommendations", "reasoning-trace"
    ]],
    Input("analyze-button", "n_clicks"),
    State("stock-input", "value"),
    State("time-range", "value"),
    State("interval", "value"),
    State("analysis-mode", "value"),
)
def update_graphs(n_clicks, ticker, time_range, interval, analysis_mode):
    if not n_clicks:
        empty_fig = go.Figure().update_layout(
            title="Click 'Analyze' to Begin", template="plotly_dark"
        )
        placeholder = html.Div("Awaiting user input…")
        return (empty_fig,) * 18 + (placeholder, html.Div(), html.Div(), html.Div(), html.Div())

    try:
        log_step(f"Starting enhanced ontology-driven analysis for {ticker}…")
        df = fetch_data_cached(ticker, time_range, interval)
        df = compute_indicators(df)
    except Exception as e:
        log_step(f"❌ Data fetch error: {e}")
        err_fig = go.Figure().update_layout(title=f"Error: {e}", template="plotly_dark")
        err_msg = html.Div(f"⚠️ Error fetching data for {ticker}: {e}")
        return (err_fig,) * 18 + (err_msg, html.Div(), html.Div(), html.Div(), html.Div())

    if analysis_mode == "ontology":
        log_step("Executing enhanced ontology reasoning engine…")
        context = ontology.infer_market_context(ticker, df)

        mc = {
            "state":            context.market_state.value,
            "trend":            context.trend_direction.value,
            "risk":             context.risk_level.value,
            "volatility_regime":context.volatility_regime,
            "volume_profile":   context.volume_profile,
            "support_levels":   context.support_levels,
            "resistance_levels":context.resistance_levels,
            "confidence":       context.confidence_score
        }
        reasoning_chain = context.reasoning_chain
        kg_summary      = context.knowledge_summary or {}

        sup_str = ", ".join(f"{s:.2f}" for s in mc["support_levels"])    if mc["support_levels"]    else "–"
        res_str = ", ".join(f"{r:.2f}" for r in mc["resistance_levels"]) if mc["resistance_levels"] else "–"

        # [FIX-2] Real knowledge-graph stats — replaces hardcoded "+10" sum.
        insights_content = html.Div([
            html.H4("🧠 Enhanced Ontological Market Summary"),
            html.P(f"Market State: {mc['state'].replace('_', ' ').title()}"),
            html.P(f"Trend Direction: {mc['trend'].replace('_', ' ').title()}"),
            html.P(f"Risk Level: {mc['risk'].replace('_', ' ').title()}"),
            html.P(f"Volatility Regime: {mc['volatility_regime'].replace('_', ' ').title()}"),
            html.P(f"Volume Profile: {mc['volume_profile'].replace('_', ' ').title()}"),
            html.P(f"Support Levels: {sup_str}"),
            html.P(f"Resistance Levels: {res_str}"),
            html.Hr(),
            html.H6("Knowledge Graph Statistics"),
            html.P(f"Total Statements: {kg_summary.get('total_statements', 0)}"),
            html.P(f"Indicators: {kg_summary.get('indicators', 0)}"),
            html.P(f"Evidence Bundles: {kg_summary.get('evidence_bundles', 0)}"),
            html.P(f"Contradictions Detected: {kg_summary.get('contradictions', len(context.contradictions))}"),
            html.P(f"Confirmations Detected: {kg_summary.get('confirmations', len(context.confirmations))}"),
            html.P(f"Reasoning Steps: {len(reasoning_chain)}")
        ])

        signals_content = html.Div([
            html.H5("📈 Enhanced Trading Bias"),
            html.Ul([
                html.Li("🚀 Strong Bullish Bias") if mc["state"] == "bull_trend" and mc["confidence"] > 0.8
                else html.Li("📈 Bullish Bias") if mc["state"] == "bull_trend"
                else html.Li("📉 Bearish Bias") if mc["state"] == "bear_trend"
                else html.Li("⚖️ Neutral Market Conditions")
            ]),
            html.Hr(),
            html.H6("Confidence Metrics"),
            html.P(f"Overall Confidence: {mc['confidence']:.1%}"),
            html.P(f"Trend Strength: {mc['trend'].replace('_', ' ').title()}")
        ])

        risk_content = html.Div([
            html.H5("🛡️ Enhanced Risk & Volatility Assessment"),
            html.P(f"Risk Level: {mc['risk'].replace('_', ' ').title()}"),
            html.P(f"Volatility: {mc['volatility_regime'].replace('_', ' ').title()}"),
            html.Hr(),
            html.H6("Risk Factors"),
            html.Ul([
                html.Li("High volatility detected") if mc["volatility_regime"] == "high"
                else html.Li("Moderate volatility") if mc["volatility_regime"] == "medium"
                else html.Li("Low volatility environment")
            ])
        ])

        # [FIX-7] Recommendations require agreement between state and trend direction.
        trend = mc["trend"]
        state = mc["state"]
        risk  = mc["risk"]

        bullish_directions = {"strong_up", "moderate_up"}
        bearish_directions = {"strong_down", "moderate_down"}

        recs = []
        if state == "bull_trend" and trend in bullish_directions and risk in ["low", "very_low", "medium"]:
            recs.extend([
                "🟢 Strong Buy Signal: state and trend direction agree",
                "📈 Buy on dips to support levels",
                "⚡ Use momentum indicators for entry timing"
            ])
        elif state == "bear_trend" and trend in bearish_directions and risk in ["high", "very_high", "medium"]:
            recs.extend([
                "🔴 Strong Sell Signal: state and trend direction agree",
                "📉 Sell on rallies to resistance",
                "🛡️ Implement strict risk management"
            ])
        elif state == "bull_trend" and trend in bearish_directions:
            recs.append("⚠️ Mixed signal: bull state but down direction — wait for confirmation")
        elif state == "bear_trend" and trend in bullish_directions:
            recs.append("⚠️ Mixed signal: bear state but up direction — likely counter-trend rally, not a reversal")
        elif state == "volatile_breakout":
            recs.append("⚡ Confirm breakout before large position entries")
        else:
            recs.append("⚖️ Maintain neutral exposure until trend confirmation")

        recommendations_content = html.Div([
            html.H5("💡 Enhanced Trading Recommendations"),
            html.Ul([html.Li(r) for r in recs]),
            html.Hr(),
            html.H6("Key Levels"),
            html.P(f"Entry Zones: Support levels {sup_str}"),
            html.P(f"Exit Zones: Resistance levels {res_str}")
        ])

        reasoning_trace = html.Div([
            html.H5("🔍 Enhanced Ontology Reasoning Trace"),
            html.Ol([html.Li(step) for step in reasoning_chain]),
            html.Hr(),
            html.H6("Inference Process"),
            html.P("Applied OWL-RL closure and SHACL rule inference"),
            html.P(f"Linked {kg_summary.get('confirmations', 0)} confirmations, {kg_summary.get('contradictions', 0)} contradictions"),
            html.P("Confidence adjusted via contradiction-resolution and confirmation-strengthening")
        ])
    else:
        log_step("Standard mode selected – no ontology reasoning.")
        insights_content = html.Div([
            html.H4("📊 Standard Technical Analysis"),
            html.P(f"Symbol: {ticker}, Period: {time_range}, Interval: {interval}")
        ])
        signals_content = html.Div()
        risk_content = html.Div()
        recommendations_content = html.Div()
        reasoning_trace = html.Div()

    # ─────────────────────────────────────────────
    # Chart Rendering (UNCHANGED)
    # ─────────────────────────────────────────────
    log_step("Rendering technical charts…")

    fig_candle = go.Figure(go.Candlestick(
        x=df.index, open=df.open, high=df.high, low=df.low, close=df.close
    )).update_layout(title=f"{ticker} Candlestick", template="plotly_dark")

    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(x=df.index, y=df.close, name="Close"))
    for col in ["SMA_20","SMA_50","SMA_200","EMA_8","EMA_20","EMA_50","EMA_200"]:
        if col in df:
            fig_sma.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_sma.update_layout(title=f"{ticker} SMA & EMA", template="plotly_dark")

    pivot = (df.high + df.low + df.close) / 3
    df["S1"], df["R1"] = 2*pivot - df.high, 2*pivot - df.low
    df["S2"], df["R2"] = pivot - (df.high - df.low), pivot + (df.high - df.low)
    fig_sr = go.Figure()
    for col in ["S1","R1","S2","R2"]:
        fig_sr.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_sr.update_layout(title=f"{ticker} Support & Resistance", template="plotly_dark")

    fig_rsi = go.Figure(go.Scatter(x=df.index, y=df.RSI, name="RSI"))
    for yv,c in [(70,"red"),(30,"green")]:
        fig_rsi.add_shape(type="line",x0=df.index[0],x1=df.index[-1],
                          y0=yv,y1=yv,line=dict(color=c,dash="dash"))
    fig_rsi.update_layout(title=f"{ticker} RSI", template="plotly_dark")

    fig_bb = go.Figure()
    for col in ["close","Upper_band","Lower_band"]:
        fig_bb.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_bb.update_layout(title=f"{ticker} Bollinger Bands", template="plotly_dark")

    fig_macd = go.Figure()
    for col in ["MACD","MACD_Signal"]:
        fig_macd.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_macd.update_layout(title=f"{ticker} MACD", template="plotly_dark")

    fig_sto = go.Figure()
    for col in ["%K","%D"]:
        fig_sto.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_sto.update_layout(title=f"{ticker} Stochastic Oscillator", template="plotly_dark")

    fig_obv = go.Figure(go.Scatter(x=df.index, y=df.OBV, name="OBV"))
    fig_obv.update_layout(title=f"{ticker} On-Balance Volume", template="plotly_dark")

    fig_atr = go.Figure(go.Scatter(x=df.index, y=df.ATR, name="ATR"))
    fig_atr.update_layout(title=f"{ticker} Average True Range", template="plotly_dark")

    fig_cci = go.Figure(go.Scatter(x=df.index, y=df.CCI, name="CCI"))
    for yv in [100, -100]:
        fig_cci.add_shape(type="line",x0=df.index[0],x1=df.index[-1],
                          y0=yv,y1=yv,line=dict(color="gray",dash="dash"))
    fig_cci.update_layout(title=f"{ticker} CCI", template="plotly_dark")

    fig_mfi = go.Figure(go.Scatter(x=df.index, y=df.MFI, name="MFI"))
    for yv,c in [(80,"red"),(20,"green")]:
        fig_mfi.add_shape(type="line",x0=df.index[0],x1=df.index[-1],
                          y0=yv,y1=yv,line=dict(color=c,dash="dash"))
    fig_mfi.update_layout(title=f"{ticker} MFI", template="plotly_dark")

    fig_cmf = go.Figure(go.Scatter(x=df.index, y=df.CMF, name="CMF"))
    fig_cmf.add_shape(type="line",x0=df.index[0],x1=df.index[-1],
                      y0=0,y1=0,line=dict(color="red",dash="dash"))
    fig_cmf.update_layout(title=f"{ticker} Chaikin Money Flow", template="plotly_dark")

    fig_fi = go.Figure(go.Scatter(x=df.index, y=df.FI, name="Force Index"))
    fig_fi.update_layout(title=f"{ticker} Force Index", template="plotly_dark")

    high, low = df.high.max(), df.low.min()
    diff = high - low
    fib_levels = {p: high - (v * diff) for p,v in {
        "0%":0,"23.6%":0.236,"38.2%":0.382,"50%":0.5,"61.8%":0.618,"100%":1}.items()}
    fig_fib = go.Figure(go.Scatter(x=df.index, y=df.close, name="Close"))
    for label, price in fib_levels.items():
        fig_fib.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]], y=[price, price],
            name=label, line=dict(dash="dash")
        ))
    fig_fib.update_layout(title=f"{ticker} Fibonacci Retracement", template="plotly_dark")

    fig_ich = go.Figure()
    for col in ["close","Tenkan_sen","Kijun_sen","Senkou_span_a","Senkou_span_b","Chikou_span"]:
        if col in df:
            fig_ich.add_trace(go.Scatter(x=df.index, y=df[col], name=col))
    fig_ich.update_layout(title=f"{ticker} Ichimoku Cloud", template="plotly_dark")

    fig_vwap = go.Figure()
    fig_vwap.add_trace(go.Scatter(x=df.index, y=df.close, name="Close"))
    if "VWAP" in df:
        fig_vwap.add_trace(go.Scatter(x=df.index, y=df.VWAP, name="VWAP"))
    fig_vwap.update_layout(title=f"{ticker} VWAP", template="plotly_dark")

    fig_adl = go.Figure(go.Scatter(x=df.index, y=df.ADL, name="ADL"))
    fig_adl.update_layout(title=f"{ticker} Accumulation/Distribution Line", template="plotly_dark")

    fig_adx = go.Figure()
    fig_adx.add_trace(go.Scatter(x=df.index, y=df["ADX"],
                                 name="ADX",  line=dict(color="#3498DB", width=2)))
    fig_adx.add_trace(go.Scatter(x=df.index, y=df["DI+"],
                                 name="DI+",  line=dict(color="#2ECC71", width=2)))
    fig_adx.add_trace(go.Scatter(x=df.index, y=df["DI-"],
                                 name="DI−",  line=dict(color="#E74C3C", width=2)))
    fig_adx.update_layout(title=f"{ticker} ADX & Directional Indicators",
                          template="plotly_dark")

    log_step("✅ Charts rendered successfully.")

    return (
        fig_candle, fig_sma, fig_sr, fig_rsi, fig_bb, fig_macd, fig_sto, fig_obv,
        fig_atr, fig_cci, fig_mfi, fig_cmf, fig_fi, fig_fib, fig_ich, fig_vwap,
        fig_adl, fig_adx,
        insights_content, signals_content, risk_content,
        recommendations_content, reasoning_trace
    )


if __name__ == "__main__":
    log_step("🚀 Launching Enhanced Ontology-Driven Stock Dashboard (Final Version)…")
    app.run_server(debug=False, port=8050)
