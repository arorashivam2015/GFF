"""Machine-readable attack-vector schema.

This module is the contract between the three pillars:

    IDENTIFY  writes AttackVector specs (YAML)
    GENERATE  reads  spec.simulation  to synthesise attack traces
    DEFEND    reads  spec.signals     to know which channels must be instrumented,
              and    spec.detection_hypotheses to seed feature engineering

Keeping the taxonomy machine-readable is what makes the loop a single system
rather than three disconnected projects.

Targets Python 3.9: no PEP-604 unions, no builtin generics in annotations.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# Morphological dimensions
#
# The taxonomy is generative, not enumerative: vectors are points in the
# cross-product of these axes. Coverage over the cross-product is what the
# "diversity of attacks identified" criterion is actually measuring.
# --------------------------------------------------------------------------


class Rail(str, Enum):
    """Payment rail / instrument the attack rides on."""

    CARD_CNP = "card_cnp"                  # card-not-present e-commerce
    CARD_CP = "card_cp"                    # card-present, POS
    CARD_TOKEN = "card_token"              # network / CoF tokenised credential
    UPI_P2P = "upi_p2p"
    UPI_P2M = "upi_p2m"
    UPI_COLLECT = "upi_collect"            # pull request — the pretexting surface
    UPI_AUTOPAY = "upi_autopay"            # recurring e-mandate on UPI
    UPI_LITE = "upi_lite"                  # on-device low-value, no PIN
    CREDIT_ON_UPI = "credit_on_upi"        # RuPay credit card linked to UPI
    AEPS = "aeps"                          # Aadhaar-enabled payment system
    IMPS = "imps"
    NEFT_RTGS = "neft_rtgs"
    NACH_EMANDATE = "nach_emandate"
    WALLET_PPI = "wallet_ppi"              # prepaid payment instrument
    AGENTIC_CHECKOUT = "agentic_checkout"  # AI-agent-initiated commerce
    CROSS_BORDER = "cross_border"
    ONBOARDING = "onboarding"              # KYC / merchant acquisition (pre-rail)


class LifecycleStage(str, Enum):
    """Where in the fraud kill-chain the technique operates."""

    RECON = "recon"                # target selection, OSINT, probing
    ACQUISITION = "acquisition"    # obtain credential / identity / victim consent
    STAGING = "staging"            # mule accounts, device farms, merchant shells
    EXECUTION = "execution"        # the money movement itself
    CASHOUT = "cashout"            # layering and extraction
    EVASION = "evasion"            # persistence and detection avoidance


class GenAIUplift(str, Enum):
    """What generative AI specifically adds. A vector with no uplift is
    classical fraud and belongs outside this taxonomy."""

    CONTENT_GENERATION = "content_generation"      # personalised text at scale
    VOICE_CLONING = "voice_cloning"
    VIDEO_SYNTHESIS = "video_synthesis"            # deepfake video / injection
    CONVERSATIONAL_AGENT = "conversational_agent"  # sustained multi-turn manipulation
    DOCUMENT_FORGERY = "document_forgery"
    TARGET_RESEARCH = "target_research"            # OSINT enrichment, victim scoring
    CODE_GENERATION = "code_generation"
    ADAPTIVE_EVASION = "adaptive_evasion"          # mutate against defender feedback
    ORCHESTRATION = "orchestration"                # agentic multi-step planning
    PROMPT_INJECTION = "prompt_injection"          # attacks *on* AI systems


class VictimSurface(str, Enum):
    CONSUMER = "consumer"
    MERCHANT = "merchant"
    ISSUER = "issuer"
    ACQUIRER = "acquirer"
    PSP = "psp"
    KYC_PROVIDER = "kyc_provider"
    AI_AGENT = "ai_agent"          # the buyer's or merchant's autonomous agent
    FRAUD_MODEL = "fraud_model"    # the defence itself is the target


class ActorTier(str, Enum):
    OPPORTUNIST = "opportunist"            # low skill, off-the-shelf tooling
    ORGANIZED = "organized"                # funded crime group, division of labour
    INSIDER = "insider"
    FAAS_VENDOR = "faas_vendor"            # fraud-as-a-service seller


class SignalChannel(str, Enum):
    """Where the attack becomes observable. Drives both simulation output
    schema and detector feature-store design."""

    TRANSACTION = "transaction"    # single tabular auth / settlement record
    SEQUENCE = "sequence"          # per-entity temporal pattern
    GRAPH = "graph"                # entity-relationship structure
    DEVICE = "device"
    IDENTITY = "identity"          # onboarding / KYC artefacts
    TEXT = "text"                  # SMS, chat, email, chargeback narrative
    BIOMETRIC = "biometric"        # liveness and match scores
    AGENT_TRACE = "agent_trace"    # AI-agent interaction logs


class Maturity(str, Enum):
    """Honesty axis. Judges asked for *emerging* fraud — we must be able to
    say which vectors are already in the wild and which are projected."""

    OBSERVED = "observed"      # documented in the wild
    EMERGING = "emerging"      # early real-world reports, scaling now
    PROJECTED = "projected"    # plausible, capability exists, not yet widespread


class AmountProfile(str, Enum):
    MICRO_PROBE = "micro_probe"              # sub-limit validation attempts
    JUST_UNDER_LIMIT = "just_under_limit"    # structured beneath a rule threshold
    TYPICAL = "typical"                      # blends into the victim's baseline
    DRAIN = "drain"                          # maximum extraction
    ESCALATING = "escalating"                # trust-building ramp
    ROUND_SUM = "round_sum"


class TemporalShape(str, Enum):
    BURST = "burst"                          # high velocity, short window
    SLOW_DRIP = "slow_drip"
    DORMANT_THEN_SPIKE = "dormant_then_spike"  # classic bust-out
    BUSINESS_HOURS = "business_hours"
    OFF_HOURS = "off_hours"
    SUSTAINED_CAMPAIGN = "sustained_campaign"


# --------------------------------------------------------------------------
# Sub-models
# --------------------------------------------------------------------------


class SimulationProfile(BaseModel):
    """Consumed by redlab.sim to render this vector into concrete traces."""

    model_config = ConfigDict(extra="forbid")

    entities: List[str] = Field(
        ...,
        description="Simulator entity types involved, e.g. cardholder, merchant, "
        "device, mule_account, beneficiary, ai_agent.",
        min_length=1,
    )
    amount_profile: AmountProfile
    temporal_shape: TemporalShape
    duration_days: List[int] = Field(
        ..., description="[min, max] campaign duration in days.", min_length=2, max_length=2
    )
    events_per_campaign: List[int] = Field(
        ..., description="[min, max] observable events produced per campaign.",
        min_length=2, max_length=2,
    )
    entity_reuse: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0 = fresh entity per event (hard for graph detection), "
        "1 = heavy reuse (easy for graph detection). Drives graph density.",
    )
    notes: Optional[str] = None

    @field_validator("duration_days", "events_per_campaign")
    @classmethod
    def _ordered_range(cls, v: List[int]) -> List[int]:
        lo, hi = v
        if lo <= 0:
            raise ValueError("range lower bound must be positive")
        if lo > hi:
            raise ValueError(f"range is inverted: [{lo}, {hi}]")
        return v


class DetectionHypothesis(BaseModel):
    """A falsifiable claim about what should catch this vector. Seeds the
    DEFEND feature set and, critically, gives us something to be wrong about."""

    model_config = ConfigDict(extra="forbid")

    channel: SignalChannel
    feature: str = Field(..., description="Feature or family of features.")
    rationale: str


class Scores(BaseModel):
    """0-5 integer ratings. Used for prioritisation and for the coverage report;
    deliberately coarse because finer precision would be false precision."""

    model_config = ConfigDict(extra="forbid")

    novelty: int = Field(..., ge=0, le=5)
    impact: int = Field(..., ge=0, le=5, description="Loss magnitude if successful.")
    scalability: int = Field(..., ge=0, le=5, description="How well GenAI scales it.")
    detection_difficulty: int = Field(..., ge=0, le=5)

    @property
    def priority(self) -> float:
        """Ranking heuristic for which vectors to simulate first."""
        return round(
            0.35 * self.novelty
            + 0.25 * self.impact
            + 0.20 * self.scalability
            + 0.20 * self.detection_difficulty,
            3,
        )


# --------------------------------------------------------------------------
# The attack vector
# --------------------------------------------------------------------------

_ID_PREFIXES = {
    "social_engineering": "SE",
    "account_takeover": "ATO",
    "synthetic_identity": "SID",
    "merchant_abuse": "MER",
    "card_testing": "CT",
    "india_rails": "IND",
    "agentic_commerce": "AGC",
    "anti_defense": "ADV",
}


class Family(str, Enum):
    SOCIAL_ENGINEERING = "social_engineering"
    ACCOUNT_TAKEOVER = "account_takeover"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    MERCHANT_ABUSE = "merchant_abuse"
    CARD_TESTING = "card_testing"
    INDIA_RAILS = "india_rails"
    AGENTIC_COMMERCE = "agentic_commerce"
    ANTI_DEFENSE = "anti_defense"


class AttackVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^PF-[A-Z]{2,3}-\d{3}$")
    name: str
    family: Family
    maturity: Maturity
    summary: str = Field(..., description="One or two sentences, mechanism-first.")

    # Morphological coordinates
    rails: List[Rail] = Field(..., min_length=1)
    stages: List[LifecycleStage] = Field(..., min_length=1)
    genai_uplift: List[GenAIUplift] = Field(..., min_length=1)
    victim_surfaces: List[VictimSurface] = Field(..., min_length=1)
    actor_tiers: List[ActorTier] = Field(..., min_length=1)

    preconditions: List[str] = Field(default_factory=list)
    signals: List[SignalChannel] = Field(..., min_length=1)
    simulation: SimulationProfile
    detection_hypotheses: List[DetectionHypothesis] = Field(..., min_length=1)
    mitigations: List[str] = Field(default_factory=list)
    scores: Scores
    references: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _id_matches_family(self) -> "AttackVector":
        expected = _ID_PREFIXES[self.family.value]
        actual = self.id.split("-")[1]
        if actual != expected:
            raise ValueError(
                f"{self.id}: family '{self.family.value}' expects prefix "
                f"'PF-{expected}-', got 'PF-{actual}-'"
            )
        return self

    @model_validator(mode="after")
    def _hypotheses_cover_declared_signals(self) -> "AttackVector":
        """A declared signal channel with no detection hypothesis is an
        instrumentation gap we would otherwise ship silently."""
        hypothesised = {h.channel for h in self.detection_hypotheses}
        missing = set(self.signals) - hypothesised
        if missing:
            raise ValueError(
                f"{self.id}: signal channels {sorted(m.value for m in missing)} "
                "have no detection hypothesis"
            )
        return self

    @property
    def priority(self) -> float:
        return self.scores.priority


class VectorFile(BaseModel):
    """Top level of each YAML file under taxonomy/vectors/."""

    model_config = ConfigDict(extra="forbid")

    family: Family
    description: Optional[str] = None
    vectors: List[AttackVector] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _all_vectors_match_file_family(self) -> "VectorFile":
        bad = [v.id for v in self.vectors if v.family != self.family]
        if bad:
            raise ValueError(
                f"vectors {bad} do not match file family '{self.family.value}'"
            )
        return self


# Exposed for the coverage report so it stays in sync with the enums above.
COVERAGE_AXES: Dict[str, type] = {
    "rails": Rail,
    "stages": LifecycleStage,
    "genai_uplift": GenAIUplift,
    "victim_surfaces": VictimSurface,
    "signals": SignalChannel,
}
