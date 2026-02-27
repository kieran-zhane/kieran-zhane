#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 17:15:10 2026

@author: kieran.zhane
"""

"""
Tech start-up MVP V0.1

author: Kieran Zhane
"""

"""
MVP Starter Script (Fictional Example) — Solar Underwriting Augmentation

What this script does:
- Estimates annual solar generation from postcode region + system size (simple proxy)
- Converts generation into conservative monthly savings (self-consumption + export)
- Computes net affordability impact: (loan repayment - savings)
- Assigns an asset risk tier (A/B/C) from simple rules
- Outputs a rules-based PD adjustment recommendation
- Returns a "decision" suggestion: APPROVE / REFER / DECLINE

IMPORTANT:
- This is NOT a production credit model.
- It is a deterministic, explainable "augmentation layer" you can run in shadow mode.
- All numbers are fictional and conservative by design.
"""

import json
from dataclasses import dataclass, asdict
from math import pow
from typing import Dict, Any, Tuple, Optional


# -----------------------------
# Data models
# -----------------------------

@dataclass(frozen=True)
class LoanInput:
    # Borrower / affordability inputs
    monthly_income_gbp: float
    monthly_existing_debt_gbp: float
    current_energy_bill_monthly_gbp: float

    # Loan terms
    loan_amount_gbp: float
    term_months: int
    annual_interest_rate: float  # e.g., 0.079 for 7.9% APR

    # Asset + property inputs
    postcode: str
    system_size_kw: float
    epc_rating: Optional[str]  # e.g., "A".."G" or None
    installer_mcs_certified: bool


@dataclass
class UnderwriteOutput:
    decision: str  # "APPROVE" | "REFER" | "DECLINE"
    reason: str

    monthly_repayment_gbp: float
    monthly_savings_conservative_gbp: float
    net_monthly_cost_after_savings_gbp: float

    repayment_to_income_before: float
    repayment_to_income_after: float

    asset_risk_tier: str  # "A"|"B"|"C"
    baseline_pd: float
    pd_adjustment: float
    climate_adjusted_pd: float

    debug: Dict[str, Any]


# -----------------------------
# Helper functions
# -----------------------------

def annuity_payment(principal: float, annual_rate: float, months: int) -> float:
    """
    Standard amortising loan payment.
    """
    if months <= 0:
        raise ValueError("term_months must be > 0")
    if principal <= 0:
        raise ValueError("loan_amount_gbp must be > 0")
    if annual_rate < 0:
        raise ValueError("annual_interest_rate must be >= 0")

    if annual_rate == 0:
        return principal / months

    r = annual_rate / 12.0
    return principal * (r * pow(1 + r, months)) / (pow(1 + r, months) - 1)


def postcode_region(postcode: str) -> str:
    """
    Very rough UK region proxy from postcode prefix.
    For MVP demo only. Extend with proper mapping later.
    """
    p = postcode.strip().upper()
    if not p:
        return "UNKNOWN"

    # Extremely simplified heuristics
    if p.startswith(("NE", "SR", "DH", "TS")):
        return "NORTH_EAST"
    if p.startswith(("NW", "M", "L")):
        return "NORTH_WEST"
    if p.startswith(("YO", "LS", "HU", "DN")):
        return "YORKSHIRE"
    if p.startswith(("B", "CV", "DE", "NG", "LE")):
        return "MIDLANDS"
    if p.startswith(("SW", "EX", "PL", "TR", "TA")):
        return "SOUTH_WEST"
    if p.startswith(("SE", "BN", "RH", "TN", "CT")):
        return "SOUTH_EAST"
    if p.startswith(("E", "N", "W", "EC", "WC")):
        return "LONDON"
    if p.startswith(("G", "EH", "FK", "AB", "DD", "PH", "IV")):
        return "SCOTLAND"
    if p.startswith(("CF", "SA", "NP", "LL")):
        return "WALES"
    return "OTHER"


def estimate_annual_generation_kwh(system_size_kw: float, region: str) -> Tuple[float, float]:
    """
    Returns (p50_kwh, p10_conservative_kwh).

    Uses fictional 'specific yield' (kWh/kWp/year) by broad region.
    Typical UK yields might be ~850–1050 kWh/kWp; we keep ranges plausible but simple.
    """
    if system_size_kw <= 0:
        raise ValueError("system_size_kw must be > 0")

    specific_yield_p50 = {
        "SCOTLAND": 850,
        "NORTH_EAST": 900,
        "NORTH_WEST": 880,
        "YORKSHIRE": 900,
        "MIDLANDS": 950,
        "LONDON": 970,
        "SOUTH_EAST": 1000,
        "SOUTH_WEST": 1020,
        "WALES": 900,
        "OTHER": 930,
        "UNKNOWN": 900,
    }.get(region, 900)

    # P10 conservatism: knock down by 12% to represent weather variance + suboptimal install.
    p50 = system_size_kw * specific_yield_p50
    p10 = p50 * 0.88

    return p50, p10


def estimate_monthly_savings_conservative(
    annual_generation_kwh_p10: float,
    tariff_import_gbp_per_kwh: float = 0.24,
    tariff_export_gbp_per_kwh: float = 0.12,
    self_consumption_rate: float = 0.55,
    additional_conservatism: float = 0.90,
) -> float:
    """
    Convert generation into conservative monthly £ savings.
    - self-consumption offsets import tariff
    - exports earn export tariff
    - apply additional conservatism (e.g., behaviour uncertainty)
    """
    if not (0 <= self_consumption_rate <= 1):
        raise ValueError("self_consumption_rate must be between 0 and 1")
    if annual_generation_kwh_p10 < 0:
        raise ValueError("annual_generation_kwh_p10 must be >= 0")

    self_use_kwh = annual_generation_kwh_p10 * self_consumption_rate
    export_kwh = annual_generation_kwh_p10 * (1 - self_consumption_rate)

    annual_value = (self_use_kwh * tariff_import_gbp_per_kwh) + (export_kwh * tariff_export_gbp_per_kwh)

    # Extra conservatism layer
    annual_value *= additional_conservatism

    return annual_value / 12.0


def asset_risk_tier(system_size_kw: float, installer_mcs_certified: bool, epc_rating: Optional[str]) -> str:
    """
    Simple, explainable risk tiering.
    """
    epc = (epc_rating or "").strip().upper()

    # Base tier
    tier = "B"

    # Installer quality
    if installer_mcs_certified:
        tier = "A"
    else:
        tier = "B"

    # Unusual / high-risk configurations (very simplistic)
    if system_size_kw < 2.0 or system_size_kw > 8.0:
        tier = "B" if tier == "A" else "C"

    # EPC as a proxy for energy efficiency; very poor EPC can imply higher energy stress,
    # but also higher savings potential. For MVP, treat very poor EPC as slightly higher risk.
    if epc in ("F", "G"):
        tier = "B" if tier == "A" else "C"

    return tier


def baseline_pd_from_debt_ratio(
    monthly_income: float,
    monthly_existing_debt: float,
    monthly_repayment: float,
) -> float:
    """
    Fictional baseline PD curve based on (debt + new payment) / income.
    This is a placeholder to illustrate how your augmentation would feed into PD.
    """
    if monthly_income <= 0:
        raise ValueError("monthly_income_gbp must be > 0")

    dti = (monthly_existing_debt + monthly_repayment) / monthly_income

    # Piecewise simplistic mapping
    if dti < 0.20:
        return 0.020
    if dti < 0.30:
        return 0.035
    if dti < 0.40:
        return 0.055
    if dti < 0.50:
        return 0.080
    return 0.120


def pd_adjustment_rules(
    savings_to_payment_ratio: float,
    asset_tier: str,
    repayment_ratio_improvement: float,
) -> float:
    """
    Rules-based PD adjustment (negative is better).
    - If savings cover a large portion of the payment and asset risk is low, reduce PD.
    - Keep conservatively bounded.
    """
    adj = 0.0

    if asset_tier == "A":
        if savings_to_payment_ratio >= 0.70:
            adj -= 0.015
        elif savings_to_payment_ratio >= 0.50:
            adj -= 0.010
        elif savings_to_payment_ratio >= 0.35:
            adj -= 0.006
    elif asset_tier == "B":
        if savings_to_payment_ratio >= 0.70:
            adj -= 0.010
        elif savings_to_payment_ratio >= 0.50:
            adj -= 0.006
        elif savings_to_payment_ratio >= 0.35:
            adj -= 0.003
    else:  # "C"
        # Very limited benefit assumed
        if savings_to_payment_ratio >= 0.70 and repayment_ratio_improvement >= 0.02:
            adj -= 0.004

    # If repayment ratio improvement is negligible, dampen adjustment.
    if repayment_ratio_improvement < 0.01:
        adj *= 0.6

    # Bound adjustment to be conservative
    return max(-0.02, min(0.0, adj))


def decision_logic(
    repayment_to_income_after: float,
    climate_adjusted_pd: float,
    asset_tier: str,
) -> Tuple[str, str]:
    """
    Fictional decisioning:
    - APPROVE if affordability is comfortable and PD is low
    - REFER if borderline
    - DECLINE if high stress
    """
    # Hard guards
    if repayment_to_income_after > 0.55:
        return "DECLINE", "Repayment-to-income too high after savings adjustment."

    if climate_adjusted_pd >= 0.11:
        return "DECLINE", "High projected default risk."

    # Approve bands
    if repayment_to_income_after <= 0.35 and climate_adjusted_pd <= 0.06 and asset_tier in ("A", "B"):
        return "APPROVE", "Affordable and low projected risk with asset-aware adjustment."

    # Otherwise refer for manual
    return "REFER", "Borderline case: refer to underwriting for manual review."


# -----------------------------
# Main underwrite function
# -----------------------------

def underwrite_solar(input_data: LoanInput) -> UnderwriteOutput:
    region = postcode_region(input_data.postcode)

    monthly_payment = annuity_payment(
        principal=input_data.loan_amount_gbp,
        annual_rate=input_data.annual_interest_rate,
        months=input_data.term_months,
    )

    # Ratios before any savings adjustment (classic affordability)
    repayment_to_income_before = (input_data.monthly_existing_debt_gbp + monthly_payment) / input_data.monthly_income_gbp

    # Yield + savings
    p50_kwh, p10_kwh = estimate_annual_generation_kwh(input_data.system_size_kw, region)

    monthly_savings = estimate_monthly_savings_conservative(
        annual_generation_kwh_p10=p10_kwh,
        tariff_import_gbp_per_kwh=0.24,
        tariff_export_gbp_per_kwh=0.12,
        self_consumption_rate=0.55,
        additional_conservatism=0.90,
    )

    # Net cost after savings
    net_monthly_cost = monthly_payment - monthly_savings

    # Adjusted affordability ratio after savings
    repayment_to_income_after = (input_data.monthly_existing_debt_gbp + max(net_monthly_cost, 0.0)) / input_data.monthly_income_gbp

    # Asset tier
    tier = asset_risk_tier(input_data.system_size_kw, input_data.installer_mcs_certified, input_data.epc_rating)

    # Baseline PD from standard DTI measure (placeholder)
    baseline_pd = baseline_pd_from_debt_ratio(
        monthly_income=input_data.monthly_income_gbp,
        monthly_existing_debt=input_data.monthly_existing_debt_gbp,
        monthly_repayment=monthly_payment,
    )

    # PD adjustment
    savings_to_payment_ratio = monthly_savings / monthly_payment if monthly_payment > 0 else 0.0
    repayment_ratio_improvement = max(0.0, repayment_to_income_before - repayment_to_income_after)

    pd_adj = pd_adjustment_rules(
        savings_to_payment_ratio=savings_to_payment_ratio,
        asset_tier=tier,
        repayment_ratio_improvement=repayment_ratio_improvement,
    )

    climate_adjusted_pd = max(0.0, baseline_pd + pd_adj)

    # Decision
    decision, reason = decision_logic(
        repayment_to_income_after=repayment_to_income_after,
        climate_adjusted_pd=climate_adjusted_pd,
        asset_tier=tier,
    )

    return UnderwriteOutput(
        decision=decision,
        reason=reason,
        monthly_repayment_gbp=round(monthly_payment, 2),
        monthly_savings_conservative_gbp=round(monthly_savings, 2),
        net_monthly_cost_after_savings_gbp=round(net_monthly_cost, 2),
        repayment_to_income_before=round(repayment_to_income_before, 4),
        repayment_to_income_after=round(repayment_to_income_after, 4),
        asset_risk_tier=tier,
        baseline_pd=round(baseline_pd, 4),
        pd_adjustment=round(pd_adj, 4),
        climate_adjusted_pd=round(climate_adjusted_pd, 4),
        debug={
            "region": region,
            "annual_generation_kwh_p50": round(p50_kwh, 0),
            "annual_generation_kwh_p10": round(p10_kwh, 0),
            "savings_to_payment_ratio": round(savings_to_payment_ratio, 4),
            "repayment_ratio_improvement": round(repayment_ratio_improvement, 4),
            "assumptions": {
                "tariff_import_gbp_per_kwh": 0.24,
                "tariff_export_gbp_per_kwh": 0.12,
                "self_consumption_rate": 0.55,
                "additional_conservatism": 0.90,
                "p10_discount": 0.12,
            },
        },
    )

def format_api_output(result: UnderwriteOutput) -> str:
    """
    Returns properly formatted JSON string for API-style output.
    """
    return json.dumps(asdict(result), indent=2)


def format_human_output(result: UnderwriteOutput) -> None:
    """
    Pretty console output for business / lender demo.
    """
    print("\n========== SOLAR UNDERWRITING AUGMENTATION ==========")
    print(f"Decision: {result.decision}")
    print(f"Reason: {result.reason}")
    print("\n--- Affordability ---")
    print(f"Monthly Repayment: £{result.monthly_repayment_gbp}")
    print(f"Conservative Monthly Savings: £{result.monthly_savings_conservative_gbp}")
    print(f"Net Monthly Cost After Savings: £{result.net_monthly_cost_after_savings_gbp}")
    print(f"Repayment-to-Income (Before): {result.repayment_to_income_before:.2%}")
    print(f"Repayment-to-Income (After):  {result.repayment_to_income_after:.2%}")
    print("\n--- Risk ---")
    print(f"Asset Risk Tier: {result.asset_risk_tier}")
    print(f"Baseline PD: {result.baseline_pd:.2%}")
    print(f"PD Adjustment: {result.pd_adjustment:.2%}")
    print(f"Climate-Adjusted PD: {result.climate_adjusted_pd:.2%}")
    print("=====================================================\n")
    

# -----------------------------
# Fictional example run
# -----------------------------

if __name__ == "__main__":
    example = LoanInput(
        monthly_income_gbp=3200.0,
        monthly_existing_debt_gbp=350.0,
        current_energy_bill_monthly_gbp=180.0,
        loan_amount_gbp=9800.0,
        term_months=84,
        annual_interest_rate=0.079,
        postcode="NE1 1AA",
        system_size_kw=4.2,
        epc_rating="D",
        installer_mcs_certified=True,
    )

    result = underwrite_solar(example)

    # API-style output
    print("API RESPONSE:")
    print(format_api_output(result))

    # Human-friendly output
    format_human_output(result)
