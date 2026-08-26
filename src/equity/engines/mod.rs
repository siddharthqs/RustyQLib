//! How it gets priced: the numerical and analytic pricing methods.

pub mod baw;
pub mod binomial;
pub mod bjerksund_stensland;
pub mod black76;
pub mod blackscholes;
pub mod carr_madan;
pub mod cos;
pub mod finite_difference;
pub mod heston_adi;
pub mod montecarlo;

use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use crate::equity::blackscholes::bs_price;
use crate::equity::bump::BumpedMarket;
use crate::equity::vanilla_option::EquityOption;

/// Shared `npv` of the analytic American engines (BAW and BS2002). Flat
/// Black-Scholes inputs are read the same way the analytic vanilla pricer
/// reads them: escrowed spot (cash dividends carved out), the curve's
/// continuous zero rate, the total carry, and the surface vol at this
/// strike. `american_price` supplies the engine's American closed form;
/// European contracts price through [`bs_price`].
pub(crate) fn american_analytic_npv(
    option: &EquityOption,
    bumped_market: Option<&BumpedMarket>,
    american_price: fn(f64, f64, f64, f64, f64, f64, PutOrCall) -> f64,
    engine: &str,
) -> f64 {
    let base = BumpedMarket::base(&option.market);
    let m = bumped_market.unwrap_or(&base);
    let maturity = option.base.maturity_date;
    let s = m.effective_spot(maturity);
    let k = option.base.strike_price;
    let r = m.risk_free_rate(maturity);
    let q = m.carry_yield();
    let sigma = m.volatility(k, maturity);
    let t = m.time_to_maturity(maturity).max(1e-8);
    let pc = *option.payoff.put_or_call();
    match option.payoff.exercise_style() {
        ContractStyle::American => american_price(s, k, r, q, sigma, t, pc),
        // an American approximation on a European contract is just the European price
        ContractStyle::European => bs_price(s, k, r, q, sigma, t, pc),
        // invariant: check_engine_support refuses Bermudan on these engines
        ContractStyle::Bermudan(_) => {
            unreachable!("Bermudan exercise is rejected on the {engine} engine before pricing")
        }
    }
}
