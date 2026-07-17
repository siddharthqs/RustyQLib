use std::error::Error;
use chrono::{Datelike, Local, NaiveDate};
use crate::equity::{binomial,finite_difference,montecarlo};
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::vols::VolSurface;
use super::super::core::quotes::Quote;
use super::super::core::traits::{Instrument,Greeks};
use super::blackscholes;
use crate::equity::utils::{Engine, PayoffType, Payoff, LongShort};
use crate::core::trade::{PutOrCall,Transection};
use crate::core::utils::{Contract,ContractStyle};
use crate::core::trade;
use serde::Deserialize;
use blackscholes::BlackScholesPricer;
use crate::core::data_models::EquityOptionData;

#[derive(Debug)]
pub struct VanillaPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
}
#[derive(Debug)]
pub struct BinaryPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
}
#[derive(Debug)]
pub struct BarrierPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
}
#[derive(Debug)]
pub struct AsianPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
}
impl Payoff for VanillaPayoff {
    fn payoff(&self, spot: f64, strike: f64) -> f64 {
        match &self.put_or_call {
            PutOrCall::Call => (spot - strike).max(0.0),
            PutOrCall::Put => (strike - spot).max(0.0),
        }
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Vanilla
    }
    fn put_or_call(&self) -> &PutOrCall {
        &self.put_or_call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }

}

/// Cash-or-nothing binary: pays one unit of currency if the option finishes
/// in the money (strictly beyond the strike), nothing otherwise.
impl Payoff for BinaryPayoff {
    fn payoff(&self, spot: f64, strike: f64) -> f64 {
        let in_the_money = match &self.put_or_call {
            PutOrCall::Call => spot > strike,
            PutOrCall::Put => spot < strike,
        };
        if in_the_money { 1.0 } else { 0.0 }
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Binary
    }
    fn put_or_call(&self) -> &PutOrCall {
        &self.put_or_call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }
}

#[derive(Debug)]
pub struct EquityOptionBase {
    pub symbol: String,
    pub currency: Option<String>,
    pub exchange: Option<String>,
    pub name: Option<String>,
    pub cusip: Option<String>,
    pub isin: Option<String>,
    pub settlement_type: Option<String>,

    //pub payoff_type: String, // Vanilla/Barrier/Binary
    pub underlying_price: Quote,
    pub current_price: Quote,
    pub strike_price: f64,
    pub dividend_yield: f64,
    /// Volatility surface; a flat surface represents a single constant vol.
    pub vol_surface: VolSurface,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    /// Discounting curve anchored at `valuation_date`; discount factors are
    /// the source of truth, rates are derived views.
    pub discount_curve: YieldCurve,
    pub entry_price: f64,
    pub long_short: LongShort,
    pub multiplier: f64,

}
#[derive(Debug)]
pub struct EquityOption {
    pub base: EquityOptionBase,
    pub payoff: Box<dyn Payoff>,
    pub engine: Engine,
    /// Monte Carlo settings (paths, time steps, scheme, sampler, seed);
    /// only consulted when `engine` is [`Engine::MonteCarlo`].
    pub mc: montecarlo::MonteCarloConfig,
}
impl EquityOption{
    pub fn time_to_maturity(&self) -> f64{
        let time_to_maturity = (self.base.maturity_date - self.base.valuation_date).num_days() as f64/365.0;
        time_to_maturity
    }

}
impl EquityOption {

    pub fn from_json(data: &EquityOptionData) -> Box<EquityOption> {
        let valuation_date = Local::now().date_naive();
        let discount_curve = match &data.discount_curve {
            Some(input) => YieldCurve::from_input(input, valuation_date)
                .expect("Invalid discount curve"),
            None => YieldCurve::flat(
                data.base.risk_free_rate.unwrap_or(0.0),
                valuation_date,
                DayCountConvention::Act365,
                Compounding::Continuous,
            )
            .expect("Invalid risk free rate"),
        };
        let vol_surface = match &data.vol_surface {
            Some(input) => VolSurface::from_input(input, valuation_date)
                .expect("Invalid vol surface"),
            None => VolSurface::flat(
                data.volatility
                    .expect("Either volatility or vol_surface must be provided"),
                valuation_date,
                DayCountConvention::Act365,
            )
            .expect("Invalid volatility"),
        };
        let maturity_date = NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d").expect("Invalid date format");
        let base_option = EquityOptionBase {
            symbol:data.base.symbol.clone(),
            currency: data.base.currency.clone(),
            exchange:data.base.exchange.clone(),
            name: data.base.name.clone(),
            cusip: data.base.cusip.clone(),
            isin: data.base.isin.clone(),
            settlement_type: data.base.settlement_type.clone(),

            underlying_price: Quote::new(data.base.underlying_price),
            current_price: Quote::new(data.current_price.unwrap_or(0.0)),
            strike_price: data.strike_price,
            vol_surface,
            maturity_date,
            discount_curve,
            entry_price: data.entry_price.unwrap_or(0.0),
            long_short: LongShort::LONG,
            dividend_yield: data.dividend.unwrap_or(0.0),
            valuation_date,
            multiplier: data.multiplier.unwrap_or(1.0),
        };
        let payoff_type = &data.payoff_type.parse::<PayoffType>().unwrap();
        let side: PutOrCall;
        let put_or_call = data.put_or_call.clone();
        match put_or_call.trim() {
            "C" | "c" | "Call" | "call" => side = PutOrCall::Call,
            "P" | "p" | "Put" | "put" => side = PutOrCall::Put,
            _ => panic!("Invalid side argument! Side has to be either 'C' or 'P'."),
        }

        let style = match data.exercise_style.as_ref().unwrap_or(&"European".to_string()).trim() {
            "European" | "european" => {
                ContractStyle::European
            }
            "American" | "american" => {
                ContractStyle::American
            }
            _ => {
                ContractStyle::European
            }
        };

        let payoff:Box<dyn Payoff> = match &payoff_type {
            PayoffType::Vanilla => Box::new(VanillaPayoff{
                put_or_call:side,
                exercise_style:style}),
            PayoffType::Binary => Box::new(BinaryPayoff{
                put_or_call:side,
                exercise_style:style}),
            _ => panic!("Payoff type not implemented yet (supported: vanilla, binary)"),
        };

        let equityoption = EquityOption {
            base: base_option,
            payoff,
            engine: match data.pricer.as_ref().map_or("Analytical",|v| v).trim() {
                "Analytical" | "analytical" | "bs" => Engine::BlackScholes,
                "MonteCarlo" | "montecarlo" | "MC" | "mc" => Engine::MonteCarlo,
                "Binomial" | "binomial" | "bino" => Engine::Binomial,
                "FiniteDifference" | "finitdifference" | "FD" | "fd" => Engine::FiniteDifference,
                _ => {
                    panic!("Invalid pricer");
                }
            },
            mc: montecarlo::MonteCarloConfig::from_data(data)
        };
        Box::new(equityoption)
    }
}

impl EquityOptionBase {
    pub fn time_to_maturity(&self) -> f64{
        let time_to_maturity = (self.maturity_date - self.valuation_date).num_days() as f64/365.0;
        time_to_maturity
    }
    /// Discount factor from the valuation date to maturity, off the curve.
    pub fn maturity_discount_factor(&self) -> f64 {
        self.discount_curve.df(self.time_to_maturity())
    }
    /// Continuously compounded zero rate to maturity implied by the curve.
    /// This is the `r` that enters d1/d2; it is consistent with
    /// [`maturity_discount_factor`](Self::maturity_discount_factor) by construction.
    pub fn risk_free_rate(&self) -> f64 {
        self.discount_curve
            .zero_rate_with(self.time_to_maturity(), Compounding::Continuous)
    }
    /// Forward price of the underlying at maturity implied by the discount
    /// curve and dividend yield: `S * exp((r - q) * T)`.
    pub fn forward_price(&self) -> f64 {
        let t = self.time_to_maturity();
        self.underlying_price.value() * ((self.risk_free_rate() - self.dividend_yield) * t).exp()
    }
    /// Black volatility for this option's strike and expiry, read off the
    /// surface (a flat surface returns its single vol).
    pub fn volatility(&self) -> f64 {
        self.vol_surface
            .vol(self.strike_price, self.forward_price(), self.time_to_maturity())
    }
    pub fn d1(&self) -> f64 {
        //Black-Scholes-Merton d1 function Parameters
        let volatility = self.volatility();
        let d1_numerator = (self.underlying_price.value() / self.strike_price).ln()
            + (self.risk_free_rate() - self.dividend_yield + 0.5 * volatility.powi(2))
            * self.time_to_maturity();

        let d1_denominator = volatility * (self.time_to_maturity().sqrt());
        return d1_numerator / d1_denominator;
    }
    pub fn d2(&self) -> f64 {
        let d2 = self.d1() - self.volatility() * self.time_to_maturity().powf(0.5);
        return d2;
    }
}
impl EquityOption {
    pub fn get_premium_at_risk(&self) -> f64 {
        let value = self.npv();
        let mut pay_off = self.payoff.payoff_amount(&self.base);
        if pay_off > 0.0 {
            return value - pay_off;
        } else {
            return value;
        }
    }
    
    /// Implied Black-Scholes volatility for `option_price` (safeguarded
    /// Newton with arbitrage-bound checks); does not modify the option.
    pub fn try_imp_vol(&self, option_price: f64) -> Result<f64, String> {
        blackscholes::implied_vol_from_price(
            self.base.underlying_price.value(),
            self.base.strike_price,
            self.base.risk_free_rate(),
            self.base.dividend_yield,
            self.time_to_maturity(),
            option_price,
            *self.payoff.put_or_call(),
        )
    }
    /// Implied vol for `option_price`; leaves the option holding a flat
    /// surface at the solved vol. Panics on arbitrage-violating prices —
    /// use [`try_imp_vol`](Self::try_imp_vol) to handle those gracefully.
    pub fn imp_vol(&mut self,option_price:f64) -> f64 {
        let vol = self.try_imp_vol(option_price).expect("implied vol solve failed");
        self.set_flat_vol(vol.max(1e-8));
        vol
    }
    pub fn get_imp_vol(&mut self) -> f64 {
        let target = self.base.current_price.value;
        self.imp_vol(target)
    }
    fn set_flat_vol(&mut self, vol: f64) {
        self.base.vol_surface = VolSurface::flat(
            vol,
            self.base.vol_surface.reference_date(),
            self.base.vol_surface.day_count(),
        )
        .expect("vol must be positive");
    }
}


impl Instrument for EquityOption  {
    fn npv(&self) -> f64 {
        let american = matches!(self.payoff.exercise_style(), ContractStyle::American);
        match self.engine {
            Engine::BlackScholes => {
                if american {
                    panic!(
                        "Analytical engine cannot price American exercise; \
                         use Binomial, FiniteDifference or MonteCarlo"
                    );
                }
                BlackScholesPricer::new().npv(&self)
            }
            Engine::MonteCarlo => montecarlo::npv(&self),
            Engine::Binomial => binomial::npv(&self),
            Engine::FiniteDifference => finite_difference::npv(&self),
        }
    }
}

/// Greeks: the Monte Carlo engine computes its own bump-and-reprice Greeks
/// with common random numbers (and so supports American exercise via
/// Longstaff-Schwartz repricing); the other engines use the analytic
/// Black-Scholes closed forms, which are the correct European
/// sensitivities regardless of which engine produced the NPV.
impl EquityOption {
    pub fn delta(&self) -> f64 {
        match self.engine {
            Engine::MonteCarlo => montecarlo::delta(&self),
            _ => BlackScholesPricer::new().delta(&self),
        }
    }
    pub fn gamma(&self) -> f64 {
        match self.engine {
            Engine::MonteCarlo => montecarlo::gamma(&self),
            _ => BlackScholesPricer::new().gamma(&self),
        }
    }
    pub fn vega(&self) -> f64 {
        match self.engine {
            Engine::MonteCarlo => montecarlo::vega(&self),
            _ => BlackScholesPricer::new().vega(&self),
        }
    }
    pub fn theta(&self) -> f64 {
        match self.engine {
            Engine::MonteCarlo => montecarlo::theta(&self),
            _ => BlackScholesPricer::new().theta(&self),
        }
    }
    pub fn rho(&self) -> f64 {
        match self.engine {
            Engine::MonteCarlo => montecarlo::rho(&self),
            _ => BlackScholesPricer::new().rho(&self),
        }
    }
}
// #[cfg(test)]
// mod tests {
//     //write a unit test for from_json
//     use super::*;
//     use crate::core::utils::{Contract,MarketData};
//     use crate::core::trade::OptionType;
//     use crate::core::trade::Transection;
//     use crate::core::utils::ContractStyle;
//     use crate::core::termstructure::YieldTermStructure;
//     use crate::core::quotes::Quote;
//     use chrono::{Datelike, Local, NaiveDate};
//     #[test]
//     fn test_from_json() {
//         let data = Contract {
//             action: "PV".to_string(),
//             market_data: Some(MarketData {
//                 underlying_price: 100.0,
//                 strike_price: 100.0,
//                 volatility: None,
//                 option_price: Some(10.0),
//                 risk_free_rate: Some(0.05),
//                 dividend: Some(0.0),
//                 maturity: "2024-01-01".to_string(),
//                 option_type: "C".to_string(),
//                 simulation: None
//             }),
//             pricer: "Analytical".to_string(),
//             asset: "".to_string(),
//             style: Some("European".to_string()),
//             rate_data: None
//         };
//         let option = EquityOption::from_json(&data);
//         assert_eq!(option.option_type, OptionType::Call);
//         assert_eq!(option.transection, Transection::Buy);
//         assert_eq!(option.underlying_price.value, 100.0);
//         assert_eq!(option.strike_price, 100.0);
//         assert_eq!(option.current_price.value, 10.0);
//         assert_eq!(option.dividend_yield, 0.0);
//         assert_eq!(option.volatility, 0.2);
//         assert_eq!(option.maturity_date, NaiveDate::from_ymd(2024, 1, 1));
//         assert_eq!(option.valuation_date, Local::today().naive_utc());
//         assert_eq!(option.engine, Engine::BlackScholes);
//         assert_eq!(option.style, ContractStyle::European);
//     }
// }

