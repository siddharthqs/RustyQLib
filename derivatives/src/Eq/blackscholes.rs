use std::f64::consts::{PI, SQRT_2};
use libm::{log, exp};
use Eq::utils::{N,dN};
use Eq::vanila_option::{EquityOption,OptionType};

impl EquityOption{
    pub fn pv(&self)->f64 {
        asset!(self.volatility>=0);
        asset!(self.time_to_maturity>=0);
        asset!(self.current_price>=0);
        if self.option_type == OptionType::Call {
            let option_price = self.current_price * N(self.d1()) * exp(-self.dividend_yield * self.time_to_maturity)
                - self.strike_price * exp(-self.risk_free_rate * self.time_to_maturity) * N(self.d2());
            return option_price
        }
        else {
            let option_price = -self.current_price * N(-self.d1()) * exp(-self.dividend_yield * self.time_to_maturity)
                + self.strike_price * exp(-self.risk_free_rate * self.time_to_maturity) * N(-self.d2());
            return option_price
        }

    }
    pub fn get_premium_at_risk(&self)->f64{
        let value = self.pv();
        let mut pay_off = 0.0;
        if self.option_type==OptionType::Call {
            pay_off = self.current_price - self.strike_price;
        }
        else if self.option_type==OptionType::Put {
             pay_off = self.strike_price -self.current_price;
        }
        if pay_off>0.0{
            return value -pay_off
        }
        else{
            return value
        }
    }
    pub fn delta(&self) -> f64 {
        let mut delta = N(self.d1());
        if self.option_type==OptionType::Call {
            delta = delta*exp(-self.dividend_yield * self.time_to_maturity);
        }
        else if self.option_type==OptionType::Put {
             delta = delta -1;
        }
        return delta;
    }
    pub fn gamma(&self) -> f64 {
        let mut gamma = dN(self.d1());
        //(St * sigma * math.sqrt(T - t))
        let var_sqrt = self.volatility*(self.time_to_maturity.sqrt());
        gamma = gamma/(self.current_price*var_sqrt);
        return gamma;
    }
    //ToDO
    //theta
    //rho
    pub fn d1(&self) -> f64 {
        //Black-Scholes-Merton d1 function Parameters
        let tmp1 = (self.current_price/self.strike_price).ln()+
            (self.risk_free_rate - self.dividend_yield+0.5*self.volatility.powi(2))*
                self.time_to_maturity;

        let tmp2 = self.volatility*(self.time_to_maturity.sqrt());
        return tmp1/tmp2;
    }
    pub fn d2(&self)->f64{
        let d2 = self.d1() - self.volatility* self.time_to_maturity.powf(0.5);
        return d2;
    }

}
