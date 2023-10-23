pub struct Quote{
    pub value: f64,
    pub bid: f64,
    pub ask: f64,
    pub mid: f64,
}
impl Quote{
    pub fn new(value: f64) -> Self {
        Quote{value: value, bid: value, ask: value, mid: value }
    }
    pub fn value(&self) -> f64 { self.value }
    pub fn valid_value(&self) -> bool { self.value>0.0}
}