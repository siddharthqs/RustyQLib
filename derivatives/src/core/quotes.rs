pub struct Quote{
    pub value: f64
}
impl Quote{
    pub fn new(value: f64) -> Self {
        Quote{value: value}
    }
    pub fn value(&self) -> f64 { self.value }
    pub fn valid_value(&self) -> bool { self.value>0.0}
}