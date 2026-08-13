#[derive(PartialEq, Debug, Clone)]
pub enum Transection {
    Buy,
    Sell,
}

/// Option side. Serializes as the contract-file convention `"C"` / `"P"`;
/// deserialization also accepts the spelled-out forms.
#[derive(PartialEq, Eq, Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum PutOrCall {
    #[serde(
        rename = "C",
        alias = "c",
        alias = "call",
        alias = "Call",
        alias = "CALL"
    )]
    Call,
    #[serde(rename = "P", alias = "p", alias = "put", alias = "Put", alias = "PUT")]
    Put,
}
