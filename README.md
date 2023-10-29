[![Build and Tests](https://github.com/siddharthqs/RustyQLib/actions/workflows/rust.yml/badge.svg)](https://github.com/siddharthqs/RustyQLib/actions/workflows/rust.yml)

# RUSTYQLib :Pricing Options with Confidence using JSON
RustyQLib is a lightweight yet robust quantitative finance library designed for pricing options.
Built entirely in Rust, it offers a unique combination of safety, performance, and expressiveness that is crucial
for handling financial data and complex calculations. RustyQlib simplifies equity option pricing without compromising
on safety, speed, or usability.
## License
RustyQlib is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See LICENSE-APACHE and LICENSE-MIT for details.
## Running
After cloning the repository and building you can run the following command:
```bash
rustyqlib file --input <FILE> --output <FILE>
````
and for pricing all contracts in a directory
```bash
rustyqlib dir --input <DIR> --output <DIR>
```
and for interactive mode
```bash
rustyqlib interactive
```
and for build mode to build vol surface or interest rate curve
```bash
rustyqlib build --input <FILE> --output <DIR>
```
Sample input file is provided in the repository (src\input\equity_option.json)
Files are in JSON format and can be easily edited with any text editor.
## Features

### JSON Input Simplicity:

- Ease of Use: Providing input data in JSON format is straightforward and human-readable.
   You can specify the parameters of your options with ease, making complex financial modeling accessible to all.
- Flexibility: JSON accommodates various data types and structures, enabling you to define not only the option details but also additional market data, historical information, and risk parameters as needed.
- Integration-Ready: RustQuant's JSON input is integration-friendly. You can seamlessly connect it to data sources, trading platforms, or other financial systems, simplifying your workflow and enhancing automation.

### JSON Output Clarity:
JSON Output Clarity
- Structured Results: RustQuant produces JSON output, that is your provided input with pricing results, Greeks, and risk profiles.

- Scalability: JSON output is highly scalable.
  You can process large batches of option pricing requests and obtain results in a structured format, streamlining portfolio management.
- Interoperability: JSON output integrates seamlessly with data visualization tools, databases, and reporting systems, enabling you to present and share your derivative pricing results effectively.
### Stypes:
- [x] European
- [x] American
- [ ] Bermudan
- [ ] Asian

### Instruments:
- [x] Equity Option
- [ ] Equity Forward Start Option
- [ ] Equity Basket
- [ ] Equity Barrier
- [ ] Equity Lookback
- [ ] Equity Asian
- [ ] Equity Rainbow
- [ ] Equity Chooser


### Pricing engines:
- [x] Black Scholes
- [x] Binomial Tree
- [x] Monte Carlo
- [ ] Finite Difference
- [ ] Longstaff-Schwartz
- [ ] Heston
- [ ] Local Volatility
- [ ] Stochastic Volatility
- [ ] Jump Diffusion


