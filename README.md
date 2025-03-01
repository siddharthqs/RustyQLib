[![Build and Tests](https://github.com/siddharthqs/RustyQLib/actions/workflows/rust.yml/badge.svg)](https://github.com/siddharthqs/RustyQLib/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Crates.io](https://img.shields.io/crates/dr/rustyqlib)
![Crates.io](https://img.shields.io/crates/v/rustyqlib)
[![codecov](https://codecov.io/gh/siddharthqs/RustyQLib/graph/badge.svg?token=879K6LTTR4)](https://codecov.io/gh/siddharthqs/RustyQLib)
# RUSTYQLib :Pricing Options with Confidence using JSON
RustyQLib is a lightweight yet robust quantitative finance library designed for pricing options.
Built entirely in Rust, it offers a unique combination of safety, performance, and expressiveness that is crucial
for handling financial data and complex calculations. RustyQlib simplifies option pricing without compromising
on safety, speed, or usability. It uses JSON to make distributed computing easier and integration with other systems or your websites.
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

### JSON Simplicity:

- Ease of Use: Providing input data in JSON format is straightforward and human-readable.
- Portability: JSON is a platform-independent format, so you can use it on any operating system.
- Flexibility: JSON accommodates various data types and structures, enabling you to define not only the option details but also additional market data, historical information, and risk parameters as needed.
- Integration-Ready: You can seamlessly connect it to data sources, trading platforms, or other financial systems, simplifying your workflow and enhancing automation.

### Stypes:
- [x] European
- [x] American
- [ ] Bermudan
- [ ] Asian

### Instruments:
#### Equity
- [x] Equity Forward
- [x] Equity Future
- [x] Equity Option
- [ ] Equity Forward Start Option
- [ ] Equity Basket
- [ ] Equity Barrier
- [ ] Equity Lookback
- [ ] Equity Asian
- [ ] Equity Rainbow
- [ ] Equity Chooser
#### Interest Rate
- [x] Deposit
- [ ] FRA
- [ ] Interest Rate Swap
#### Commodities
- [x] Commodity Option
- [ ] Commodity Forward Start Option
- [ ] Commodity Barrier
- [ ] Commodity Lookback

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


