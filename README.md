# Gulden_2013
Python implementation of the agent based model of international trade introduced in Gulden (2013).

| Paraneter                           | Description                                                                                  |
| ----------------------------------- | -------------------------------------------------------------------------------------------- |
|  Initialisations                    |                                                                                              |
| citizen counts                      | Population (static)                                                                          |
| countries                           | list of countries                                                                            |
| Industries                          | list of industries                                                                           |
| A                                   | a dictionary holding arrays of TFP values for every industry in a country                    |
| P                                   | a dictionary holding arrays of initial prices for every industry in a country                |
| alpha                               | a dictionary holding arrays of output elasticity of labour for every industrty in a country  |
| beta                                | a dictionary holding arrays of output elasticity of capital for every industrty in a country |     
| weights                             | Share parameter for each industry                                                            |
| elasticities                        | vector of substitution parameters for each industry                                          |
| sigma                               | elastitcity of substitution (constant)                                                       |
|                                     |                                                                                              |
| Model Parameters                    |                                                                                              |
| p_choice                            | Probability that an agent will reasses job and investment choices                            |
| price adjustment rate               | additive change factor of price                                                              |
|  target_trade_change                | step size of trade increase between trading countries                                        |
|                                     |                                                                                              |
| Model Functions                     |                                                                                              |
| Utility Function                    | geometric mean, CES utility functions                                                        |
| Demand Function                     | Hyperbolic Demand                                                                            |
| Production Function                 | Cobb Douglas Production                                                                      |
| Wage Function                       | Marginal returns of labour                                                                   |
| Price Function                      | Marginal rate of substitution                                                                |
