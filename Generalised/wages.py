from typing import Tuple, List



def wageAsMarginalProductROIAsResidual(country, ##cannot explicitly call the nation because of cyclical references
                                       industry_id: int,
                                       industries: List[str],
                                       production_function) -> Tuple[float,float]:
    i = industry_id
    inc_labor = country.labor[industries[i]] + 1
    inc_production = production_function(country.A[i], country.alpha[industries[i]],
                                         inc_labor,
                                         country.beta[industries[i]],
                                         country.capital[industries[i]])

    wage: float = country.prices[industries[i]] * (inc_production - country.production[industries[i]])
    # self.wage[industries[i]] = (self.prices[industries[i]] * self.production[industries[i]])/self.labor[industries[i]]
    expected_wage_bill: float = wage * country.labor[industries[i]]

    roi: float = ((country.prices[industries[i]] * country.production[industries[i]]) - expected_wage_bill) / \
                 country.capital[industries[i]]
    
    return wage,roi


def wageAsShareOfProduct(country,
                                       industry_id: int,
                                       industries: List[str],
                                       production_function,
                         share: float = 0.25) -> Tuple[float, float]:
    i = industry_id
    inc_labor = country.labor[industries[i]] + 1
    inc_production = production_function(country.A[i], country.alpha[industries[i]],
                                         inc_labor,
                                         country.beta[industries[i]],
                                         country.capital[industries[i]])

    wage: float = share * (country.prices[industries[i]] * country.production[industries[i]])/country.labor[industries[i]]
    expected_wage_bill: float = country.wage[industries[i]] * country.labor[industries[i]]

    roi: float = (1-share) * (country.prices[industries[i]] * country.production[industries[i]]) / \
                 country.capital[industries[i]]

    return wage, roi


def wagesAsShareOfMarginalProduct(country, ##cannot explicitly call the nation because of cyclical references
                                       industry_id: int,
                                       industries: List[str],
                                       production_function,
                         share: float = 0.25) -> Tuple[float, float]:
    i = industry_id
    inc_labor = country.labor[industries[i]] + 1
    inc_production = production_function(country.A[i], country.alpha[industries[i]],
                                         inc_labor,
                                         country.beta[industries[i]],
                                         country.capital[industries[i]])

    wage: float = share* country.prices[industries[i]] * (inc_production - country.production[industries[i]])
    # self.wage[industries[i]] = (self.prices[industries[i]] * self.production[industries[i]])/self.labor[industries[i]]
    expected_wage_bill: float = wage * country.labor[industries[i]]

    roi: float = ((country.prices[industries[i]] * country.production[industries[i]]) - expected_wage_bill) / \
                 country.capital[industries[i]]

    return wage,roi