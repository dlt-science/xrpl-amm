#  conftest.py
import pytest
from amm.env import AMMi, User


@pytest.fixture()
def asset1() -> str:
    return 'UCL'


@pytest.fixture()
def asset2() -> str:
    return 'CBT'


@pytest.fixture()
def users(asset1, asset2) -> dict:
    return {'alice': User('alice', {'XRP': 100, asset1: 120, asset2: 100}),
            'bob': User('bob', {'XRP': 324, asset1: 243, asset2: 132}),
            'firas': User('firas', {'XRP': 400, asset1: 192, asset2: 453}),
            'jiahua': User('jiahua', {'XRP': 232, asset1: 78, asset2: 121}),
            'david': User('david', {'XRP': 856, asset1: 89, asset2: 100}),
            'aanchal': User('aanchal', {'XRP': 34, asset1: 356, asset2: 99}),
            'yebo': User('yebo', {'XRP': 432, asset1: 112, asset2: 359}),
            'zhang': User('zhang', {'XRP': 121, asset1: 354, asset2: 67}),
            'brad': User('brad', {'XRP': 175, asset1: 111, asset2: 111}),
            'arthur': User('arthur', {'XRP': 78, asset1: 59, asset2: 159})}


@pytest.fixture()
def alice(users: dict) -> User:
    return users['alice']


@pytest.fixture()
def amm(users: dict, asset1: str, asset2: str) -> AMMi:
    amm = users['alice'].createAMM(ammID=1, asset1=asset1,
                                   asset2=asset2, amount1=40, amount2=30)
    return amm


@pytest.fixture()
# balances of alice and amm after alice created the amm
def prev_bal(asset1: str, asset2: str) -> dict:
    prev_amm_bal = {asset1: 40, asset2: 30, 'LPTokens': 34.64101615137755}
    conservation_func = (
        prev_amm_bal[asset1]**0.5) * (prev_amm_bal[asset2]**0.5)
    prev_alice_bal = {'XRP': 100, asset1: 80,
                      asset2: 70, 'LPTokens': 34.64101615137755}
    return {'amm': prev_amm_bal, 'alice': prev_alice_bal, 'conservation_func': conservation_func}
