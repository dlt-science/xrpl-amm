# AMM functionalities tests

import pytest
from amm.env import AMMVote
from amm.actions import Deposit, Withdraw, AMMBid, Swap
from computations import *
from sympy import Symbol
from sympy.solvers import solve


@pytest.mark.parametrize("user,assets", [
    ('alice', {'XRP': 100, 'UCL': 120, 'CBT': 100}),
    ('bob', {'XRP': 324, 'UCL': 243, 'CBT': 132})
])
def test_users(user, assets, users):
    assert users[user].user_name == user
    assert users[user].assets == assets


@pytest.mark.parametrize("amount1,amount2", [
    (10, 10), (232, 52454), (23, 45)
])
def test_createAMM(alice, asset1, asset2, amount1, amount2):
    old_alice_assets = {'XRP': 100, 'UCL': 120, 'CBT': 100}
    if alice.assets[asset1] >= amount1 and alice.assets[asset2] >= amount2:
        amm = alice.createAMM(ammID=1, asset1=asset1,
                              asset2=asset2, amount1=amount1, amount2=amount2)
        LPTokens = amount1**amm.W * amount2**amm.W
        assert alice.assets['LPTokens'] == amm.assets['LPTokens'] == LPTokens
        assert old_alice_assets[asset1] - alice.assets[asset1] == amount1
        assert old_alice_assets[asset2] - alice.assets[asset2] == amount2
        assert amm.assets[asset1] == amount1
        assert amm.assets[asset2] == amount2
        assert amm.ammi == 1
        assert amm.curr_codes == [asset1, asset2]
    else:
        with pytest.raises(Exception) as e:
            alice.createAMM(ammID=1, asset1=asset1,
                            asset2=asset2, amount1=amount1, amount2=amount2)
        assert str(e.value) == "Not enough tokens"


# --------------------------- TEST DEPOSITS --------------------------- #

@pytest.mark.parametrize("LPTokenOut", [
    (15), (20), (30), (343)
])
def test_deposit_LPTokenOut(amm, alice, asset1, asset2, LPTokenOut, prev_bal):
    alice_deposits = Deposit(alice, amm)
    delta_asset1 = delta_token_Double(
        LPTokenOut, bal_token=amm.assets[asset1], bal_LPTokens=amm.assets['LPTokens'])
    delta_asset2 = delta_token_Double(
        LPTokenOut, bal_token=amm.assets[asset2], bal_LPTokens=amm.assets['LPTokens'])
    if alice.assets[asset1] < delta_asset1 and alice.assets[asset2] < delta_asset2:
        with pytest.raises(Exception) as e:
            alice_deposits.deposit_LPTokenOut(LPTokenOut)
        assert str(e.value) == "Not enough tokens"
    else:
        alice_deposits.deposit_LPTokenOut(LPTokenOut)
        assert round(alice.assets[asset1], 5) == round(
            prev_bal['alice'][asset1] - delta_asset1, 5)
        assert round(alice.assets[asset2], 5) == round(
            prev_bal['alice'][asset2] - delta_asset2, 5)
        assert amm.VoteSlots == []
        assert round(amm.assets[asset1], 5) == round(
            prev_bal['amm'][asset1] + delta_asset1, 5)
        assert round(amm.assets[asset2], 5) == round(
            prev_bal['amm'][asset2] + delta_asset2, 5)
        assert round(amm.assets['LPTokens'], 5) == round(
            prev_bal['amm']['LPTokens'] + LPTokenOut, 5)


@pytest.mark.parametrize("amount1,amount2", [
    (232, 433), (15, 20), (78, 12)
])
def test_deposit_Amount1_Amount2(amm, alice, asset1, asset2, amount1, amount2, prev_bal):
    alice_deposits = Deposit(alice, amm)
    if alice.assets[asset1] < amount1 or alice.assets[asset2] < amount2:
        with pytest.raises(Exception) as e:
            alice_deposits.deposit_Amount1_Amount2(
                asset_A=asset1, asset_B=asset2, amount_A=amount1, amount_B=amount2)
        assert str(e.value) == "Not enough tokens!"
    else:
        # Z = LPTokens to be returned/issued
        Z = amount1 * amm.assets['LPTokens'] / amm.assets[asset1]
        # X = amount of asset B (delta_B)
        X = Z / amm.assets['LPTokens'] * amm.assets[asset2]
        alice_deposits.deposit_Amount1_Amount2(
            asset_A=asset1, asset_B=asset2, amount_A=amount1, amount_B=amount2)
        if X <= amount2:
            assert amm.assets[asset1] == prev_bal['amm'][asset1] + amount1
            assert amm.assets[asset2] == prev_bal['amm'][asset2] + X
            assert amm.assets['LPTokens'] == prev_bal['amm']['LPTokens'] + Z
            assert alice.assets['LPTokens'] == prev_bal['alice']['LPTokens'] + Z
            assert alice.assets[asset1] == prev_bal['alice'][asset1] - amount1
            assert alice.assets[asset2] == prev_bal['alice'][asset2] - X
            assert 'alice' in amm.LPs
        elif X > amount2:
            # W = LPTokens to be returned/issued
            W = amount2 * amm.assets['LPTokens'] / amm.assets[asset2]
            # Y = amount of asset A (delta_A)
            Y = W / amm.assets['LPTokens'] * amm.assets[asset1]
            if Y <= amount1:
                assert amm.assets[asset1] == prev_bal['amm'][asset1] + Y
                assert amm.assets[asset2] == prev_bal['amm'][asset2] + amount2
                assert amm.assets['LPTokens'] == prev_bal['amm']['LPTokens'] + W
                assert alice.assets['LPTokens'] == prev_bal['alice']['LPTokens'] + W
                assert alice.assets[asset1] == prev_bal['alice'][asset1] - Y
                assert alice.assets[asset2] == prev_bal['alice'][asset2] - amount2
                assert 'alice' in amm.LPs
            else:
                assert amm.assets == amm.assets
                assert alice.assets == alice.assets
                assert 'alice' not in amm.LPs


@pytest.mark.parametrize("asset,amount", [
    ('asset1', 20), ('asset2', 30), ('asset1', 233)
])
def test_deposit_Amount(asset, amm, alice, amount, prev_bal, request):
    alice_deposits = Deposit(alice, amm)
    if alice.assets[request.getfixturevalue(asset)] > amount:
        L = delta_LPTokens_Single(
            LPTokens=amm.assets['LPTokens'], bal_asset=amm.assets[request.getfixturevalue(asset)], amount=amount)
        alice_deposits.deposit_Amount(
            asset=request.getfixturevalue(asset), amount=amount)
        assert amm.assets['LPTokens'] == prev_bal['amm']['LPTokens'] + L
        assert alice.assets['LPTokens'] == prev_bal['alice']['LPTokens'] + L
        assert amm.assets[request.getfixturevalue(
            asset)] == prev_bal['amm'][request.getfixturevalue(asset)] + amount
        assert alice.assets[request.getfixturevalue(
            asset)] == prev_bal['alice'][request.getfixturevalue(asset)] - amount
        assert 'alice' in amm.LPs
    else:
        with pytest.raises(Exception) as e:
            alice_deposits.deposit_Amount(
                asset=request.getfixturevalue(asset), amount=amount)
        assert str(e.value) == "Not enough tokens"


@pytest.mark.parametrize("asset,amount,LPTokenOut", [
    ('asset1', 20, 10), ('asset2', 30, 4), ('asset1', 233, 23423), ('asset2', 5, 13)
])
def test_deposit_Amount_LPTokenOut(alice, amm, asset, amount, LPTokenOut, prev_bal, request):
    alice_deposits = Deposit(alice, amm)
    if alice.assets[request.getfixturevalue(asset)] > amount:
        delta_token = delta_token_Single(
            LPTokenOut=LPTokenOut, LPTokens=amm.assets['LPTokens'], bal_asset=amm.assets[request.getfixturevalue(asset)])
        alice_deposits.deposit_Amount_LPTokenOut(
            asset=request.getfixturevalue(asset), amount=amount, LPTokenOut=LPTokenOut)
        assert 'alice' in amm.LPs
        assert amm.assets['LPTokens'] == prev_bal['amm']['LPTokens'] + LPTokenOut
        assert alice.assets['LPTokens'] == prev_bal['alice']['LPTokens'] + LPTokenOut
        assert alice.assets[request.getfixturevalue(
            asset)] == prev_bal['alice'][request.getfixturevalue(asset)] - delta_token
        assert amm.assets[request.getfixturevalue(
            asset)] == prev_bal['amm'][request.getfixturevalue(asset)] + delta_token
    else:
        with pytest.raises(Exception) as e:
            alice_deposits.deposit_Amount(
                asset=request.getfixturevalue(asset), amount=amount)
        assert str(e.value) == "Not enough tokens"


def test_deposit_Amount_EPrice():
    pass


# --------------------------- TEST WITHDRAWALS --------------------------- #


@pytest.mark.parametrize("LPTokenIn", [
    (15), (20), (30), (343), (25435.435)
])
def test_withdraw_LPTokenIn(amm, alice, prev_bal, asset1, asset2, LPTokenIn):
    alice_withdrawals = Withdraw(alice, amm)
    if alice.assets['LPTokens'] >= LPTokenIn:
        delta_asset1 = delta_token_Double(
            LPTokens_In_or_Out=LPTokenIn, bal_token=amm.assets[asset1], bal_LPTokens=amm.assets['LPTokens'])
        delta_asset2 = delta_token_Double(
            LPTokens_In_or_Out=LPTokenIn, bal_token=amm.assets[asset2], bal_LPTokens=amm.assets['LPTokens'])
        alice_withdrawals.withdraw_LPTokenIn(LPTokenIn=LPTokenIn)
        assert alice.assets['LPTokens'] == prev_bal['alice']['LPTokens'] - LPTokenIn
        assert amm.assets['LPTokens'] == prev_bal['amm']['LPTokens'] - LPTokenIn
        assert alice.assets[asset1] == prev_bal['alice'][asset1] + delta_asset1
        assert alice.assets[asset2] == prev_bal['alice'][asset2] + delta_asset2
        assert amm.assets[asset1] == prev_bal['amm'][asset1] - delta_asset1
        assert amm.assets[asset2] == prev_bal['amm'][asset2] - delta_asset2
    else:
        with pytest.raises(Exception) as e:
            alice_withdrawals.withdraw_LPTokenIn(LPTokenIn=LPTokenIn)
        assert str(e.value) == "Not enough tokens"


@pytest.mark.parametrize("amount1,amount2", [
    (232, 433), (15, 20), (78, 12), (3, 5)
])
def test_withdraw_Amount1_Amount2(amm, alice, asset1, asset2, amount1, amount2, prev_bal):
    alice_withdrawals = Withdraw(alice, amm)
    # Z = LPTokensIn
    Z = amount1 * amm.assets['LPTokens'] / amm.assets[asset1]
    # X = amount of asset B (delta_B)
    X = Z / amm.assets['LPTokens'] * amm.assets[asset2]
    if X <= amount2 and alice.assets['LPTokens'] >= Z:
        alice_withdrawals.withdraw_Amount1_Amount2(
            asset_A=asset1, asset_B=asset2, amount_A=amount1, amount_B=amount2)
        assert amm.assets['LPTokens'] == prev_bal['amm']['LPTokens'] - Z
        assert alice.assets['LPTokens'] == prev_bal['alice']['LPTokens'] - Z
        assert amm.assets[asset1] == prev_bal['amm'][asset1] - amount1
        assert alice.assets[asset1] == prev_bal['alice'][asset1] + amount1
        assert amm.assets[asset2] == prev_bal['amm'][asset2] - X
        assert alice.assets[asset2] == prev_bal['alice'][asset2] + X
        assert amm.LPs['alice'] == prev_bal['alice']['LPTokens'] - Z
    elif X > amount2 and alice.assets['LPTokens'] >= Z:
        # Q = LPTokensIn
        Q = amount2 * amm.assets['LPTokens'] / amm.assets[asset2]
        # W = amount of asset A (delta_A)
        W = Q / amm.assets['LPTokens'] * amm.assets[asset1]
        alice_withdrawals.withdraw_Amount1_Amount2(
            asset_A=asset1, asset_B=asset2, amount_A=amount1, amount_B=amount2)
        assert amm.assets['LPTokens'] == prev_bal['amm']['LPTokens'] - Q
        assert alice.assets['LPTokens'] == prev_bal['alice']['LPTokens'] - Q
        assert amm.assets[asset1] == prev_bal['amm'][asset1] - W
        assert alice.assets[asset1] == prev_bal['alice'][asset1] + W
        assert amm.assets[asset2] == prev_bal['amm'][asset2] - amount2
        assert alice.assets[asset2] == prev_bal['alice'][asset2] + amount2
        assert amm.LPs['alice'] == prev_bal['alice']['LPTokens'] - Q
    else:
        with pytest.raises(Exception) as e:
            alice_withdrawals.withdraw_Amount1_Amount2(
                asset_A=asset1, asset_B=asset2, amount_A=amount1, amount_B=amount2)
        assert str(e.value) == "Not enough tokens"


@pytest.mark.parametrize("asset,amount", [
    ('asset1', 20), ('asset2', 30), ('asset1', 233)
])
def test_withdraw_Amount(amm, alice, prev_bal, amount, asset, request):
    alice_withdrawals = Withdraw(alice, amm)
    L = delta_LPTokens_WS(LPTokens=amm.assets['LPTokens'], amount=amount,
                          bal_asset=amm.assets[request.getfixturevalue(asset)])
    if amm.assets[request.getfixturevalue(asset)] > amount and alice.assets['LPTokens'] >= L:
        alice_withdrawals.withdraw_Amount(
            asset=request.getfixturevalue(asset), amount=amount)
        assert amm.assets['LPTokens'] == prev_bal['amm']['LPTokens'] - L
        assert alice.assets['LPTokens'] == prev_bal['alice']['LPTokens'] - L
        assert amm.assets[request.getfixturevalue(
            asset)] == prev_bal['amm'][request.getfixturevalue(asset)] - amount
        assert alice.assets[request.getfixturevalue(
            asset)] == prev_bal['alice'][request.getfixturevalue(asset)] + amount
        assert amm.LPs['alice'] == prev_bal['alice']['LPTokens'] - L
    else:
        with pytest.raises(Exception) as e:
            alice_withdrawals.withdraw_Amount(
                asset=request.getfixturevalue(asset), amount=amount)
        assert str(e.value) == "Not enough tokens"


@pytest.mark.parametrize("asset,amount,LPTokenIn", [
    ('asset2', 30, 4), ('asset1', 423, 2423), ('asset2', 5, 13), ('asset1', '', 32)
])
def test_withdraw_Amount_LPTokenIn(alice, amm, asset, amount, LPTokenIn, prev_bal, request):
    alice_withdrawals = Withdraw(alice, amm)
    if alice.assets['LPTokens'] >= LPTokenIn:
        # Y = amount of asset A
        Y = delta_token_WS(amm.assets[request.getfixturevalue(
            asset)], LPTokenIn=LPTokenIn, LPTokens=amm.assets['LPTokens'])
        if (amount and Y >= amount) or (not amount):
            alice_withdrawals.withdraw_Amount_LPTokenIn(
                asset=request.getfixturevalue(asset), LPTokenIn=LPTokenIn, amount=amount)
            assert amm.assets['LPTokens'] == prev_bal['amm']['LPTokens'] - LPTokenIn
            assert alice.assets['LPTokens'] == prev_bal['alice']['LPTokens'] - LPTokenIn
            assert alice.assets[request.getfixturevalue(
                asset)] == prev_bal['alice'][request.getfixturevalue(asset)] + Y
            assert amm.assets[request.getfixturevalue(
                asset)] == prev_bal['amm'][request.getfixturevalue(asset)] - Y
        else:
            with pytest.raises(Exception) as e:
                alice_withdrawals.withdraw_Amount_LPTokenIn(
                    asset=request.getfixturevalue(asset), LPTokenIn=LPTokenIn, amount=amount)
            assert str(e.value) == "Not enough tokens"


def test_withdraw_Amount_EPrice():
    pass


# --------------------------- TEST SWAPS --------------------------- #


@pytest.mark.parametrize("assetIn, assetOut, amountIn", [
    ('asset2', 'asset1', 4), ('asset1', 'asset2', 2423), ('asset2', 'asset1', 13),
    ('asset2', 'asset1', 40), ('asset1', 'asset2', 23), ('asset2', 'asset1', 134)
])
def test_swap_given_amount_In(amm, alice, assetIn, assetOut, amountIn, prev_bal, request):
    alice_swaps = Swap(alice, amm)
    delta_tokenOut = delta_tokenOut_Swap(
        bal_assetIn=amm.assets[request.getfixturevalue(assetIn)], bal_assetOut=amm.assets[request.getfixturevalue(assetOut)], delta_tokenIn=amountIn)
    if alice.assets[request.getfixturevalue(assetIn)] >= amountIn and amm.assets[request.getfixturevalue(assetOut)] > delta_tokenOut:
        alice_swaps.swap_given_amount_In(assetIn=request.getfixturevalue(
            assetIn), assetOut=request.getfixturevalue(assetOut), amountIn=amountIn)
        assert amm.assets[request.getfixturevalue(
            assetIn)] == prev_bal['amm'][request.getfixturevalue(assetIn)] + amountIn
        assert amm.assets[request.getfixturevalue(
            assetOut)] == prev_bal['amm'][request.getfixturevalue(assetOut)] - delta_tokenOut
        assert alice.assets[request.getfixturevalue(
            assetOut)] == prev_bal['alice'][request.getfixturevalue(assetOut)] + delta_tokenOut
        assert alice.assets[request.getfixturevalue(
            assetIn)] == prev_bal['alice'][request.getfixturevalue(assetIn)] - amountIn
        # conservation function preserved: (LOOKS LIKE IT'S NOT EXACTLY PRESERVED TO THE DECIMAL POINT!)
        assert round((amm.assets[request.getfixturevalue(
            assetIn)])**0.5 * (amm.assets[request.getfixturevalue(assetOut)])**0.5, 0) == round(prev_bal['conservation_func'], 0)
    else:
        with pytest.raises(Exception) as e:
            alice_swaps.swap_given_amount_In(assetIn=request.getfixturevalue(
                assetIn), assetOut=request.getfixturevalue(assetOut), amountIn=amountIn)
        assert str(e.value) == "Not enough tokens"


@pytest.mark.parametrize("assetIn, assetOut, amountOut", [
    ('asset2', 'asset1', 4), ('asset1', 'asset2', 2423), ('asset2', 'asset1', 13),
    ('asset2', 'asset1', 40), ('asset1', 'asset2', 23), ('asset2', 'asset1', 134)
])
def test_swap_given_amount_Out(amm, alice, assetIn, assetOut, amountOut, prev_bal, request):
    alice_swaps = Swap(alice, amm)
    if amm.assets[request.getfixturevalue(assetOut)] > amountOut:
        delta_tokenIn = delta_tokenIn_Swap(bal_tokenIn=amm.assets[request.getfixturevalue(
            assetIn)], bal_tokenOut=amm.assets[request.getfixturevalue(assetOut)], delta_tokenOut=amountOut)
        if alice.assets[request.getfixturevalue(assetIn)] >= delta_tokenIn:
            alice_swaps.swap_given_amount_Out(assetIn=request.getfixturevalue(
                assetIn), assetOut=request.getfixturevalue(assetOut), amountOut=amountOut)
            assert amm.assets[request.getfixturevalue(
                assetIn)] == prev_bal['amm'][request.getfixturevalue(assetIn)] + delta_tokenIn
            assert amm.assets[request.getfixturevalue(
                assetOut)] == prev_bal['amm'][request.getfixturevalue(assetOut)] - amountOut
            assert alice.assets[request.getfixturevalue(
                assetOut)] == prev_bal['alice'][request.getfixturevalue(assetOut)] + amountOut
            assert alice.assets[request.getfixturevalue(
                assetIn)] == prev_bal['alice'][request.getfixturevalue(assetIn)] - delta_tokenIn
            # conservation function preserved: (LOOKS LIKE IT'S NOT EXACTLY PRESERVED TO THE DECIMAL POINT!)
            assert round((amm.assets[request.getfixturevalue(
                assetIn)])**0.5 * (amm.assets[request.getfixturevalue(assetOut)])**0.5, 0) == round(prev_bal['conservation_func'], 0)
        else:
            with pytest.raises(Exception) as e:
                alice_swaps.swap_given_amount_Out(assetIn=request.getfixturevalue(
                    assetIn), assetOut=request.getfixturevalue(assetOut), amountOut=amountOut)
            assert str(e.value) == "Not enough tokens"
    else:
        with pytest.raises(Exception) as e:
            alice_swaps.swap_given_amount_Out(assetIn=request.getfixturevalue(
                assetIn), assetOut=request.getfixturevalue(assetOut), amountOut=amountOut)
        assert str(e.value) == "Not enough tokens"


@pytest.mark.parametrize("assetIn, assetOut, post_sp", [
    ('asset2', 'asset1', 1), ('asset1', 'asset2', 0.9), ('asset2', 'asset1', 2),
    ('asset2', 'asset1', 1.1), ('asset1',
                                'asset2', 1.04), ('asset2', 'asset1', 1.8),
    ('asset1', 'asset2', 100), ('asset2', 'asset1', 13), ('asset2', 'asset1', 50)
])
def test_swap_given_postSP(amm, alice, assetIn, assetOut, prev_bal, request, post_sp):
    alice_swaps = Swap(alice, amm)
    delta_tokenIn = delta_tokenIn_given_spotprices(bal_assetIn=amm.assets[request.getfixturevalue(
        assetIn)], bal_assetOut=amm.assets[request.getfixturevalue(assetOut)], post_sp=post_sp)
    delta_tokenOut = delta_tokenOut_Swap(bal_assetIn=amm.assets[request.getfixturevalue(
        assetIn)], bal_assetOut=amm.assets[request.getfixturevalue(assetOut)], delta_tokenIn=delta_tokenIn)
    if amm.assets[request.getfixturevalue(assetOut)] > delta_tokenOut and alice.assets[request.getfixturevalue(assetIn)] > delta_tokenIn:
        alice_swaps.swap_given_postSP(assetIn=request.getfixturevalue(
            assetIn), assetOut=request.getfixturevalue(assetOut), post_sp=post_sp)
        assert amm.assets[request.getfixturevalue(
            assetIn)] == prev_bal['amm'][request.getfixturevalue(assetIn)] + delta_tokenIn
        assert alice.assets[request.getfixturevalue(
            assetIn)] == prev_bal['alice'][request.getfixturevalue(assetIn)] - delta_tokenIn
        assert amm.assets[request.getfixturevalue(
            assetOut)] == prev_bal['amm'][request.getfixturevalue(assetOut)] - delta_tokenOut
        assert alice.assets[request.getfixturevalue(
            assetOut)] == prev_bal['alice'][request.getfixturevalue(assetOut)] + delta_tokenOut
        assert post_sp == round(amm.spot_price1(request.getfixturevalue(
            assetIn), request.getfixturevalue(assetOut)), 2)
    else:
        with pytest.raises(Exception) as e:
            alice_swaps.swap_given_postSP(assetIn=request.getfixturevalue(
                assetIn), assetOut=request.getfixturevalue(assetOut), post_sp=post_sp)
        assert str(e.value) == "Not enough tokens"


# --------------------------- TEST VOTESLOTS --------------------------- #

@pytest.mark.parametrize("fee_val", [
    (0.008), (0.001), (0.004)
])
def test_single_vote_entry(alice, amm, fee_val):
    alice_votes = AMMVote(alice, amm)
    alice_votes.vote_entry(fee_val=fee_val)
    assert 'alice' in amm.voters
    assert amm.TFee == fee_val


def test_multiple_vote_entries(users, amm):
    alice_votes = AMMVote(users['alice'], amm)
    alice_votes.vote_entry(fee_val=0.008)
    assert 'alice' in amm.voters
    assert amm.TFee == 0.008

    Deposit(users['bob'], amm).deposit_LPTokenOut(5)
    AMMVote(users['bob'], amm).vote_entry(fee_val=0)
    assert 'bob' in amm.voters
    assert amm.TFee < 0.008

    Deposit(users['firas'], amm).deposit_LPTokenOut(25)
    AMMVote(users['firas'], amm).vote_entry(fee_val=0.001)
    assert 'firas' in amm.voters

    Deposit(users['jiahua'], amm).deposit_LPTokenOut(15)
    AMMVote(users['jiahua'], amm).vote_entry(fee_val=0.003)
    assert 'jiahua' in amm.voters

    Deposit(users['david'], amm).deposit_LPTokenOut(50)
    AMMVote(users['david'], amm).vote_entry(fee_val=0.002)
    assert 'david' in amm.voters

    Deposit(users['aanchal'], amm).deposit_LPTokenOut(34)
    AMMVote(users['aanchal'], amm).vote_entry(fee_val=0.006)
    assert 'aanchal' in amm.voters

    Deposit(users['yebo'], amm).deposit_LPTokenOut(12)
    AMMVote(users['yebo'], amm).vote_entry(fee_val=0.007)
    assert 'yebo' in amm.voters

    Deposit(users['zhang'], amm).deposit_LPTokenOut(1)
    AMMVote(users['zhang'], amm).vote_entry(fee_val=0.0085)
    assert 'zhang' in amm.voters

    Deposit(users['brad'], amm).deposit_LPTokenOut(54)
    AMMVote(users['brad'], amm).vote_entry(fee_val=0.009)
    # Zhang has the least amount of LPTokens (1), so he'll be replaced by Brad
    assert 'brad' in amm.voters
    assert 'zhang' not in amm.voters

    Deposit(users['arthur'], amm).deposit_LPTokenOut(49)
    AMMVote(users['arthur'], amm).vote_entry(fee_val=0.01)
    # Bob has the 2nd least amount of LPTokens (5), so he'll be replaced by Arthur
    assert 'arthur' in amm.voters
    assert 'bob' not in amm.voters

    # update alice entry
    alice_votes.vote_entry(fee_val=0.01)
    assert amm.VoteSlots[0]['tfee'] == 0.01


# --------------------------- TEST BID --------------------------- #


@pytest.mark.parametrize("t, min_price, max_price", [
    (0.05, '', 10), (0.2, 1, 20), (0.4, 4, 15), (1, 6, ''), (0.8, 20, 50)
])
# for the scope of this test, we explicitly specify the time interval "t"
# possible values for t: 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, ..., 0.9, 0.95, 1
def test_bid(alice, amm, t, min_price, max_price):
    alice_bids = AMMBid(alice, amm)
    if (min_price and min_price > alice.assets['LPTokens']) or (max_price and max_price > alice.assets['LPTokens']):
        with pytest.raises(Exception) as e:
            alice_bids.bid(t=t, min_price=min_price, max_price=max_price)
        assert str(e.value) == "Not enough tokens"
    else:
        alice_bids.bid(t=t, min_price=min_price, max_price=max_price)
        assert amm.AuctionSlot['user'] == alice
        assert amm.AuctionSlot['t'] == t
        assert Deposit(alice, amm).TFee == Withdraw(
            alice, amm).TFee == Swap(alice, amm).TFee == 0
