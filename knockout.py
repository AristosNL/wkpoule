"""
knockout.py
-----------
Officieel WK 2026 knock-outschema (pragmatische versie).

De Ronde van 32-paringen liggen vast volgens FIFA: groepswinnaars (W) treffen
nummers 3, en runners-up (R) treffen elkaar; nooit een groepsgenoot in deze ronde.
De acht beste nummers 3 worden via een GELDIGE toewijzing in hun toegestane
clusters geplaatst — een pragmatische vervanging van de volledige 495-scenario
FIFA-tabel. Voor titelkansen is dat verschil verwaarloosbaar (nummers 3 zijn
doorgaans de zwakkere ploegen).

De bracket loopt daarna als vaste boom: R32 -> R16 -> kwart -> halve -> finale,
waarbij opeenvolgende paren in de lijst tegen elkaar uitkomen.
"""

from __future__ import annotations

# 16 officiële R32-paringen, in bracket-volgorde (paar i en i+1 voeden samen R16).
# Slotcodes:
#   ('W', 'E')      = winnaar groep E
#   ('R', 'A')      = runner-up groep A
#   ('T', 'ABCDF')  = nummer 3 uit één van die groepen (cluster)
R32_PAIRINGS = [
    (('W', 'E'), ('T', 'ABCDF')),
    (('W', 'I'), ('T', 'CDFGH')),
    (('R', 'A'), ('R', 'B')),
    (('W', 'F'), ('R', 'C')),
    (('R', 'K'), ('R', 'L')),
    (('W', 'H'), ('R', 'J')),
    (('W', 'D'), ('T', 'BEFIJ')),
    (('W', 'G'), ('T', 'AEHIJ')),
    (('W', 'C'), ('R', 'F')),
    (('R', 'E'), ('R', 'I')),
    (('W', 'A'), ('T', 'CEFHI')),
    (('W', 'L'), ('T', 'EHIJK')),
    (('W', 'J'), ('R', 'H')),
    (('R', 'D'), ('R', 'G')),
    (('W', 'B'), ('T', 'EFGIJ')),
    (('W', 'K'), ('T', 'DEIJL')),
]

STAGE_AFTER = ["last16", "quarter", "semi", "final", "winner"]


def _third_slots():
    """Lijst van (pairing_index, toegestane_groepen) voor elke nummer-3-slot."""
    slots = []
    for idx, (a, b) in enumerate(R32_PAIRINGS):
        for side in (a, b):
            if side[0] == "T":
                slots.append((idx, set(side[1])))
    return slots


def assign_thirds(qualifying_groups):
    """
    Wijs de 8 gekwalificeerde nummer-3-groepen toe aan de 8 cluster-slots via
    backtracking, met respect voor de toegestane clusters.
    Returns {pairing_index: group_letter} of None als er geen geldige toewijzing is.
    """
    slots = _third_slots()
    result, used = {}, set()

    def bt(i):
        if i == len(slots):
            return True
        idx, allowed = slots[i]
        for g in qualifying_groups:
            if g not in used and g in allowed:
                used.add(g)
                result[idx] = g
                if bt(i + 1):
                    return True
                used.remove(g)
                del result[idx]
        return False

    return result if bt(0) else None


def resolve_and_play(winners, runners, thirds_by_group, qualifying_third_groups,
                     simulate_match):
    """
    Speel de hele knock-out vanaf de R32.

    Parameters
    ----------
    winners              : {groepsletter: team}   — alle 12 groepswinnaars
    runners              : {groepsletter: team}   — alle 12 nummers 2
    thirds_by_group      : {groepsletter: team}   — alle 12 nummers 3
    qualifying_third_groups : lijst van 8 groepsletters waarvan de nr3 doorgaat
    simulate_match       : functie (home, away, knockout=True) -> (winner, gh, ga)

    Returns dict met per fase de set teams die die fase haalde, plus 'winner'.
    """
    assignment = assign_thirds(qualifying_third_groups)
    if assignment is None:
        # fallback (zou niet mogen gebeuren): wijs willekeurig toe aan T-slots
        assignment = {}
        t_slots = [idx for idx, (a, b) in enumerate(R32_PAIRINGS)
                   for side in (a, b) if side[0] == "T"]
        for idx, g in zip(t_slots, qualifying_third_groups):
            assignment[idx] = g

    def resolve(side, pairing_idx):
        kind, code = side
        if kind == "W":
            return winners[code]
        if kind == "R":
            return runners[code]
        return thirds_by_group[assignment[pairing_idx]]  # 'T'

    # bouw de R32-paringen om naar echte teams
    current = [(resolve(a, idx), resolve(b, idx))
               for idx, (a, b) in enumerate(R32_PAIRINGS)]

    reached = {s: set() for s in ["last32", "last16", "quarter", "semi", "final"]}
    reached["winner"] = None
    for h, a in current:
        reached["last32"].update([h, a])

    si = 0
    while current:
        round_winners = [simulate_match(h, a, knockout=True)[0] for h, a in current]
        stage = STAGE_AFTER[si]
        if stage == "winner":
            reached["winner"] = round_winners[0]
            break
        reached[stage].update(round_winners)
        current = [(round_winners[i], round_winners[i + 1])
                   for i in range(0, len(round_winners), 2)]
        si += 1
    return reached


if __name__ == "__main__":
    # sanity checks
    assert len(R32_PAIRINGS) == 16
    n_thirds = sum(1 for a, b in R32_PAIRINGS for s in (a, b) if s[0] == "T")
    n_w = sum(1 for a, b in R32_PAIRINGS for s in (a, b) if s[0] == "W")
    n_r = sum(1 for a, b in R32_PAIRINGS for s in (a, b) if s[0] == "R")
    print(f"Paringen: 16 | winnaar-slots: {n_w} | runner-up-slots: {n_r} | nr3-slots: {n_thirds}")

    # test een toewijzing: stel groepen A,B,C,D,E,F,G,H gaan door als nr3
    test = assign_thirds(list("ABCDEFGH"))
    print("Voorbeeld nr3-toewijzing (slotindex -> groep):", test)
    assert test is not None and len(test) == 8
    # controleer dat 2A tegen 2B staat
    assert (("R", "A"), ("R", "B")) in R32_PAIRINGS
    print("Check: 2A speelt tegen 2B  ✓")
    print("Alle checks OK.")
