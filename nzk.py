#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__authors__ = ['Nicholas Haltmeyer <hanicho1@umbc.edu>',
               'Zeyu Ning          <zeyning1@umbc.edu>',
               'Kaustubh Agrahari  <kaus1@umbc.edu>']
__version__ = 'Phase II'
__license__ = 'GPLv3'

import copy
import random
import sys
import numpy as np
from abc import ABC
from functools import reduce
from itertools import product
from bitstring import Bits, BitArray
import new_eleusis as eleusis

TEE = 'equal(AS, AS)' # ⊤
EET = 'equal(AS, KC)' # ⊥

class Lattice(Bits):
    """Child of Bits supporting a partial order for the given Boolean lattice

    """

    def __lt__(self, other):
        return self <= other and self != other

    def __le__(self, other):
        for i in range(len(self)):
            if self[i] and not other[i]:
                return False

        return True

    def __gt__(self, other):
        return other < self

    def __ge__(self, other):
        return other <= self


class MutableLattice(BitArray):
    """Child of BitArray supporting a partial order for the given Boolean lattice

    """

    def __lt__(self, other):
        return self < other and self != other

    def __le__(self, other):
        for i in range(len(self)):
            if self[i] and not other[i]:
                return False

        return True

    def __gt__(self, other):
        return other < self

    def __ge__(self, other):
        return other <= self


class Latticizer(ABC):
    """Converts a card triple to a Lattice and a ``new_eleusis`` rule

    This class defines, under the provided schema, a mapping from a card
    triple to a conjunction of predicates. Supports mapping a conjunction
    of predicates to a Tree object given by ``new_eleusis``.

    Attributes:
        schema (Iterable): collection of predicates used to map a given triple

    """

    def __init__(self, schema):
        self._schema = schema

    def __repr__(self):
        return '<Latticizer {0}>'.format(self._schema)

    def __len__(self):
        return len(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def order(self):
        order = 1

        for each in self._schema:
            if 'previous2' in each:
                order = 3
            elif 'previous' in each and order < 3:
                order = 2

        return order

    def latticize(self, x):
        """Maps a card triple to an array of Boolean predicates

        Args:
            x (tuple[str]): card triple of ``(previous2, previous, current)``

        Returns:
            A Lattice with indices set to True if the given predicate
                in the schema is evaluated as True, False otherwise

        """

        b = MutableLattice(len(self))

        for i in range(len(self._schema)):
            rule = eleusis.parse(self._schema[i])

            # This happens because new_eleusis.py doesn't bool
            #   on {'True', 'False'} literals
            assert not isinstance(rule.evaluate(x), str)

            b[i] = rule.evaluate(x)

        return Lattice(b)

    def delatticize(self, b):
        """Maps an array of Boolean predicates to a rule in ``new_eleusis``

        Args:
            b (Lattice): array of Boolean predicates

        Returns:
            A Tree object given as the conjunction of predicates

        """

        def delatticize2(rules):
            splits = []

            if len(rules) == 1:
                return rules[0]
            elif len(rules) == 2:
                return 'andf({0}, {1})'.format(rules[0], rules[1])

            for i in range(0, len(rules), 2):
                splits.append('andf({0}, {1})'
                              .format(rules[i],
                                      rules[i+1]
                                      if i < len(rules)-1
                                      else TEE))

            return delatticize2(splits)

        rules = []
        c = 0

        for i in range(len(b)):
            if b[i]:
                rules.append(self._schema[i])
                c += 1

        return eleusis.parse(delatticize2(rules))


class SNN(ABC):
    """Switching Neural Network for pdpBf reconstruction

    Learns a partially defined positive Boolean function using the LSC
    algorithm. [1]_ Supports converting the learned function to a rule
    as defined in ``new_eleusis``.

    [1] Muselli, Marco, and Enrico Ferrari.
     "Coupling Logical Analysis of Data and Shadow Clustering for
     partially defined positive Boolean function reconstruction."
     *IEEE Transactions on Knowledge and Data Engineering* 23.1 (2011): 37-50.

    Attributes:
        n (int): number of predicates used in the model
        latticizer (Latticizer): mapping of cards to a predicate array
        implicants (list[Lattice]): learned classification gates
        degenerate (bool): whether or not the learned model is degenerate
        rule (Tree): representation of the learned rule in ``new_eleusis``

    """

    def __init__(self, latticizer):
        self._latticizer = latticizer
        self._implicants = []
        self._degenerate = False

    def __repr__(self):
        return '<SNN {0} : {1}>'.format(self.n, len(self._implicants))

    def __len__(self):
        """Counts the number of terms present over all implicants

        """

        if len(self._implicants) == 0:
            return 0

        counter = lambda a: a.count(True)
        adder = lambda a, b: a+b

        return reduce(adder, map(counter, self._implicants))

    @property
    def n(self):
        return len(self._latticizer)

    @property
    def latticizer(self):
        return self._latticizer

    @property
    def implicants(self):
        return copy.copy(self._implicants)

    @property
    def degenerate(self):
        return self._degenerate

    @property
    def rule(self):
        """Converts the learned rule into a ``Tree`` object

        """

        def rule2(rules):
            splits = []

            if len(rules) == 1:
                return rules[0]
            elif len(rules) == 2:
                return 'orf({0}, {1})'.format(rules[0], rules[1])

            for i in range(0, len(rules), 2):
                splits.append('orf({0}, {1})'
                              .format(rules[i],
                                      rules[i+1]
                                      if i < len(rules)-1
                                      else EET))

            return rule2(splits)

        if len(self._implicants) == 0:
            return eleusis.parse(EET)
        elif len(self._implicants) == 1:
            return self._latticizer.delatticize(self._implicants[0])

        rules = list(map(lambda a: self._latticizer.delatticize(a),
                         self._implicants))

        return eleusis.parse(rule2(rules))

    def classify(self, datum):
        """Performs classification of the provided Boolean array

        Args:
            datum (Lattice): the input array

        Returns:
            True if any implicant AND ``datum`` is equal to the given
                implicant, False otherwise

        """

        assert isinstance(datum, Lattice)
        assert len(datum) == self.n

        if len(self._implicants) == 0:
            return False

        return reduce(lambda a, b: a or b,
                      map(lambda a: a & datum == a, self._implicants))

    def train(self, data, depth=1, method='msc'):
        """Trains the model against the provided data

        Note:
            ``self.degenerate`` is set to True if the given data is not monotone

        Args:
            data (dict[Lattice, bool]): partially defined Boolean function
            depth (int): search depth used in LSC
            method (str): search heuristic used in LSC

        """

        assert isinstance(data, dict)
        assert depth >= 1 and depth <= self.n

        for datum in data:
            assert isinstance(datum, Lattice)
            assert len(datum) == self.n
            assert isinstance(data[datum], bool)

        T = list(filter(lambda x: data[x], data))
        F = list(filter(lambda x: not data[x], data))

        if SNN._monotone_check(T, F):
            self._degenerate = False
        else:
            self._degenerate = True

        method = method.lower()

        if method != 'msc' and method != 'dsc':
            raise NotImplementedError('Only LSC-MSC and LSC-DSC are implemented')

        self._implicants = self._lsc(T, F, depth, method)

    @staticmethod
    def _monotone_check(T, F):
        """Checks if the given partially defined Boolean function is monotone

        """

        # If there are any T that are less than F
        for x in T:
            if True in map(lambda y: x <= y, F):
                print('Warning: non-monotonic pdpBf given', file=sys.stderr)
                return False

        return True

    def _lsc(self, T, F, d, method):
        """Top level LSC algorithm

        Args:
            T (list[Lattice]): positive samples
            F (list[Lattice]): negative samples
            d (int): search depth
            method (str): search heuristic

        Returns:
            A list of implicants that satisfies the given data

        """

        A = self._lsc_depth(T, F, d, list(range(self.n)), [], [])

        # Let S be the set of patterns x ∈ T such that there is no
        #     a ∈ A with a ≤ x .
        notS = []

        for x in T:
            for a in A:
                if a <= x:
                    notS.append(x)

        S = list(filter(lambda x: x not in notS, T))

        if len(S) == 0:
            return A

        while len(S) > 0:
            x = random.choice(S)
            a = self._lsc_heuristic(x, T, F, S, method)
            A.append(a)

            # Remove from S all the patterns z for which a ≤ z
            rem = []

            for z in S:
                if a <= z:
                    rem.append(z)

            S = list(filter(lambda z: z not in rem, S))

        return A

    def _lsc_depth(self, T, F, d, I, J, A):
        """Performs the Depth procedure in LSC

        """

        while len(I) > 0:
            i = max(I)
            del I[I.index(i)]

            IJ = list(set(I) | set(J))
            sIJ = self._s(IJ)

            sIJ_lte_x = list(map(lambda x: sIJ <= x, T))
            a_lte_sIJ = list(map(lambda a: a <= sIJ, A))
            sIJ_lte_y = list(map(lambda y: sIJ <= y, F))

            if True in sIJ_lte_x and True not in a_lte_sIJ and True not in sIJ_lte_y:
                # If there is x ∈ T such that s(I ∪ J) ≤ x and
                #    there is no a ∈ A such that a ≤ s(I ∪ J) and
                #    there is no y ∈ F such that s(I ∪ J) ≤ y then

                A.append(sIJ)

                # Remove from A all the implicants a > s(I ∪ J) .
                rem = []

                for a in A:
                    if a > sIJ:
                        rem.append(a)

                A = list(filter(lambda a: a not in rem, A))
            else:
                # If I is not empty and |I ∪ J| > n − d
                #    then call Depth( T, F, d, I, J, A ).
                if len(I) > 0 and len(IJ) > self.n-d:
                    self._lsc_depth(T, F, d, copy.copy(I), copy.copy(J), A)

            J.append(i)

        return A

    def _lsc_heuristic(self, x, T, F, S, method):
        """ Heuristic evaluation used in LSC

        """

        I = self._P(x)
        J = []

        Si0 = dict(zip(I, map(lambda i: SNN._zeros(S, i), I)))
        Ti0 = dict(zip(I, map(lambda i: SNN._zeros(T, i), I)))
        Fi0 = dict(zip(I, map(lambda i: SNN._zeros(F, i), I)))

        while len(I) > 0:
            IJ = list(set(I) | set(J))
            pIJ = self._p(IJ)
            # For each i ∈ I compute d l (p(I ∪ J), F i 0 ) .
            dl_pIJ_Fi0 = dict(zip(I, map(lambda i: SNN._Dl(pIJ, Fi0[i])
                                                   if len(Fi0[i]) > 0
                                                   else np.nan, I)))

            rem = []

            for i in I:
                if dl_pIJ_Fi0[i] == 1:
                    rem.append(i)

            I = list(filter(lambda i: i not in rem, I))
            J.extend(rem)

            if len(I) != 0:
                # Remove from I the index i that maximizes the cost vector
                costs = {}

                for i in I:
                    if method == 'msc':
                        costs[i] = (len(Si0[i]),
                                    len(Ti0[i]),
                                    dl_pIJ_Fi0[i])
                    elif method == 'dsc':
                        costs[i] = (dl_pIJ_Fi0[i],
                                    len(Ti0[i]),
                                    len(Si0[i]))
                    else:
                        assert method == 'msc' or method == 'dsc'

                del I[I.index(max(costs, key=lambda c: costs[c]))]

        return self._p(J)

    def _P(self, a):
        """Produces the subset of indices for which a[j] == True

        """

        J = []

        for j in range(len(a)):
            if a[j]:
                J.append(j)

        return J

    def _p(self, J):
        """Produces a Boolean array set to True of all indices in J

        """

        for j in J:
            assert j <= self.n

        a = MutableLattice(self.n)

        for j in J:
            a[j] = True

        return Lattice(a)

    def _S(self, a):
        """Produces the subset of indices for which a[j] == False

        """

        J = []

        for j in range(len(a)):
            if not a[j]:
                J.append(j)

        return J

    def _s(self, J):
        """Produces a Boolean array set to False for all indices in J

        """

        for j in J:
            assert j <= self.n

        a = MutableLattice(self.n)
        a.invert()

        for j in J:
            a[j] = False

        return Lattice(a)

    @staticmethod
    def _dl(x, y):
        """Computes the lower distance between two Boolean arrays

        The lower distance is defined as the number of bits for which
        x[i] is True and y[i] is False.

        Note:
            Primary bottleneck function during learning
        """

        assert len(x) == len(y)

        dist = 0

        for i in range(len(x)):
            if x[i] and not y[i]:
                dist += 1

        return dist

    @staticmethod
    def _du(x, y):
        """Computes the upper distance between two Boolean arrays

        """

        return SNN._dl(y, x)

    @staticmethod
    def _Dl(x, Z):
        """Computes the minimum lower distance for the given list

        """

        return min(map(lambda z: SNN._dl(x, z), Z))

    @staticmethod
    def _Du(x, Z):
        """Computes the minimum upper distance for the given list

        """

        return min(map(lambda z: SNN._du(x, z), Z))

    @staticmethod
    def _zeros(Z, i):
        """Computes the subset of Z where the given index i is set to False

        """

        assert len(Z) > 0
        assert False not in map(lambda z: isinstance(z, Lattice), Z)
        assert False not in map(lambda z: len(z) > i, Z)

        return list(filter(lambda z: not z[i], Z))


class Game(ABC):
    """Encapsulation for a game of New Eleusis

    """

    def __init__(self, first, second):
        # assert rule.evaluate((None, None, first))
        # assert rule.evaluate((None, first, second))

        # self._rule = rule
        self._board = [(first, []), (second, [])]

    def __repr__(self):
        return '<Game {0}, {1}>'.format(self._board, self._rule)

#    @property
#    def rule(self):
#        return self._rule

    @property
    def board(self):
        return copy.copy(self._board)

    @property
    def history(self):
        hist = []

        for pair in self._board:
            hist.append((pair[0], True))

            for illegal in pair[1]:
                hist.append((illegal, False))

        return hist

    @property
    def previous(self):
        assert len(self._board) >= 2

        return self.history[-1][0]

    @property
    def previous2(self):
        assert len(self._board) >= 2

        return self.history[-2][0]

#    def score(self, rule, hypothesis):
#        """Computes the score of a given hypothesis
#
#        """
#
#        hist = self.history
#        score = 0
#        covers = True
#
#        for i in range(len(hist)):
#            # +1 for every successful play over 20 and under 200
#            # +2 for every failed play
#            if i >= 19 and i < 200:
#                if hist[i][1]:
#                    score += 1
#                else:
#                    score += 2
#
#            previous2 = hist[i-2][0] if i >= 2 else None
#            previous = hist[i-1][0] if i >= 1 else None
#            current = hist[i][0]
#
#            if 'previous2' in str(hypothesis) and previous2 is None:
#                continue
#            elif 'previous' in str(hypothesis) and previous is None:
#                continue
#
#            if (covers and
#                hypothesis.evaluate((previous2, previous, current)) != hist[i][1]):
#                covers = False
#
#        # +15 for a rule that is not equivalent to the correct rule
#        # +30 for a rule that does not describe all cards on the board
#        if not covers:
#            score += 45
#        else:
#            if not Game._rules_eq(rule, hypothesis):
#                score += 15
#
#        return score

    def grams(self, n):
        """Compute n-grams of the board, mapped to legality

        """

        grams = {}
        hist = self.history

        for i in range(len(hist)):
            # Pad the n-gram with None
            gram = [None]*(n-(i+1))
            padding = len(gram)

            for j in range(n-padding):
                # Prepend to the list
                gram.insert(padding, hist[i-j][0])

            grams[tuple(gram)] = hist[i][1]

        return grams

    def play(self, card, legal):
        """Plays the card provided

        Args:
            card (str): French playing card (e.g., 'AS', '1D', ...)

        Returns:
            Whether the play was legal
        """

        # legal = self._rule.evaluate((self.previous2, self.previous, card))

        if legal:
            self._board.append((card, []))
        else:
            self._board[-1][1].append(card)

        return legal

    @staticmethod
    def _rules_eq(rule1, rule2):
        """Determines whether the given rules are equal

        Note:
            Equality here is determined by checking if every possible input
                for both rules maps to the same output

        """

        if 'previous2' in str(rule1) or 'previous2' in str(rule2):
            n = 3
        elif 'previous' in str(rule1) or 'previous' in str(rule2):
            n = 2
        else:
            n = 1

        deck = list(map(lambda p: ''.join(p),
                        product(['A', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                 'J', 'Q', 'K'],
                                ['S', 'H', 'D', 'C'])))

        if n == 1:
            for card in deck:
                if (rule1.evaluate((None, None, card))
                 != rule2.evaluate((None, None, card))):
                    return False
        elif n == 2:
            decks = list(product(deck, deck))

            for cards in decks:
                if (rule1.evaluate((None, cards[0], cards[1]))
                 != rule2.evaluate((None, cards[0], cards[1]))):
                    return False
        elif n == 3:
            decks = list(product(deck, deck, deck))

            for cards in decks:
                if (rule1.evaluate(cards) != rule2.evaluate(cards)):
                    return False

        return True


class Scientist(ABC):
    """Encapsulation for scientist operation in New Eleusis

    Attributes:
        game (Game): game attached to the scientist
        models (list[SNN]): list of models being used
        played3 (set[tuple]): trigrams already played
        played2 (set[tuple]): bigrams already played
        played1 (set[tuple]): unigrams already played
        cutoff (int): number of iterations to end play where hypothesis does
            not change

    """

    def __init__(self, game, latticizers, cutoff=20):
        self.game = game
        self.models = []
        self.played3 = set()
        self.played2 = set()
        self.played1 = set()
        self.cutoff = cutoff

        for latticizer in latticizers:
            self.models.append(SNN(latticizer))

    def __repr__(self):
        return '<Scientist {0}>'.format(self.hypothesis)

    @property
    def belief(self):
        return min(self.models, key=lambda model: len(model))
    
    @property
    def hypothesis(self):
        """Returns the most concise rule currently held

        """

        return self.belief.rule

    def choice(self, hand):
        """Selects a card to play

        TODO:
            Make choices that specifically challenge predicates in the hypothesis
                e.g., if equal(value(current), 2) set, try 2? with varying suits

        """

        previous2 = self.game.previous2
        previous = self.game.previous
        plays = {}

        for card in hand:
            plays[card] = 0

        for model in self.models:
            hypothesis = model.rule

            for card in hand:
                # Ignore plays we've already made
                if model.latticizer.order == 3 and (previous2, previous, card) in self.played3:
                    continue
                elif model.latticizer.order == 2 and (previous, card) in self.played2:
                    continue
                elif (card) in self.played1:
                    continue

                if hypothesis.evaluate((previous2, previous, card)):
                    plays[card] += 1

        top = []
        val = -1

        # Randomly select the card to play in the event of a tie
        for play in plays:
            if plays[play] > val:
                top = [play]
                val = plays[play]
            elif plays[play] == val:
                top.append(play)

        return random.choice(top)

    def update(self):
        """Updates the models according to the current board

        TODO:
            Use ``subprocess`` module to run updates in parallel

        """

        grams = self.game.grams(3)

        for gram in grams:
            self.played3.add(gram)
            self.played2.add(gram[1:])
            self.played1.add(gram[2:])

        for model in self.models:
            data = {}

            for gram in grams:
                # This occurs for n-gram models where n > 1
                # for grams in which None is present (the first plays)
                if gram[0] is None and model.latticizer.order == 3:
                    continue
                elif gram[1] is None and model.latticizer.order >= 2:
                    continue

                bits = model.latticizer.latticize(gram)
                data[bits] = grams[gram]

            model.train(data)

    def _phase1(self):
        """Boilerplate for Phase I operation

        Note:
            Cuts off play when the hypothesis does not change

        """

        i = 2
        c = 0
        done = False

        while not done:
            old_repr = str(self.hypothesis)
            self.update()
            self.game.play(self.choice)
            new_repr = str(self.hypothesis)

            if old_repr == new_repr:
                c += 1
            else:
                c = 0

            i += 1

            # print(i, c)
            # print(new_repr)

            if i >= 30 or c >= self.cutoff:
                done = True


unigram = Latticizer((
    'equal(color(current), B)',
    'equal(color(current), R)',
    'equal(suit(current), S)',
    'equal(suit(current), H)',
    'equal(suit(current), D)',
    'equal(suit(current), C)',
    'equal(value(current), 1)',
    'equal(value(current), 2)',
    'equal(value(current), 3)',
    'equal(value(current), 4)',
    'equal(value(current), 5)',
    'equal(value(current), 6)',
    'equal(value(current), 7)',
    'equal(value(current), 8)',
    'equal(value(current), 9)',
    'equal(value(current), 10)',
    'equal(value(current), 11)',
    'equal(value(current), 12)',
    'equal(value(current), 13)',
    'is_royal(current)',
    'notf(is_royal(current))',
    'even(current)',
    'odd(current)'
))

bigram = Latticizer((
    'equal(suit(current), S)',
    'equal(suit(current), H)',
    'equal(suit(current), D)',
    'equal(suit(current), C)',
    'equal(color(current), B)',
    'equal(color(current), R)',
    'equal(value(current), 1)',
    'equal(value(current), 2)',
    'equal(value(current), 3)',
    'equal(value(current), 4)',
    'equal(value(current), 5)',
    'equal(value(current), 6)',
    'equal(value(current), 7)',
    'equal(value(current), 8)',
    'equal(value(current), 9)',
    'equal(value(current), 10)',
    'equal(value(current), 11)',
    'equal(value(current), 12)',
    'equal(value(current), 13)',
    'is_royal(current)',
    'notf(is_royal(current))',
    'even(current)',
    'odd(current)',
    'equal(suit(previous), S)',
    'equal(suit(previous), H)',
    'equal(suit(previous), D)',
    'equal(suit(previous), C)',
    'equal(color(previous), B)',
    'equal(color(previous), R)',
    'equal(value(previous), 1)',
    'equal(value(previous), 2)',
    'equal(value(previous), 3)',
    'equal(value(previous), 4)',
    'equal(value(previous), 5)',
    'equal(value(previous), 6)',
    'equal(value(previous), 7)',
    'equal(value(previous), 8)',
    'equal(value(previous), 9)',
    'equal(value(previous), 10)',
    'equal(value(previous), 11)',
    'equal(value(previous), 12)',
    'equal(value(previous), 13)',
    'is_royal(previous)',
    'notf(is_royal(previous))',
    'even(previous)',
    'odd(previous)',
    'equal(current, previous)',
    'less(current, previous)',
    'greater(current, previous)',
    'equal(suit(current), suit(previous))',
    'notf(equal(suit(current), suit(previous)))',
    'equal(color(current), color(previous))',
    'notf(equal(color(current), color(previous)))',
    'equal(value(current), value(previous))',
    'notf(equal(value(current), value(previous)))',
))

trigram = Latticizer((
    'equal(suit(current), S)',
    'equal(suit(current), H)',
    'equal(suit(current), D)',
    'equal(suit(current), C)',
    'equal(color(current), B)',
    'equal(color(current), R)',
    'equal(value(current), 1)',
    'equal(value(current), 2)',
    'equal(value(current), 3)',
    'equal(value(current), 4)',
    'equal(value(current), 5)',
    'equal(value(current), 6)',
    'equal(value(current), 7)',
    'equal(value(current), 8)',
    'equal(value(current), 9)',
    'equal(value(current), 10)',
    'equal(value(current), 11)',
    'equal(value(current), 12)',
    'equal(value(current), 13)',
    'is_royal(current)',
    'notf(is_royal(current))',
    'even(current)',
    'odd(current)',
    'equal(suit(previous), S)',
    'equal(suit(previous), H)',
    'equal(suit(previous), D)',
    'equal(suit(previous), C)',
    'equal(color(previous), B)',
    'equal(color(previous), R)',
    'equal(value(previous), 1)',
    'equal(value(previous), 2)',
    'equal(value(previous), 3)',
    'equal(value(previous), 4)',
    'equal(value(previous), 5)',
    'equal(value(previous), 6)',
    'equal(value(previous), 7)',
    'equal(value(previous), 8)',
    'equal(value(previous), 9)',
    'equal(value(previous), 10)',
    'equal(value(previous), 11)',
    'equal(value(previous), 12)',
    'equal(value(previous), 13)',
    'is_royal(previous)',
    'notf(is_royal(previous))',
    'even(previous)',
    'odd(previous)',
    'equal(suit(previous2), S)',
    'equal(suit(previous2), H)',
    'equal(suit(previous2), D)',
    'equal(suit(previous2), C)',
    'equal(color(previous2), B)',
    'equal(color(previous2), R)',
    'equal(value(previous2), 1)',
    'equal(value(previous2), 2)',
    'equal(value(previous2), 3)',
    'equal(value(previous2), 4)',
    'equal(value(previous2), 5)',
    'equal(value(previous2), 6)',
    'equal(value(previous2), 7)',
    'equal(value(previous2), 8)',
    'equal(value(previous2), 9)',
    'equal(value(previous2), 10)',
    'equal(value(previous2), 11)',
    'equal(value(previous2), 12)',
    'equal(value(previous2), 13)',
    'is_royal(previous2)',
    'notf(is_royal(previous2))',
    'even(previous2)',
    'odd(previous2)',
    'equal(current, previous)',
    'less(current, previous)',
    'greater(current, previous)',
    'equal(suit(current), suit(previous))',
    'notf(equal(suit(current), suit(previous)))',
    'equal(color(current), color(previous))',
    'notf(equal(color(current), color(previous)))',
    'equal(value(current), value(previous))',
    'notf(equal(value(current), value(previous)))',
    'equal(current, previous2)',
    'less(current, previous2)',
    'greater(current, previous2)',
    'equal(suit(current), suit(previous2))',
    'notf(equal(suit(current), suit(previous2)))',
    'equal(color(current), color(previous2))',
    'notf(equal(color(current), color(previous2)))',
    'equal(value(current), value(previous2))',
    'notf(equal(value(current), value(previous2)))',
    'equal(previous, previous2)',
    'less(previous, previous2)',
    'greater(previous, previous2)',
    'equal(suit(previous), suit(previous2))',
    'notf(equal(suit(previous), suit(previous2)))',
    'equal(color(previous), color(previous2))',
    'notf(equal(color(previous), color(previous2)))',
    'equal(value(previous), value(previous2))',
    'notf(equal(value(previous), value(previous2)))'
))

def legal_cards(rule, previous, previous2):
    deck = list(map(lambda p: ''.join(p),
                    product(['A', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'],
                            ['S', 'H', 'D', 'C'])))

    return list(filter(lambda card: rule.evaluate((previous2, previous, card)), deck))

#G = None
#S = None
#
#def setRule(rule):
#    global G
#    global S
#    try:
#        rule = eleusis.parse(rule)
#        first = random.choice(legal_cards(rule, None, None))
#        second = random.choice(legal_cards(rule, first, None))
#    except:
#        card1 = input('Please input the first card:')
#        card2 = input('Please input the second card:')
#        first = card1
#        second = card2
#    G = Game(rule, first, second)
#    S = Scientist(G, [unigram, bigram])
#
#def rule():
#    global G
#
#    return G.rule
#
#def boardState():
#    global G
#
#    return G.board
#
#def play(card):
#    global G
#
#    return G.play(card)
#
#def scientist():
#    global S
#
#    S._phase1()
#    return S.hypothesis
#
#def score():
#    global G
#
#    return G.score(S.hypothesis)
#
#
#if __name__ == '__main__':
#    import sys
#
#    seed = random.randrange(sys.maxsize)
#    random.seed(seed)
#    print('==== {0} ====\n'.format(seed))
#
#    setRule('equal(color(previous), color(current))')
#    print('rule:', rule())
#    print('hypothesis:', scientist())
#    print('boardState:', boardState())
#    print('score:', score())
