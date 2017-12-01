#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__authors__ = ['Nicholas Haltmeyer <hanicho1@umbc.edu>',
               'Zeyu Ning          <zeyning1@umbc.edu>',
               'Kaustubh Agrahari  <kaus1@umbc.edu>']
__version__ = 'Phase I'
__license__ = 'GPLv3'

import nzk
from random import randint
import new_eleusis

global game_ended
game_ended = False

def generate_random_card():
    values = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    suits = ["S", "H", "D", "C"]
    return values[randint(0, len(values) - 1)] + suits[randint(0, len(suits) - 1)]

class Player(object):
    """
    'cards' is a list of three valid cards to be given by the dealer at the beginning of the game.
    """
    def __init__(self, cards):
        self.hand = [generate_random_card() for i in range(14)]
        self.game = nzk.Game(cards[0], cards[1])
        self.game.play(cards[2], True)
        self.played = [] # Keeps track of the indexes in the history that are ours

        self.scientist = nzk.Scientist(self.game, [nzk.unigram, nzk.bigram, nzk.trigram])
        self.threshold = 5
        self.counter = 0

    def play(self):
        print('Play time!')

        if (len(self.scientist.belief) < self.threshold
            and self.counter > 5):
            return str(self.scientist.hypothesis)

        if(len(self.hand)):
            playCard = self.scientist.choice(self.hand)
            del self.hand[self.hand.index(playCard)]

            print ('Playing card:', playCard)
            self.played.append(len(self.game.history))
            
            return playCard
        else:
            print('Out of cards!')
            return str(self.scientist.hypothesis)

    def update_card_to_boardstate(self, card, result):
        self.counter += 1
        self.game.play(card, result)
        self.scientist.update()


class Adversary(object):
    def __init__(self):
        self.hand = [generate_random_card() for i in range(14)]

    def play(self):
        """
        'cards' is a list of three valid cards to be given by the dealer at the beginning of the game.
        Your scientist should play a card out of its given hand.
        """
        # Return a rule with a probability of 1/14
        prob_list = [i for i in range(14)]
        prob = prob_list[randint(0, 13)]
        if prob == 100:
            # Generate a random rule
            rule = ""
            conditions = ["equal", "greater"]
            properties = ["suit", "value"]
            cond = conditions[randint(0, len(properties) - 1)]
            if cond == "greater":
                prop = "value"
            else:
                prop = properties[randint(0, len(properties) - 1)]

            rule += cond + "(" + prop + "(current), " + prop + "(previous)), "
            return rule[:-2] + ")"
        else:
            return self.hand[randint(0, len(self.hand) - 1)]


# The players in the game
# Set a rule for testing
rule = 'equal(color(previous), color(current))'
cards = ["10C", "2C", "4S"]
tree = new_eleusis.parse(rule)

# player and adversary
player = Player(cards)
adversary1 = Adversary()
adversary2 = Adversary()
adversary3 = Adversary()


"""
In each round scientist is called and you need to return a card or rule.
The cards passed to scientist are the last 3 cards played.
Use these to update your board state.
"""
for round_num in range(14):
    # Each player plays a card or guesses a rule
    try:
        # Player 1 plays
        player_card_rule = player.play()
        if new_eleusis.is_card(player_card_rule):
            # checking whether card played is correct or wrong
            temp_cards= [cards[-2],cards[-1], player_card_rule]
            result = tree.evaluate(tuple(temp_cards)) # (card1,card2,card3)
            player.update_card_to_boardstate(player_card_rule, result)
            if result:
                 del cards[0]
                 cards.append(player_card_rule)
            # player updating board state based on card played and result
        else:
            raise Exception('player1 exception')

        # Adversary 1 plays
        ad1_card_rule = adversary1.play()
        if new_eleusis.is_card(ad1_card_rule):
            temp_cards = [cards[-2], cards[-1], ad1_card_rule]
            result = tree.evaluate(tuple(temp_cards)) # (card1,card2,card3)
            player.update_card_to_boardstate(ad1_card_rule, result)
            if result:
                del cards[0]
                cards.append(ad1_card_rule)
        else:
            raise Exception('adv1 exception')

        # Adversary 2 plays
        ad2_card_rule = adversary2.play()
        if new_eleusis.is_card(ad2_card_rule):
            temp_cards = [cards[-2], cards[-1], ad2_card_rule]
            result = tree.evaluate(tuple(temp_cards))  # (card1,card2,card3)
            player.update_card_to_boardstate(ad2_card_rule, result)
            if result:
                del cards[0]
                cards.append(ad2_card_rule)
        else:
            raise Exception('adv2 exception')

        # Adversary 3 plays
        ad3_card_rule = adversary3.play()
        if new_eleusis.is_card(ad3_card_rule):
            temp_cards = [cards[-2], cards[-1], ad3_card_rule]
            result = tree.evaluate(tuple(temp_cards)) # (card1,card2,card3)
            player.update_card_to_boardstate(ad3_card_rule, result)
            if result:
                del cards[0]
                cards.append(ad3_card_rule)
        else:
            raise Exception('adv3 exception')

    except Exception as e:
        print(e)
        game_ended = True
        break

# Everyone has to guess a rule
rule_player = player.play()

def score(player, rule):
    """Computes the score of a given hypothesis

    """

    hist = player.game.history
    score = 0
    covers = True

    for i in range(len(hist)):
        # +1 for every successful play over 20 and under 200
        # +2 for every failed play
        if i >= 19 and i < 200 and i in player.played:
            if hist[i][1]:
                score += 1
            else:
                score += 2

        previous2 = hist[i-2][0] if i >= 2 else None
        previous = hist[i-1][0] if i >= 1 else None
        current = hist[i][0]

        if 'previous2' in str(player.scientist.hypothesis) and previous2 is None:
            continue
        elif 'previous' in str(player.scientist.hypothesis) and previous is None:
            continue

        if (covers and
            player.scientist.hypothesis.evaluate((previous2, previous, current)) != hist[i][1]):
            covers = False

    # +15 for a rule that is not equivalent to the correct rule
    # +30 for a rule that does not describe all cards on the board
    if not covers:
        score += 45
    else:
        if not nzk.Game._rules_eq(rule, player.scientist.hypothesis):
            score += 15

    return score

# Check if the guessed rule is correct and print the score
# print('score:', player.game.score(tree, player.scientist.hypothesis))
print('Score:', score(player, tree))
print('True rule:', tree)
print('Final Hypothesis:', player.scientist.hypothesis)