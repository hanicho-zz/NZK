# Put your program name in place of program_name

from nzk import *
from random import randint
from new_eleusis import *

global game_ended
game_ended = False


def generate_random_card():
    values = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    suits = ["S", "H", "D", "C"]
    return values[randint(0, len(values) - 1)] + suits[randint(0, len(suits) - 1)]


class Player(object):
    """
    'cards' is a list of three valid cards to be given by the dealer at the beginning of the game.
    """
    def __init__(self, cards):
        self.board_state = cards;
        self.hand = [generate_random_card() for i in range(14)]

    def play(self):
        """

        Your scientist should play a card out of its given hand OR return a rule, not both.
        'game_ended' parameter is a flag that is set to True once the game ends. It is False by default
        """
        return scientist(self.hand, game_ended)

       #  update board state
    def update_card_to_boardstate(card,result):

         """
         update your board state with card based on the result

         """








# The players in the game


# Set a rule for testing
rule = "if(is_royal(current), False)"
cards = ["10H", "2C", "4S"]
tree = parse(rule)


# player and adversary

player = Player(cards)
adversary1 = Adversary()
adversary2 = Adversary()
adversary3 = Adversary()

# The three cards that adhere to the rule


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
        if is_card(player_card_rule):
            # checking whether card played is correct or wrong
            temp_cards= [cards[-2],cards[-1], player_card_rule]
            result = tree.evaluate(tuple(temp_cards)) # (card1,card2,card3)
            if result:
                 del cards[0]
                 cards.append(player_card_rule)
            # player updating board state based on card played and result
            player.update_card_to_boardstate(player_card_rule, result)

        else:
            raise Exception('')

        # Adversary 1 plays
        ad1_card_rule = adversary1.play()
        if is_card(ad1_card_rule):
            temp_cards = [cards[-2], cards[-1], ad1_card_rule]
            result = tree.evaluate(tuple(temp_cards)) # (card1,card2,card3)
            if result:
                del cards[0]
                cards.append(ad1_card_rule)
            player.update_card_to_boardstate(ad1_card_rule, result)

        else:
            raise Exception('')

        # Adversary 2 plays
        ad2_card_rule = adversary2.play()
        if is_card(ad2_card_rule):
            temp_cards = [cards[-2], cards[-1], ad2_card_rule]
            result = tree.evaluate(tuple(temp_cards))  # (card1,card2,card3)
            if result:
                del cards[0]
                cards.append(ad2_card_rule)
            player.update_card_to_boardstate(ad2_card_rule, result)
        else:
            raise Exception('')

        # Adversary 3 plays
        ad3_card_rule = adversary3.play()
        if is_card(ad3_card_rule):
            temp_cards = [cards[-2], cards[-1], ad3_card_rule]
            result = tree.evaluate(tuple(temp_cards)) # (card1,card2,card3)
            if result:
                del cards[0]
                cards.append(ad3_card_rule)
            player.update_card_to_boardstate(ad3_card_rule, result)
        else:
            raise Exception('')

    except:
        game_ended = True
        break

# Everyone has to guess a rule
rule_player = player.play()

# Check if the guessed rule is correct and print the score
score(player)
