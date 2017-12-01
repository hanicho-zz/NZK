# Put your program name in place of program_name

import nzk
from random import randint
import new_eleusis
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
        global G
        global S
        G = nzk.Game(tree, cards[-2], cards[-1])
        S = nzk.Scientist(G, [nzk.unigram, nzk.bigram])

    def play(self):
        print('Play time!')
        if(len(self.hand)):
            card_list = []
            for (histitem, legality) in list(reversed(G.history)):
                #print(legality)
                if(legality) and len(card_list)<2:
                    card_list.append(histitem)
                elif len(card_list)>=2:
                    break 
                else:
                    continue
            for itemNum in range(len(self.hand)):
                card_tuple = tuple(reversed([self.hand[itemNum]] + card_list))
                result = tree.evaluate(card_tuple)
                if not result:
                    #Playing cards which are illegal with hypothesis
                    playCard = self.hand.pop(itemNum)
                    break
                else:
                    #If all cards are legal, just pop the last one
                    playCard = self.hand.pop()
                    break
            print ('Playing card:', playCard)
            return playCard
        else:
            print('Out of cards!')
       #  update board state
    def update_card_to_boardstate(self, card, result):
        G.play(card)





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
        if prob == 4:
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
cards = ["10H", "2C", "4S"]
tree = new_eleusis.parse(rule)


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
            raise Exception('')

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
            raise Exception('')

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
            raise Exception('')

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
            raise Exception('')
    except:
        print('exception')
'''
    except:
        game_ended = True
        break
'''

# Everyone has to guess a rule
rule_player = player.play()

def scientist():
    global S

    S._phase1()
    return S.hypothesis
# Check if the guessed rule is correct and print the score
scientist()
print('score:', G.score(S.hypothesis))
print('Final Hypothesis:', S.hypothesis)