Authors = 'Nicholas Haltmeyer <hanicho1@umbc.edu>',
               'Zeyu Ning          <zeyning1@umbc.edu>',
               'Kaustubh Agrahari  <kaus1@umbc.edu>'

Phase II New Eleusis Player

The game.py file contains all the components needed, it has imported the needed function and classes from new_eleusis.py and nzk.py. 

Some details for score function: the current score function only works for our own player but not for the adversaries provided; Since the adversaries does not keep track of the cards they played. The scoring method is aligned with the documentation.

Please run game.py for model testing, the current game.py will run a testing session containing our player and 3 adversaries, each player has a hand of 14 cards. The session will stop when the player has exhausted all the cards or any of them are calling prophet and returning hypothesis.