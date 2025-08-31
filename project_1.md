TicTacToe code in *Code/Sutton-Barto/chapter01/tic_tac_toe.py*

1. Two actions
	1. Search
		1. Gives more change of finding a can
	2. Wait
		1. Less chance of finding a can
2. Two probabilities
	1. Alpha
	2. Beta
3. There is a battery and a charging station
4. Battery states:
	1. High
	2. Low
5. Flow:
	1. Check battery
	2. Decide if 
		1. Searches for cans
		2. Remains stationary and wait someone to give him a can
		3. Get back home to recharge
	3. When battery is high, recharging is not allowed
6. Action sets:
	1. Action(high):
		1. Search
		2. Wait
	2. Action(low):
		1. Search
		2. Wait
		3. Recharge
7. Rewards:
	1. Nothing happens => 0
	2. Battery ends => negative (-3)
	3. Gets cans => positive (+1)
8. Searching => More cans but change of battery level depleting
	1. High battery depletes to low battery
		1. Probabilitty alpha to keep energy at high
	2. Low battery depletes to no battery
		1. Probability of beta to keep battery at low
9. No cans can be found when going back to recharg
10. No cans can be found on the step that depletes battery (form high to low and low to no battery)