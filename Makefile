

define-challenge:
	dt-challenges-cli define -C ./object-detection/assets/environment/challenges/ --config lx22-objdet.challenge.yaml
	dt-challenges-cli define -C ./modcon/assets/environment/challenges/ --config lx22-modcon.challenge.yaml
	dt-challenges-cli define -C ./state-estimation/assets/environment/challenges --config lx22-state-estimation.challenge.yaml
	dt-challenges-cli define -C ./visual-lane-servoing/assets/environment/challenges/ --config lx22-visservoing.challenge.yaml
