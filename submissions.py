import numpy as np
from HMM import *
from alarm import *
from carnet import *

# # load
# model = HMM()
# model.load('two_english')

# # generate
# model = HMM()
# model.load('partofspeech.browntags.trained')

# random_observation = model.generate(20)

# print('Generated Observation:', random_observation)

# # forward
# model = HMM()
# model.load('partofspeech.browntags.trained')

# with open('ambiguous_sents.obs', 'r') as obs_file:
#     lines = obs_file.read().splitlines()
#     observation = Observation([], lines[0].split())

# final_state = model.forward(observation)
# print("Most likely final state:", final_state)

# # viterbi
# model = HMM()
# model.load('partofspeech.browntags.trained')


# with open('ambiguous_sents.obs', 'r') as obs_file:
#     lines = obs_file.read().splitlines()
#     observation = Observation([], lines[0].split())

# state_sequence = model.viterbi(observation)

# print("Most likely sequence of states:", ' '.join(state_sequence))

# #
# # Probability of Mary Calling given John called
# q1 = alarm_infer.query(variables=["MaryCalls"], evidence={"JohnCalls": "yes"})
# print(q1)
# # Probability of both John and Mary calling given Alarm
# q2 = alarm_infer.query(variables=["JohnCalls", "MaryCalls"], evidence={"Alarm": "yes"})
# print(q2)
# # Probability of Alarm given Mary called
# q3 = alarm_infer.query(variables=["Alarm"], evidence={"MaryCalls": "yes"})
# print(q3)


# ##
# # Probability that the battery is not working given that the car will not move
# q1 = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
# print(q1)
# # Probability that the car will not start given that the radio is not working
# q2 = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
# print(q2)
# # Probability of the radio working given that the battery is working and the car has gas
# q3 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
# print(q3)
# # Probability of ignition failing given that the car doesn't move and the car is out of gas
# q4 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
# print(q4)
# # Probability that the car starts given that the radio works and the car has gas
# q5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
# print(q5)