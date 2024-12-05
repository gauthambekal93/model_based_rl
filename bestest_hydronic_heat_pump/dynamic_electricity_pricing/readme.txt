sExperiment_V1:
This experiment only uses policy agent on dynamic pricing scenario

Experiment_V2:
This experiment uses policy agent and model-based-RL on dynamic pricing scenario

Experiment_V3:
This experiment only uses policy agent on constant pricing scenario

Experiment_V4:
This experiment only uses policy agent with sparse rewards on constant pricing scenario. Here the sparse reward is every 50th time step.

Experiment_V5:
This experiment only uses policy agent with sparse rewards on constant pricing scenario. Here the sparse reward is every 20th time step

Experiment_V6:
This experiment uses policy agent and the learnt env model from dynamic pricing scenario for sparse rewards on constant pricing scenario. Here the sparse reward is every 20th time step

Experiment_V7:
This experiment uses policy agent and the learnt env model from dynamic pricing scenario for sparse rewards on constant pricing scenario. Here the sparse reward is every 50th time step
The learning rate is reduced to 0.001 # the 0.0001 was not the correct learning rate most probably

Experiment_V8:
This experiment uses policy agent from dynamic pricing scenario for sparse rewards on constant pricing scenario. Here the sparse reward is every 50th time step
The learning rate is reduced to 0.001


Experiment_V9:
This experiment uses policy agent from dynamic pricing scenario for sparse rewards on constant pricing scenario. Here the sparse reward is every 50th time step
The learning rate is 0.0001
Test and train episode length is of 14 days


 
Experiment_V10:
This experiment uses policy agent and the learnt env model from dynamic pricing scenario for sparse rewards on constant pricing scenario. Here the sparse reward is every 50th time step
The learning rate is reduced to 0.001
Test and train episode length is of 14 days

 
Experiment_V11:
This experiment uses policy agent from dynamic pricing scenario for sparse rewards on constant pricing scenario. Here the sparse reward is every 50th time step
The learning rate is reduced to 0.001
Test and train episode length is of 14 days

 
Experiment_V12:
This experiment uses policy agent for sparse rewards on constant pricing scenario. Here the sparse reward is every 50th time step
The learning rate is reduced to 0.001
Test and train episode length is of 14 days

 
Experiment_V13:
This experiment uses policy agent for sparse rewards with dynamic pricing on Bestest_Hydronic usecase. Here the sparse reward is every 50th time step
The learning rate is reduced to 0.001
Test and train episode length is of 14 days


Experiment_V14:
This experiment uses policy agent which was pretrained on Bestest_Hydronic_heat_Pump with dynamic pricing. 
It was then fine tuned on sparse rewards with dynamic pricing on Bestest_Hydronic usecase. Here the sparse reward is every 50th time step
The learning rate is reduced to 0.001
Test and train episode length is of 14 days


Experiment_V15:
This experiment uses policy agent  and the learnt env model  which was pretrained on Bestest_Hydronic_heat_Pump with dynamic pricing. 
It was then fine tuned on sparse rewards with dynamic pricing on Bestest_Hydronic usecase. Here the sparse reward is every 50th time step
The learning rate is reduced to 0.001
Test and train episode length is of 14 days


Experiment_V16:
This experiment uses actor-critic network with 10 action dimensions.
The experiment uses random dates for train episodes and tested on 4 test periods (Jan 17. April 19, Nov 15, Dec 08 all for 2024)
Episode length is of 7 days
Learning rate of actor is 0.0001
Learning rate of critic is 0.001
Ran for 300 epochs


Experiment_V17:
This experiment uses actor-critic network with 10 action dimensions.
The experiment uses only March 01 for train episodes and tested on 4 test periods (Jan 17. April 19, Nov 15, Dec 08 all for 2024)
Episode length is of 7 days
Learning rate of actor is 0.001
Learning rate of critic is 0.005
Ran for 500 epochs


Experiment_V18:
Meta learning approach for model based reinforcement learning.
We did not go ahead with this approach for now, reason being I think week over week data does not vary much from the env_model perspective. The reason being we are only predicting one step ahead.


Experiment_V19:
Uses VAE as env model, but the implementation is INCORRECT !!! since when the trajectory gets switched from env model to actual model it will continue trajectory from where actual env left and not at the point where env model left.


Experiment_V20:
Only used to recreate the error which we described in Experiment_V19

Experiment_V21:
We SAC based RL agent.
The pseudo code was based on (https://spinningup.openai.com/en/latest/algorithms/sac.html) without any env model.

