[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=mathworks/Implement-Incremental-Learning-on-Simulink-)
# Implement Incremental Learning on Simulink® by applying System Object™

This page introduces how to implement incremental learning on Simulink concretely, divided into 2 use cases (classification and regression respectively).

Copyright (c) 2022, The MathWorks, Inc.

# Introduction

Incremental learning (also known as online learning) enables to create an updatable learner differing from traditional static pre-trained learners because this capability improves the learner using newly obtained observation data. Therefore incremental learning is pretty robust when we address a problem that it's impossible to acquire sufficient dataset for training in advance. In addition, incremental learning can be apply not only for static dataset appearing on this page but for dynamical dataset from various devices such as online remote sensors, web camera, packet data on network and so on. Please see [this page](https://www.mathworks.com/help/stats/incremental-learning-overview.html) if you are interested in the details of incremental learning.

# Key Takeaways

Though incremental learning provided by Statistics and Machine Learning Toolbox™ is focused definitely on this page, System Object is also remarkable feature in case of implementing this capability on Simulink. This is one of MATLAB® class and is designed specifically for implementing and simulating dynamic systems with inputs that change over time. So this System Object has good chemistry with processing a certain amount of data chunk recursively like streaming data processing. Please see the following page in the details of System Object.

https://www.mathworks.com/help/matlab/matlab_prog/what-are-system-objects.html

# Requirements

* MATLAB R2022a
* Simulink
* DSP System Toolbox™
* Signal Processing Toolbox™
* Statistics and Machine Learning Toolbox

# Remarks

If you want to do a same approach using deep learning as this content, it's possible because Deep Learning Toolbox provides the Stateful Classify/Predict block. These blocks enable you to implement an updatable learner on Simulink without System Object.
