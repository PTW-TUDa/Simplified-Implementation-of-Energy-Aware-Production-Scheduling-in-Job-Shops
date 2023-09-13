Simplified Implementation of Energy-Aware Production Scheduling
==================================================================

Climate change presents a pressing global challenge, necessitating urgent actions to reduce greenhouse gas emissions 
and limit global warming. As a significant energy consumer, the industrial sector plays a crucial role in advancing 
sustainable practices and mitigating the impact of climate change. In this context, this thesis focuses on developing 
an implementation procedure and an energy-aware production scheduling system architecture to optimize production
schedules while considering production-related and energy-related objectives.

The research goal of this thesis is to simplify the implementation of energy-aware production scheduling systems in real
production systems. To achieve this, the thesis addresses three key research areas: (1) to find whether the absence of
standardized procedures and architectures hinders the implementation of energy-aware production scheduling systems for
job shops, (2) to propose a standardized and partially automated implementation procedure for energy-aware production
scheduling, and (3) to design an architecture supporting the implementation procedure.

The proposed implementation procedure includes a structured system configuration and deployment approach, ensuring
alignment with stakeholder requirements. It comprises three phases: discovery and planning, development and 
configuration, and testing and deployment. The energy-aware production scheduling system architecture implements a
cyber-physical production system with a virtual representation of the actual production system. The architecture
incorporates the Non-Dominated Sorting Genetic Algorithm-II optimization algorithm with a graph-based solution encoding
and the production system environment, which adapts to specific production system requirements. An energy model
parameter estimation module supports the automatic configuration of production machine energy models.

Evaluation of the proposed concepts in the ETA Research Factory demonstrates the system's success in reducing energy
consumption while maintaining production-related objectives. The energy-aware production scheduling system achieves
average energy cost savings of 13 % and 18 % compared to traditional Shortest Processing Time dispatching rules while
slightly improving or marginally decreasing production-related performance, respectively.

This thesis contributes to the field of energy-aware production scheduling by providing an implementation procedure and
an adaptable architecture that fulfills the set requirements and success criteria. The proposed concepts offer practical
solutions for adopting energy-aware production scheduling systems in industrial settings, promoting environmentally
conscious and economically viable production practices. The thesis also identifies areas for improvement and future
research, ensuring the continuous development of energy-efficient and sustainable manufacturing processes.

**Keywords:** Demand Response, Energy-Efficiency, Energy-Flexibility, Cyber-Physical Production System, 
          Production Machine Energy Model, Implementation Procedure, Scheduling System Architecture


Installation
------------------------

**This package is tested with Julia 1.9.2 and Python 3.9.13**

To install and use this package you need to have `Julia <https://julialang.org/downloads/>`_ installed. Once this is 
done, the package can be activated using the Julia Package manager. To do this, open ``julia`` in in a console, then 
type ``]`` to open the package manager. The console should show the ``pkg>`` prompt. Now instantiate the package using

.. code-block::

    pkg> activate .
    pkg> instantiate

This should install all required dependencies with the correct versions. Sometimes this process is prone to failure -- 
in that case you have to install all dependencies manually (see Project.toml for this). 

Since this package interacts with the python library eta_utility, you should afterwards install that and ensure that 
Julia's PyCall is linked to the correct interpreter. To do this, return to the normal terminal prompt and execute the 
following commands.

As noted in the eta_utility (versio n2.2.4b2) documentation, this version is only compatible with specific versions of 
pip and setuptools due to limitations in the gym and stable_baselines3 dependencies. Therefore, make sure to install 
the correct versions first.

.. code-block::

    $> python -m pip install setuptools==65.5 pip==21
    $> pip install -r requirements.txt
    $> install-julia

The ``install-julia`` command is provided by eta_utility to ensure that PyCall is built correctly. After the
commands complete, you can use this package package by calling one of the starting scripts.

Usage
-----------

After completing the installation process, you can start either of the activation scripts. The ``scheduling.py`` files
in the ``trials`` folders start the production scheduling optimization and the ``TrainModels.jl`` file in the same 
folder starts the energy model parameter estimation.

Citing this project
--------------------

Please cite this project using the publication:

.. code-block::

    Grosch, B. (2023) Simplified Implementation of Energy-Aware Production Scheduling.
    Technische Universität Darmstadt. Dissertation. Publication in Review.
