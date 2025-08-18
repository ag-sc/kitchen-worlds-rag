# Kitchen Worlds with Retrieval-Augmented Generation

This project is a fork and an extension of the original [Kitchen Worlds](https://github.com/Learning-and-Intelligent-Systems/kitchen-worlds) environment developed by the [The Learning & Intelligent Systems Group @ MIT](https://lis.csail.mit.edu/).
For more information on the simulated environment, please see their repository as well as their [project website](https://learning-and-intelligent-systems.github.io/kitchen-worlds/).

## Project Goal

This projects aims at investigating the possible adantages of providing TAMP-focused Large Language Models with additional, embodied commonsense knowledge through [Retrieval-Augemented Generation (RAG)](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html).
For this, it evaluates the following databases combined in the [RAG4Robots](https://github.com/ag-sc/RAG4Robots) repository:
- Recipes from Recipe1M+[^1]
- How-To Articles from a WikiHow corpus[^2]
- Transcript of two tutorial videos on *cutting fruits & vegetables*[^3][^4]
- Object-Location tuples from the [CommonSense Knowledge Graph (CSKG)](https://cskg.readthedocs.io/en/latest/)[^5]

## Installation

Please follow these installation instructions below which are roughly the same as the ones from the original repository.
As an alternative, there is a [dockerfile](./docker/Dockerfile) available.

1. Clone the repo along with the submodules. It may take a few minutes. 

```shell
git clone --recursive https://github.com/ag-sc/kitchen-worlds-rag.git
git submodule sync && git submodule update --init --recursive
```

2. Install dependencies. It may take a dozen minutes.

```shell
conda env create -f environment.yml
conda activate kitchen
## sudo apt-get install graphviz graphviz-dev  ## on Ubuntu
```

3. Build FastDownward, the task planner used by PDDLStream planner.

```shell
## sudo apt install cmake g++ git make python3  ## if not already installed
(cd pddlstream; ./downward/build.py)
```

4. Build IK solvers (If using mobile manipulators; skip this if you're only using the floating gripper).

* (If on Ubuntu, this one is better) TracIK for whole-body IK that solves for base, torso, and arm together

    ```shell
    sudo apt-get install libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev libnlopt-cxx-dev swig
    pip install git+https://github.com/mjd3/tracikpy.git
    ```

* IKFast solver for arm planning (the default IK), which needs to be compiled for each robot type. Here's example for PR2:

    ```shell
    ## sudo apt-get install python-dev
    (cd pybullet_planning/pybullet_tools/ikfast/pr2; python setup.py)
    ```

## References

Please cite the original *kitchen-world* environment by using the following papers in your research:

```text 
@inproceedings{yang2024guiding,
    title     = {{Guiding Long-Horizon Task and Motion Planning with Vision Language Models}},
    author    = {Yang, Zhutian and Garrett, Caelan and Fox, Dieter and Lozano-P{\'e}rez, Tom{\'a}s and Kaelbling, Leslie Pack},
    booktitle = {ICRA},
    year      = {2025},
    organization  = {IEEE}
}

@inproceedings{yang2023piginet, 
    author    = {Zhutian Yang AND Caelan R Garrett AND Tomas Lozano-Perez AND Leslie Kaelbling AND Dieter Fox}, 
    title     = {{Sequence-Based Plan Feasibility Prediction for Efficient Task and Motion Planning}}, 
    booktitle = {Proceedings of Robotics: Science and Systems}, 
    year      = {2023}, 
    address   = {Daegu, Republic of Korea}, 
    month     = {July}, 
    doi       = {10.15607/RSS.2023.XIX.061} 
} 
```

If you use this version of the repository and its RAG-based features, please cite the following research:

```text 
TBA
```

---

## Acknowledgements

In addition to the wonderful work provided by the creators of the original repository (Zhutian Yang, Tomas Lozano-Perez, Jiayuan Mao, Weiyu Liu), this project vendors [pybullet-planning](https://github.com/zt-yang/pybullet_planning) (MIT License, © 2019 Caelan Garrett & Zhutian Yang).
See pybullet_planning/LICENSE for details.

Additionally, we acknowledge the use of the following data sources for our RAG system:

[^1]: J. Marín et al., ‘Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images’, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 1, pp. 187–203, Jan. 2021, doi: 10.1109/TPAMI.2019.2927476.

[^2]: L. Zhang, Q. Lyu, and C. Callison-Burch, ‘Reasoning about Goals, Steps, and Temporal Ordering with WikiHow’, in Proc. of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Online: Association for Computational Linguistics, 2020, pp. 4630–4639. doi: 10.18653/v1/2020.emnlp-main.374.

[^3]: Epicurious, YouTube. How To Chop Every Vegetable | Method Mastery | Epicurious, (Jan. 31, 2020). [Online Video]. Available: https://youtu.be/p28wMbunulQ?si

[^4]: Epicurious, YouTube. How To Slice Every Fruit | Method Mastery | Epicurious, (Nov. 06, 2019). [Online Video]. Available: https://youtu.be/VjINuQX4hbM

[^5]: F. Ilievski, P. Szekely, and B. Zhang, ‘CSKG: The CommonSense Knowledge Graph’, in The Semantic Web, vol. 12731, R. Verborgh, K. Hose, H. Paulheim, P.-A. Champin, M. Maleshkova, O. Corcho, P. Ristoski, and M. Alam, Eds., Cham: Springer International Publishing, 2021, pp. 680–696. doi: 10.1007/978-3-030-77385-4_41.
