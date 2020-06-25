<img src="./docs/images/rl_unplugged_tasks_v0.png" width="50%">

# RL Unplugged: Benchmarks for Offline Reinforcement Learning

RL Unplugged is suite of benchmarks for offline reinforcement learning. The RL
Unplugged is designed around the following considerations: to facilitate ease of
use, we provide the datasets with a unified API which makes it easy for the
practitioner to work with all data in the suite once a general pipeline has been
established. This is a dataset accompanying the paper
[RL Unplugged: Benchmarks for Offline Reinforcement Learning]([https://arxiv.org/abs/2006.13888]).

In this suite of benchmarks, we try to focus on the following problems:

-   High dimensional action spaces, for example the locomotion humanoid domains,
    we have 56 dimensional actions.

-   High dimensional observations.

-   Partial observability, observations have egocentric vision.

-   Difficulty of exploration, using states of the art algorithms and imitation
    to generate data for difficult environments.

-   Real world challenges.

The data is available under
[RL Unplugged GCP bucket](https://console.cloud.google.com/storage/browser/rl_unplugged).

Data loading code and examples will be available soon.

## Atari Dataset

We are releasing a large and diverse dataset of gameplay following the protocol
described by Agarwal et al. (2020), which can be used to evaluate several
discrete offline RL algorithms. The dataset is generated by running an online
DQN agent and recording transitions from its replay during training with sticky
actions (Machado et al., 2018). As stated in (Agarwal et al.,2020), for each
game we use data from five runs with 50 million transitions each. States in each
transition include stacks of four frames to be able to do frame-stacking with
our baselines. We release datasets for 46 Atari games. For details on how the
dataset was generated, please refer to the paper.

## Deepmind Locomotion Dataset

These tasks are made up of the corridor locomotion tasks involving the CMU
Humanoid, for which prior efforts have either used motion capture data (Merel et
al., 2019a,b) or training from scratch (Song et al., 2020). In addition, the DM
Locomotion repository contains a set of tasks adapted to be suited to a virtual
rodent (see Merel et al., 2020). We emphasize that the DM Locomotion tasks
feature the combination of challenging high-DoF continuous control along with
perception from rich egocentric observations. For details on how the dataset was
generated, please refer to the paper.

## Deepmind Control Suite Dataset

DeepMind Control Suite (Tassa et al., 2018) is a set of control tasks
implemented in MuJoCo (Todorov et al., 2012). We consider a subset of the tasks
provided in the suite that cover a wide range of difficulties.

Most of the datasets in this domain are generated using D4PG. For the
environments Manipulator insert ball and Manipulator insert peg we use V-MPO
(Song et al., 2020) to generate the data as D4PG is unable to solve these tasks.
We release datasets for 9 control suite tasks. For details on how the dataset
was generated, please refer to the paper.

## Realworld RL Dataset

Examples in the dataset represent SARS transitions stored when running a
partially online trained agent as described in
[RWRL](https://arxiv.org/abs/1904.12901).

We release 8 datasets in total -- with no combined challenge and easy combined
challenge on the cartpole, walker, quadruped, and humanoid tasks. For details on
how the dataset was generated, please refer to the paper.

## Citation

Please use the following bibtex for citations:

```
@misc{gulcehre2020rl,
    title={RL Unplugged: Benchmarks for Offline Reinforcement Learning},
    author={Caglar Gulcehre and Ziyu Wang and Alexander Novikov and Tom Le Paine
        and  Sergio Gómez Colmenarejo and Konrad Zolna and Rishabh Agarwal and
        Josh Merel and Daniel Mankowitz and Cosmin Paduraru and Gabriel
        Dulac-Arnold and Jerry Li and Mohammad Norouzi and Matt Hoffman and
        Ofir Nachum and George Tucker and Nicolas Heess and Nando deFreitas},
    year={2020},
    eprint={2006.13888},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

# Dataset Metadata

The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">RL Unplugged</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">
      Data accompanying
[RL Unplugged: Benchmarks for Offline Reinforcement Learning]().
      </code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">DeepMind</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/DeepMind</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">https://identifiers.org/arxiv:2006.13888</code></td>
  </tr>
</table>
</div>