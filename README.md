# Representation-and-storage-of-grounded-information



[![Paper](https://img.shields.io/badge/arXiv-2601.09365-B31B1B.svg)](https://arxiv.org/abs/2601.09365)

This repository provides the implementations for the **finetuning** pipelines of
**[Frame of Reference: Addressing the Challenges of Common Ground Representation in Situational Dialogs](https://arxiv.org/abs/2601.09365)**.

---

## How to run 

In this repo we explore context modifications to improve Common Ground representations in LLMs.

---

For the finetuning environment:

```bash
git clone https://github.com/biswesh456/Representation-and-storage-of-grounded-information

cd Representation-and-storage-of-grounded-information

# Create and activate the finetuning environment
conda env create -n grounded-info -f environment.yml

conda activate grounded-info
```

And run the train and eval scripts :

```bash
python source/main.py

python source/eval.py
```

---

## 📖 Citation

To cite this work, please use the following bibtex.

```bibtex
@misc{mohapatra2026framereferenceaddressingchallenges,
      title={Frame of Reference: Addressing the Challenges of Common Ground Representation in Situational Dialogs}, 
      author={Biswesh Mohapatra and Théo Charlot and Giovanni Duca and Mayank Palan and Laurent Romary and Justine Cassell},
      year={2026},
      eprint={2601.09365},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.09365}, 
}
```

---
