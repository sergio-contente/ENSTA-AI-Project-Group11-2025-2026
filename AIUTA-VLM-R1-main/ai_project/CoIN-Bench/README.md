---
pretty_name: Collaborative Instance Object Navigation
size_categories:
- 1K<n<10K
---

# Collaborative Instance Object Navigation - CoIN-Bench dataset (v1) - ICCV 25

<!-- Provide a quick summary of the dataset. -->

We introduce Collaborative Instance object Navigation (CoIN), a new task setting where the agent actively resolve uncertainties about the target instance during navigation in
natural, template-free, open-ended dialogues with human.

To download the dataset, just run the following command:
```bash
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ftaioli/CoIN-Bench", repo_type="dataset", local_dir="CoIN-Bench")
```

## Dataset Details
Please see our ICCV 25 accepted paper: [```Collaborative Instance Object Navigation: Leveraging Uncertainty-Awareness to Minimize Human-Agent Dialogues.```](https://arxiv.org/abs/2412.01250)

For more information, visit our [Github repo.](https://github.com/intelligolabs/CoIN)


<!-- ## Uses -->

<!-- Address questions around how the dataset is intended to be used. -->


## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->
This repository contains the CoIN-Bench dataset, which is structured as follows:
- ```val_unseen```: Contains only novel objects not present in the training set.
- ```val_seen```:  Includes objects that also appear in the training set.
- ```val_seen_synonyms```: Contains objects from the training set but with synonymous names.

<!-- [More Information Needed] -->

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->



- **Curated by:** [Francesco Taioli](https://francescotaioli.github.io/) and Edoardo Zorzi.

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

```
@misc{taioli2025collaborativeinstanceobjectnavigation,
      title={Collaborative Instance Object Navigation: Leveraging Uncertainty-Awareness to Minimize Human-Agent Dialogues}, 
      author={Francesco Taioli and Edoardo Zorzi and Gianni Franchi and Alberto Castellini and Alessandro Farinelli and Marco Cristani and Yiming Wang},
      year={2025},
      eprint={2412.01250},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.01250}, 
}
```
