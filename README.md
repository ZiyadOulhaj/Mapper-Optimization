
# Differentiable Mapper for Topological Optimization of Data Representation 


## mapper.py

Contains an implementation of the regular Mapper Algorithm, with support for .html visualization using the Pyvis Python library.

## SoftMapper.py

Implements the smooth cover assignment scheme and contains code for contructing several Mappers in parallel efficiently and for computing persistence diagrams for Mapper graphs. See the notebooks for examples of how to use the code.

## human.off - octopus.off - table.off

Datasets for the 3-dimensional shapes.

## seurat_normalized.csv (found [here](https://drive.google.com/file/d/1e6_zN74-2mo4pMm-udBDQOsHJKHj1Q-5/view?usp=sharing))

Normalized expression matrix for the human preimplantation dataset.

## 3d_shapes_.ipynb

Notebook that implements filter optimization for 3-dimensional shapes.

## Single_cell_sctda.ipynb

Notebook that implements filter optimization for the human preimplantation dataset. To run the notebook, the dataset (found [here](https://www.dropbox.com/s/ma80a641miteyxf/scTDA%20Tutorial.tar.gz?dl=0) and [here](https://drive.google.com/file/d/1e6_zN74-2mo4pMm-udBDQOsHJKHj1Q-5/view?usp=sharing)) must first be downloaded and put in the Datasets folder.

## Single_cell_ot.ipynb

Notebook that implements filter optimization for the MEF reprogramming dataset. To run the notebook, the dataset (found [here](https://drive.google.com/open?id=1E494DhIx5RLy0qv_6eWa9426Bfmq28po)) must first be downloaded and put in the Datasets folder.