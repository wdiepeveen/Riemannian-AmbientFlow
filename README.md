# Riemannian AmbientFlow

    [1] W. Diepeveen, O. Leong.  
    Riemannian AmbientFlow: Towards Simultaneous Manifold Learning and Generative Modeling from Corrupted Data.
    arXiv preprint arXiv:[xxxx].[yyyyy]. 2026 MM DD.

Setup
-----

The recommended (and tested) setup is based on Python 3.13. Install the following dependencies with anaconda:

    # Create conda environment
    conda create --name raf python=3.13
    conda activate raf

    # Clone source code and install
    git clone https://github.com/wdiepeveen/Riemannian-AmbientFlow.git
    cd "Riemannian-AmbientFlow"
    pip install -r requirements.txt


Reproducing the experiments in [1]
----------------------------------

To produce the results in [1]. 
* For the sinusoid experiments run:
  *  `sinusoid_riemannian_ambient_flow.ipynb`
* For mnist experiments run:
  *  `mnist_deblur_riemannian_ambient_flow_recon.ipynb`
