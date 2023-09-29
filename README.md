# VECOTRA 
<p align="center">
<img src ="./images/repo_logo.png"><br>
</p>

# VECOTRA : VElocity COnstrained TRAnsport

Python implementation of the constrained dynamics described in:
 - [1] Ibrahim, A.A.; Muehlebach, M.; De Bacco, C. (2023) *[Optimal transport with constraints: from mirror descent to classical mechanics](https://arxiv.org/abs/2309.04727)*. arXiv preprint arXiv:2309.04727.

If you use this code please cite [1].   

Copyright (c) 2023 [Abdullahi Adinoyi Ibrahim](https://github.com/aadinoyiibrahim) and [Caterina De Bacco](http://cdebacco.com).

# How to use this code

### Requirements

All the dependencies needed to run the algorithms can be installed using ```setup.py```, in a pre-built ```conda``` environment. <br/>
In particular, this script needs to be executed as:

```bash
python setup.py
```

You are ready to use our code!

### Usage example

Example usage is inside the notebook `example.ipynb `.  
The main code implementing the dynamics is `dyn.py`.

In the example.ipynb file, we show the Unconstrained Vs. Contrained Vs. Clip
 - Unconstrained: is the unconstrained method 
 - Contrained : is the Edge Capacity contraint of the method in [1]
 - Clip: a clipped version

An example of topology of a synthetic network showing flows of all O-D pairs is depicted in
<p align="center">
<img src ="./images/topology.png"><br>
</p>

As a last step, the distribution of variable $\mu$ before and after the edge capacity is implemented for the above topologies:
<p align="center">
<img src ="./images/mu_behavior.png"><br>
</p>

## License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
