# Block Distributed Majorize-Minimize Memory Gradient Algorithm

This repository contains the Python implementation of the *Block Distributed Majorize-Minimize Memory Gradient Algorithm* applied to the problem of 3D images restoration potentially handling a depth-variant blur. 

![Deblurring](/images/logo.png)

### Prerequisites and installation
These instructions will get you a copy of the BD3MG algorithm running on your multi-processor local or remote Unix machine. Note that BD3MG is based on the python multiprocessing library in order to handle process distribution. If you want to get the code working on other distributions (Windows and MacOs), you might need to change process afinity handling in the code.

This version of BD3MG runs on Python (>=3.5) with common libraries listed in the *requirements.txt* file.

### Installing

In order to run the different functions in the BD3MG package, follow the steps below on your command line: 

```
git clone https://github.com/mathieuchal/BD3MG/.git
cd BD3MG
pip install -r requirements.txt
```

## Testing

To ensure that the algorithm functions well on your machine, you can try the synthetic deblurring and denosing problem proposed in the package by entering

```
python fhzdg ghodhifgfgnjdgh,;fhj;
```

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

## Authors

* **Mathieu Chalvidal** - *e-mail*: mathieu.chalvidal@cnrs.fr - PhD Student ANITI 
* **Emilie Chouzenoux** - [website](http://www-syscom.univ-mlv.fr/~chouzeno/)
