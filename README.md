# Decision_tree-code
Our protocol utilizes the SPDZ framework, and thus, we recommend readers to refer to the deployment of SPDZ to understand the compilation and execution instructions for different secret sharing schemes.(https://github.com/data61/MP-SPDZ.git)

For our regression tree training protocol, denoted as "Regression_Tree_Training.mpc," the training data is "Regression_Tree_data.txt." Prior to training, we input the training data into "/Player-Data/Input-P0-0." During the protocol implementation, the first step involves compiling the protocol files. Subsequently, the protocol files can be executed using the secret sharing scheme provided by the framework. Below, we provide the compilation and simulation execution steps for the protocol files using the three-party replicated secret sharing scheme.
Compile:    ./compile.py -R 64 Regression_Tree_Training.mpc
Run:     Scripts/ring.sh Regression_Tree_Training

Regarding our classification tree training protocol, denoted as "Classification_Tree_Trainin.mpc," the training data is "Classification_Tree_data.txt." Similar to the regression tree, we input the training data into the "/Player-Data/Input-P1-0" before training. Implementing the classification tree training protocol follows a similar process, wherein we compile the files first and then select a suitable secret sharing scheme for execution. Below, we outline the compilation and simulation execution steps for the protocol files using the three-party replicated secret sharing scheme.
Compile:    ./compile.py -R 64 Classification_Tree_Trainin.mpc
Run:    Scripts/ring.sh Classification_Tree_Trainin
