
      1.  For  making the MLP codes,  you should have done the following
          
          a) download AMD-APP-SDK-v2.9-lnx64.tgz and clAmdBlas-1.10.321.tar.gz from the AMD developer website
          b) Install AMD APP SDK under thte /opt directory of your Linux 
          c) Install the clAmdBlas package under the /opt/directory of your Linux and make the linking 
              #> cd /opt;  ln -s clAmdBlas-1.10.321 clAmdBlas 

          d) Under your CodeBlocks project environment, set your compiler and linker options to point to the 
             AMD APP SDK and clAmdBlas directory
 

      2.  For running the MLP testing provided by the testMLP project,  you first should prepare a directory architecture
          on your Linux as follows

        work/                                                       ; assume it is the root of your working directory
            MNIST/                                                  ; the directory to save the MNIST original dataset
            MNIST2/                                                 ; (optional) the pre-processed MNIST dataset used by me
            MNIST3/                                                 ; (optional) another pre-processed MNIST dataset used by me
            MLP-Test/test/                                          ; Copy all binary and runtime stuff here
                         kernels.cl                                ; kernel file need for running the OpenCL codes
                         testMLP                                   ; Binary of produced testMLP project, just copied to here
                         libMLP.so                                 ; Dynamic library produced by libMLP project, copied to here
                         clAmdBlas.so                              ; Copied to here from /opt/clAmdBlas/lib64 if doesn't set the
                                                                         LD_LIBRARY_PATH to include /opt/clAmdBlas/lib64/
                         mlp_training_init.conf                    ; (optional) MLP training configuration if you want to use pretrained weights
                         mlp_nnet_init.dat                         ; (optional) pretrained neural network weights 


       Then,  do  
        #> export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH       
        #> ./testMLP 

       to run the testing. 

       Modify the codes in mnist_test.cpp of the testMLP project to adapt the testing codes to your environment; If you want to
       try different values of MLP training parameters, you also need modify the codes in mnist_test.cpp.  

       Just check the codes !


       
                                                   
