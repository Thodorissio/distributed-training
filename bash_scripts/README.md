We have 3 scripts one for each vm and the highest level script that trigers them from the master vm (the run_parallel.sh)
the scripts must be executed simultaneously on all 3 vms .This is achieved by the run_parallel.sh functionality.

the commands are : 
chmod +x script_name.sh 
./script_name.sh  option1 option2


option1 -> name of model

option1 = 

          cifar_bouras

          cifar_teo
         
          mnist
          
          fashion_mnist
          
          imdb
          
          densenet
          
option2 -> number of nodes         

option2 = 

          1    (the model will be trained only on master vm meaning vm 2)

          2    (the model will be trained  on master vm meaning vm 2 and on vm 1)
          
          3    (the model will be trained on all 3 vms)

THE run_parallel.sh SCRIPT IS THE ONLY ONE YOU ACTUALLY NEED.



You run it on master vm with the same options as above:

./run_parallel.sh option1 option2      ,  while you are in the bash_scripts directory on master vm.

and it connects to the other vms and distributes the training , trigering the 2 other scripts (scrpt_1 and script_3) if needed.
      
      
