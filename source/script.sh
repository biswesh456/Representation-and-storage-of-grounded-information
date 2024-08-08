
# 4 GPUs 
#oarsub -p "gpu='YES' and gpumem>30000 and (host='nefgpu52.inria.fr' or host='nefgpu53.inria.fr' or host='nefgpu54.inria.fr' or host='nefgpu55.inria.fr' or host='nefgpu56.inria.fr' or host='nefgpu57.inria.fr' or host='nefgpu58.inria.fr' or host='nefgpu59.inria.fr' or host='nefgpu60.inria.fr' or host='nefgpu61.inria.fr')" -l /gpunum=4,walltime=00:20:00  -t besteffort ./python_exec

#oarsub -p "gpu='YES' and gpumem>60000 and gpucapability>='8.0'" -l /nodes=1/gpunum=2,walltime=00:20:00  -t besteffort --stdout=../data/OARoutput/OAR.%jobid%.stdout --stderr=../data/OARoutput/OAR.%jobid%.stderr ./python_exec


# 1 GPU
#oarsub -p "gpu='YES' and gpumem>30000 and gpucapability>='8.0'" -l /nodes=1/gpunum=1,walltime=00:3:00  -t besteffort --stdout=../data/OARoutput/OAR.%jobid%.stdout --stderr=../data/OARoutput/OAR.%jobid%.stderr ./python_exec

# 1 GPU interactive for dev
oarsub -p "gpu='YES' and gpumem>35000 and gpucapability>='7.0'" -l /nodes=1/gpunum=1,walltime=01:00:00  -t besteffort --stdout=../data/OARoutput/OAR.%jobid%.stdout --stderr=../data/OARoutput/OAR.%jobid%.stderr -I ./python_exec
