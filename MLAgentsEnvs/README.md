# hw21-ml-agents-julia
# Converting to public repository
Any and all Unity software of any description (including components) (1) whose source is to be made available other than under a Unity source code license or (2) in respect of which a public announcement is to be made concerning its inner workings, may be licensed and released only upon the prior approval of Legal.
The process for that is to access, complete, and submit this [FORM](https://docs.google.com/forms/d/e/1FAIpQLSe3H6PARLPIkWVjdB_zMvuIuIVtrqNiGlEt1yshkMCmCMirvA/viewform).


install julia repo as 
```julia
]develop git@github.com:Unity-Technologies/hw21-ml-agents-julia.git
```

to run experiments add the following as dependencies
```julia
add BenchmarkTools Distributions PyCall Flux Formatting
add git@github.com:DecisionMakingAI/DecisionMakingEnvironments.jl.git
add git@github.com:DecisionMakingAI/DecisionMakingPolicies.jl.git
add git@github.com:DecisionMakingAI/DecisionMakingUtils.jl.git
```

The last three packages are open source repositories to help with RL problems. 