#printlnusing Images
using PyPlot
using FileIO
using Optim
using Random
using Statistics

include("Common.jl")

#---------------------------------------------------------
# Load features and labels from file.
#---------------------------------------------------------
function loaddata(path::String)
  data = load(path)
  features = data["features"]
  labels = data["labels"]
  @assert length(labels) == size(features,1)
  return features::Array{Float64,2}, labels::Array{Float64,1}
end

#---------------------------------------------------------
# Show a 2-dimensional plot for the given features with
# different colors according to the labels.
#---------------------------------------------------------
function showbefore(features::Array{Float64,2},labels::Array{Float64,1})

  green = features[findall(labels.<1.0),:]
  blue = features[findall(labels.>=1.0),:]

  PyPlot.figure()
  PyPlot.scatter(green[:,1],green[:,2],c="green",label="0")
  PyPlot.scatter(blue[:,1],blue[:,2],c="blue",label="1")
  PyPlot.legend()

  return nothing::Nothing
end


#---------------------------------------------------------
# Show a 2-dimensional plot for the given features along
# with the decision boundary.
#---------------------------------------------------------
function showafter(features::Array{Float64,2},labels::Array{Float64,1},Ws::Vector{Any}, bs::Vector{Any})

  PyPlot.figure()
  x1 = [j for j=-2:0.1:2.5 ]
  x1 = reshape(x1, 1, size(x1,1))
  y1 = [j for j=-2.5:0.1:3 ]

  xx = zeros(size(y1,1), size(x1,2))
  for j=1:size(y1,1)
    xx[j,:] = x1[:]
  end
  yy = zeros(size(y1,1), size(x1,2))
  for j=1:size(x1,2)
    yy[:,j] = y1
  end

  Z, c = predict(hcat(xx[:], yy[:]) , Ws, bs)
  Z = reshape(Z[:,1,1], size(xx,1), size(xx,2))
  contourf(xx,yy,Z)

  p,c = predict(features,Ws,bs)
  green = features[findall(c[:,1].<1.0),:]
  blue = features[findall(c[:,1].>=1.0),:]

  PyPlot.scatter(green[:,1],green[:,2],c="green",label="0")
  PyPlot.scatter(blue[:,1],blue[:,2],c="blue",label="1")
  PyPlot.legend()

  return nothing::Nothing
end


#---------------------------------------------------------
# Implements the sigmoid function.
#---------------------------------------------------------
function sigmoid(z)
  s = 1 ./ (1 .+ exp.(-z))
  return s
end


#---------------------------------------------------------
# Implements the derivative of the sigmoid function.
#---------------------------------------------------------
function dsigmoid_dz(z)
  ds = sigmoid(z) .* (1 .- sigmoid(z))
  # ds = -exp.(-z) .* (1 .+ exp.(-z)) .^ (-2)
  return ds
end


#---------------------------------------------------------
# Evaluates the loss function of the MLP.
#---------------------------------------------------------
function nnloss(theta::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}, netdefinition::Array{Int, 1})
  Ws, bs = thetaToWeights(theta, netdefinition)
  ### Ws in Form: Ws[numberOfLayer][NeuronOfLayer, NeuronOfLastLayer]
  p = zeros(size(y,1))
  loss = 0

  for i=1:size(X,1)
    z = X[i,:]
    p_i, c, b = forwardPass(size(netdefinition,1)-1, theta, netdefinition, z)

    p[i] = sum(p_i)
    loss += y[i] * log(p[i]) + (1-y[i])*log(1-p[i])
  end
  loss = -loss / size(y,1)
  display(loss)
  # nnlossgrad([1.0], theta, X, y, netdefinition)
  return loss::Float64
end

function forwardPass(level, theta, netdefinition, x)
  Ws, bs = thetaToWeights(theta, netdefinition)
  z = x
  a = 0
  for k=1:level #iterate k-layers
    a = Ws[k]*z .+ bs[k]
    z = sigmoid(a)
    if k==(level-1)
      x = z
    end
  end
  y = z
  z = a
  x = vcat(x, 1)
  return y, z, x #return output and last neuron and last inputs
end


#---------------------------------------------------------
# Evaluate the gradient of the MLP loss w.r.t. Ws and Bs
# The gradient should be stored in the vector 'storage'
#---------------------------------------------------------
function nnlossgrad(storage::Array{Float64,1}, theta::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}, netdefinition::Array{Int, 1})
  # if size(storage,1) != size(theta,1)
  #   return storage
  # else
  #
  #   l
  #   for k=size(netdefinition):-1:2
  #     if size(storage,1) ==
  #   end
  #
  #   #Neue Grad. von W berechnen
  #   for i=1:size(y,1)
  #     p, z, x = forwardPass(size(netdefinition,1)-1, theta, netdefinition, X[i,:])
  #     de = -2*y[i]*p + y[i] + p / ( (1-p)*p )
  #     newGrads = de .* dsigmoid_dz(z) .* x
  #   end
  #   storage = vcat(storage, newGrads)
  #
  #   storage = nnlossgrad(storage, theta, X, y, netdefinition)
  #
  # end
  # display(sum(storage))
  sumstorage = zeros(size(theta,1))
  pltstr = zeros(size(theta,1),size(y,1))
  Ws, bs = thetaToWeights(theta, netdefinition)

  Ws[end] = reshape(Ws[end], size(Ws[end],2), size(Ws[end],1))
  for i=1:1#size(y,1)

    #First Layer
    p, z, x = forwardPass(size(netdefinition,1)-1, theta, netdefinition, X[i,:])
    p = p[1]

    delta = -(-2*y[i]*p + y[i] + p) / ( (1-p)*p )   # * dsigmoid_dz(z[1])
    storageW = ( delta .* dsigmoid_dz(z) .* x )[1:end-1]
    storageB = ( delta .* dsigmoid_dz(z) .* x )[end]

    #all other Layers
    for r=size(netdefinition,1)-2:-1:1
      p, z, x = forwardPass(r, theta, netdefinition, X[i,:])
      p_1,z_1 ,x_1 = forwardPass(r+1, theta, netdefinition, X[i,:])
      delta_new = zeros(netdefinition[r+1])

      for l=1:netdefinition[r+1]
        for k=1:netdefinition[r+2]
          delta_new[l] += (delta[k] * dsigmoid_dz(z_1[k]) *  Ws[r+1][l, k] )
        end
      end
      delta = delta_new
      newStorage = (delta .* dsigmoid_dz(z) * x')

      newStorageW = newStorage[:,1:end-1]
      newStorageW = newStorageW[:]
      newStorageB = newStorage[:,end]
      storageW = vcat(newStorageW, storageW)
      storageB = vcat(newStorageB, storageB)
    end
    sumstorage += [storageW; storageB]
    # display(sumstorage)
    # pltstr[:,i] = [storageW; storageB]
  end
  # plot(sumstorage[1], "-o")
  # println(sumstorage[1])
  # display( 1/size(y,1).*sumstorage - storage )
  # figure()
  # plot(Float64.(pltstr'))
  # display(pltstr)
  # figure()
  # plot(sum(pltstr, dims=2)./size(y,1))
  storage = 1/size(y,1) .* sumstorage

  return storage::Array{Float64,1}
end


#---------------------------------------------------------
# Use LBFGS to optimize the MLP loss
#---------------------------------------------------------
function train(trainfeatures::Array{Float64,2}, trainlabels::Array{Float64,1}, netdefinition::Array{Int, 1})
  sigmaW = 0.01
  sigmaB = 0.001
  Ws, bs = initWeights(netdefinition, sigmaW, sigmaB)
  initTheta = weightsToTheta(Ws, bs)
  # storage = zeros(size(initTheta,1))
  # nlos = nnloss(initTheta, trainfeatures, trainlabels, netdefinition)
  # gradlos = nnlossgrad(storage, initTheta, trainfeatures, trainlabels, netdefinition)
  # println("nlos:", nlos)
  # println("gradlos:", gradlos)

  # figure()
  # theta_1 = initTheta
  # theta_0 = initTheta
  # n=1
  # h=0.001
  # nnlosss=zeros(1000)
  # gradlosss=zeros(1000)
  # while abs(nnloss(theta_0, trainfeatures, trainlabels, netdefinition)) >0.1 && n<300
  #   theta_1 = theta_0
  #   theta_0 = theta_1-(h*nnlossgrad(storage, theta_0, trainfeatures, trainlabels, netdefinition))
  #   # println("n: ", n)
  #   # println("grad: ", theta_0)
  #   gradlosss[n] = sum(theta_0)
  #   nnlosss[n] = nnloss(theta_0, trainfeatures, trainlabels, netdefinition)
  #   n+=1
  # end
  # figure()
  # plot(gradlosss)
  # figure()
  # plot(nnlosss)

  # figure()
  Ws, bs = initWeights(netdefinition, sigmaW, sigmaB)
  initTheta = weightsToTheta(Ws, bs)
  #storage = zeros(size(initTheta,1))

  function g!(storage,Theta)
    storage[:] = nnlossgrad(storage, Theta, trainfeatures, trainlabels, netdefinition)
  end

  function f(Theta)
    return nnloss(Theta, trainfeatures, trainlabels, netdefinition)
  end

  res = optimize(f, initTheta, LBFGS(), Optim.Options(iterations = 1000, g_tol = 1e-12))
  # res = optimize(Theta -> nnloss(Theta, trainfeatures, trainlabels, netdefinition), initTheta, LBFGS())

  # min = Optim.minimum(res)
  # while abs(Optim.minimum(res))>0.1 && n<100
  #   Ws, bs = initWeights(netdefinition, sigmaW, sigmaB)
  #   initTheta = weightsToTheta(Ws, bs)
  #   res = optimize(Theta -> nnloss(Theta, trainfeatures, trainlabels, netdefinition), initTheta, LBFGS(); inplace = false)
  #   n+=1
  #   if Optim.minimum(res)<min
  #     min = Optim.minimum(res)
  #   end
  #   display(min)
  # end
    # thetaOptim = optimize(nlos, gradlos, initTheta, LBFGS(); inplace = false)

  Optim.summary(res)
  println("")
  println("Minimizer")
  display(Optim.minimizer(res))
  println("")
  println("Minimum")
  display(min)
  minTheta = Optim.minimizer(res)
  Ws, bs = thetaToWeights(minTheta, netdefinition)
  println("")
  println("Weights")
  display(Ws)
  println("")
  println("Biases")
  display(bs)
  display(Optim.g_converged(res))
  return Ws::Vector{Any},bs::Vector{Any}
end


#---------------------------------------------------------
# Predict the classes of the given data points using Ws and Bs.
# p, N x 1 array of Array{Float,2}, contains the output class scores (continuous value) for each input feature.
# c, N x 1 array of Array{Float,2}, contains the output class label (either 0 or 1) for each input feature.
#---------------------------------------------------------
function predict(X::Array{Float64,2}, Ws::Vector{Any}, bs::Vector{Any})
  p =Float64[]

  for i=1:size(X,1)
    z = X[i,:]
    a = 0
    for k=1:size(Ws,1) #iterate k-layers
      a = Ws[k]*z .+ bs[k]
      z = sigmoid(a)
      if k==(size(Ws,1)-1)
        x = z
      end
    end
    y = z
    z = a
    x = vcat(x, 1)
    p=vcat(p,y)

  end

  c=round.(p)
  p= hcat(p,X)
  c= hcat(c,X)
  return p::Array{Float64,2}, c::Array{Float64,2}
end


#---------------------------------------------------------
# A helper function which concatenates weights and biases into a variable theta
#---------------------------------------------------------
function weightsToTheta(Ws::Vector{Any}, bs::Vector{Any})
  theta = []
  for i=1:size(Ws,1)
    theta = vcat(theta, Ws[i][:])
  end
  for i=1:size(bs,1)
    theta = vcat(theta, bs[i][:])
  end
  theta = Float64.(theta)
  # theta = Float64.(vcat(Vector(Ws),Vector(bs)))
  # theta = reshape(hcat(Ws[1],bs[1])',(length(Ws[1])+length(bs[1]),1))
  # for i=2:size(Ws,1)
  #   theta = vcat(theta,reshape(hcat(Ws[i],bs[i])' , (length(Ws[i])+length(bs[i]) , 1)))
  # end
  return theta::Vector{Float64}
end


#---------------------------------------------------------
# A helper function which decomposes and reshapes weights and biases from the variable theta
#---------------------------------------------------------
function thetaToWeights(theta::Vector{Float64}, netdefinition::Array{Int,1})
  # nWs=0
  # for i=1:size(netdefinition,1)-1
  #   nWs=nWs+netdefinition[i]*netdefinition[i+1]
  # end
  # Ws = Any[]
  # bs = Any[]
  # Ws = vcat(Ws,theta[1:nWs])

  Ws = Any[]
  bs = Any[]
  offset = 1
  for i=1:size(netdefinition,1)-1
    nLevel     = netdefinition[i]
    nNextLevel = netdefinition[i+1]
    push!(Ws, reshape(theta[offset:offset+nNextLevel*nLevel-1], nNextLevel, nLevel) )
    offset += nNextLevel*nLevel
  end
  for i=2:size(netdefinition,1)
    push!(bs, theta[offset:offset+netdefinition[i]-1])
    offset +=  netdefinition[i]
  end
  # bs = vcat(bs,theta[offset+1:end])

  return Ws::Vector{Any}, bs::Vector{Any}
end


#---------------------------------------------------------
# Initialize weights and biases from Gaussian distributions
#---------------------------------------------------------
function initWeights(netdefinition::Array{Int,1}, sigmaW::Float64, sigmaB::Float64)
  nWs=0   #################### Random Muss mit variabler gauss verteilung#############################
  ####################################################################################################
  # for i=1:size(netdefinition,1)-1
  #   nWs=nWs+netdefinition[i]*netdefinition[i+1]
  # end
  # Ws = Any[]

  Ws = Any[]
  bs = Any[]
  for i=1:size(netdefinition,1)-1
    push!(Ws, randn(netdefinition[i+1],netdefinition[i]).*sigmaW)
    push!(bs, randn(netdefinition[i+1]).*sigmaB)
  end

  # Ws = vcat(Ws,randn(nWs))
  # bs = vcat(bs,randn(sum(netdefinition[2:end])))

  # ## Ws indizes wie bei Fuzzy also Ws[schicht r][neuron j von schicht r, von neuron l von schicht r-1]
  # Ws = [ randn(netdefinition[i+1],netdefinition[i]) for i=1:size(netdefinition,1)-1]
  # bs = [ randn(netdefinition[i+1]) for i=1:size(netdefinition,1)-1]
  return Ws::Vector{Any}, bs::Vector{Any}
end


# Problem 2: Multilayer Perceptron

function problem2()
  # make results reproducable
  Random.seed!(10)

  # # LINEAR SEPARABLE DATA
  # # load data
  # features,labels = loaddata("separable.jld2")
  #
  # # show data points
  # showbefore(features,labels)
  # title("Data for Separable Case")
  #
  # # train MLP
  # Ws,bs = train(features,labels, [2,4,1])
  #
  # # # show optimum and plot decision boundary
  # showafter(features,labels,Ws,bs)
  # title("Learned Decision Boundary for Separable Case")


  # ## LINEAR NON-SEPARABLE DATA
  # # load data
  # features2,labels2 = loaddata("nonseparable.jld2")
  #
  # # show data points
  # showbefore(features2,labels2)
  # title("Data for Non-Separable Case")
  #
  # # train MLP
  # Ws,bs = train(features2,labels2, [2,4,1])
  #
  # # show optimum and plot decision boundary
  # showafter(features2,labels2,Ws, bs)
  # title("Learned Decision Boundary for Non-Separable Case")

  # PLANE-BIKE-CLASSIFICATION FROM PROBLEM 2
  # load data
  trainfeatures,trainlabels = loaddata("imgstrain.jld2")
  testfeatures,testlabels = loaddata("imgstest.jld2")

  # train MLP and predict classes
  Ws,bs = train(trainfeatures,trainlabels, [50,40,30,1])
  _,trainpredictions = predict(trainfeatures, Ws, bs)
  _,testpredictions = predict(testfeatures, Ws, bs)

  # show error
  trainerror = sum(trainpredictions.!=trainlabels)/length(trainlabels)
  testerror = sum(testpredictions.!=testlabels)/length(testlabels)
  println("Training Error Rate: $(round(100*trainerror,digits=2))%")
  println("Testing Error Rate: $(round(100*testerror,digits=2))%")

  return
end
PyPlot.close("all")
# problem2()
