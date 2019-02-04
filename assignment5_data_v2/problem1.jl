using Images
using PyPlot
using Clustering
using MultivariateStats
using Printf
using Random

include("Common.jl")

#---------------------------------------------------------
# Type aliases for arrays of images/features
#---------------------------------------------------------
const ImageList = Array{Array{Float64,2},1}
const FeatureList = Array{Array{Float64,2},1}


#---------------------------------------------------------
# Structure for storing datasets
#
# Fields:
#   images      images associated with this dataset
#   labels      corresponding labels
#   n           number of examples
#---------------------------------------------------------
struct Dataset
  images::Array{Array{Float64,2},1}
  labels::Array{Float64,1}
  n::Int
end

#---------------------------------------------------------
# Provides Dataset.length() method.
#---------------------------------------------------------
import Base.length
function length(x::Dataset)
  @assert length(x.images) == length(x.labels) == x.n "The length of the dataset is inconsistent."
  return x.n
end


#---------------------------------------------------------
# Structure for storing SIFT parameters.
#
# Fields:
#   fsize         filter size
#   sigma         standard deviation for filtering
#   threshold     SIFT threshold
#   boundary      number of boundary pixels to ignore
#---------------------------------------------------------
struct Parameters
  fsize::Int
  sigma::Float64
  threshold::Float64
  boundary::Int
end


#---------------------------------------------------------
# Helper: Concatenates two datasets.
#---------------------------------------------------------
function concat(d1::Dataset, d2::Dataset)
  return Dataset([d1.images; d2.images], [d1.labels; d2.labels], d1.n + d2.n)
end


#---------------------------------------------------------
# Helper: Create a train/test split of a dataset.
#---------------------------------------------------------
function traintestsplit(d::Dataset, p::Float64)
  ntrain = Int(floor(d.n * p))
  ntest = d.n - ntrain
  permuted_idx = randperm(d.n)

  train = Dataset(d.images[permuted_idx[1:ntrain]], d.labels[permuted_idx[1:ntrain]], ntrain)
  test = Dataset(d.images[permuted_idx[1+ntrain:end]], d.labels[permuted_idx[1+ntrain:end]], ntest)

  return train,test
end


#---------------------------------------------------------
# Create input data by separating planes and bikes randomly
# into two equally sized sets.
#
# Note: Use the Dataset type from above.
#
# OUTPUTS:
#   trainingset      Dataset of length 120, contraining bike and plane images
#   testingset       Dataset of length 120, contraining bike and plane images
#
#---------------------------------------------------------
function loadimages()
  nbikes = 106 # number of planes
  nplanes = 134 # number of bikes

  ### Your implementations for loading images here -------
  bikes = Dataset(ImageList[],zeros(Float64,0),nbikes)
  for i=1:nbikes
    file = string("bikes","/",lpad(string(i),3,"0"),".png")
    # println(file)
    img = Float64.(PyPlot.imread(file))
    push!(bikes.images,img)
    push!(bikes.labels,0)
  end
  planes = Dataset(ImageList[],zeros(Float64,0),nplanes)
  for i=1:nplanes
    file = string("planes","/",lpad(string(i),3,"0"),".png")
    # println(file)
    img = Float64.(PyPlot.imread(file))
    push!(planes.images,img)
    push!(planes.labels,1)
  end
  ### ----------------------------------------------------

  trainplanes, testplanes = traintestsplit(planes, 0.5)
  trainbikes, testbikes = traintestsplit(bikes, 0.5)

  trainingset = concat(trainbikes, trainplanes)
  testingset = concat(testbikes, testplanes)

  @assert length(trainingset) == 120
  @assert length(testingset) == 120
  # figure()
  #imshow(trainingset.images[1])#Has to be random? plane/bike
  return trainingset::Dataset, testingset::Dataset

end


#---------------------------------------------------------
# Extract SIFT features for all images
# For each image in the images::ImageList, first find interest points by applying the Harris corner detector.
# Then extract SIFT to compute the features at these points.
# Use params.sigma for the Harris corner detector and SIFT together.
#---------------------------------------------------------
function extractfeatures(images::ImageList, params::Parameters)
  features = Array{Float64,2}[]
  # println(typeof(features))
  for i = 1:length(images)
    py,px = Common.detect_interestpoints(images[i], params.fsize, params.threshold, params.sigma, params.boundary)
    points = hcat(px,py)
    push!(features, Common.sift(points,images[i],params.sigma))
  end
  # display(size(features[1],2))
  @assert length(features) == length(images)
  for i = 1:length(features)
    @assert size(features[i],1) == 128
  end
  return features::FeatureList
end


#---------------------------------------------------------
# Build a concatenated feature matrix from all given features
#---------------------------------------------------------
function concatenatefeatures(features::FeatureList)
  X = features[1]
  for i=2:size(features,1)
    X = hcat(X,features[i])
  end
  # display(X)


  @assert size(X,1) == 128
  return X::Array{Float64,2}
end

#---------------------------------------------------------
# Build a codebook for a given feature matrix by k-means clustering with K clusters
#---------------------------------------------------------
function computecodebook(X::Array{Float64,2},K::Int)
  R = kmeans(X, K; maxiter=200, display=:iter)
  codebook = R.centers
  # display(codebook)

  @assert size(codebook) == (size(X,1),K)
  return codebook::Array{Float64,2}
end


#---------------------------------------------------------
# Compute a histogram over the codebook for all given features
#---------------------------------------------------------
function computehistogram(features::FeatureList,codebook::Array{Float64,2},K::Int)
  H = zeros(Float64,K,size(features,1))
  # display(features[1][:,1])
  for i=1:size(features,1)
    for j=1:size(features[i],2)
        distances = (codebook.-features[i][:,j]).^2
        distances = sum(distances,dims=1)
        # display(argmin(distances)[2])
        H[argmin(distances)[2],i] = H[argmin(distances)[2],i]+1
    end
  end
  # display(H)
  H=H./sum(H,dims=1)
  # display(H)
  # display(sum(H,dims=1))
  # figure()
  # PyPlot.imshow(H)
  @assert size(H) == (K,length(features))
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Visualize a feature matrix by projection to the first
# two principal components. Points get colored according to class labels y.
#---------------------------------------------------------
function visualizefeatures(X::Array{Float64,2}, y)
  # mu = mean(X,dims=2)
  mu = sum(X,dims=2)/size(X,2)
  C = X.-mu
  PCA = MultivariateStats.pcasvd(C,mu[:,1],1;maxoutdim=2)
  M = MultivariateStats.transform(PCA,X)

  col = [0 for i in y]
  col = hcat(col,[1-i for i in y])
  col = hcat(col,[i for i in y])

  PyPlot.figure()
  PyPlot.scatter(M[1,:],M[2,:],c=col)



  return nothing::Nothing
end


# Problem 1: Bag of Words Model: Codebook

function problem1()
  # make results reproducable
  Random.seed!(0)

  # parameters
  params = Parameters(15, 1.4, 1e-7, 10)
  K = 5

  # load trainging and testing data
  traininginputs,testinginputs = loadimages()

  # extract features from images
  trainingfeatures = extractfeatures(traininginputs.images, params)
  testingfeatures = extractfeatures(testinginputs.images, params)

  # construct feature matrix from the training features
  X = concatenatefeatures(trainingfeatures)

  # write codebook
  codebook = computecodebook(X,K)

  # compute histogram
  traininghistogram = computehistogram(trainingfeatures,codebook,K)
  testinghistogram = computehistogram(testingfeatures,codebook,K)

  # # visualize training features
  visualizefeatures(traininghistogram, traininginputs.labels)

  return nothing::Nothing
end
# PyPlot.close("all")
problem1()
