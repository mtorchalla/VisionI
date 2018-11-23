using Images
using LinearAlgebra
using PyPlot
using Printf
using Statistics

# Load images from the yale_faces directory and return a MxN data matrix,
# where M is the number of pixels per face image and N is the number of images.
# Also return the dimensions of a single face image and the number of all face images
function loadfaces()
  faceDir = "yale_faces/"
  imfolders = readdir(faceDir)
  img_w = 84 #Assuming all img have same width and height
  img_h = 96
  facedim = [img_w; img_h]
  #Get all image files
  img_files = []
  for i=1:length(imfolders)
    dirname = string(faceDir, imfolders[i], "/")
    if isdir(dirname) #isFolder?
      append!(img_files, dirname.*readdir(dirname))
    end
  end
  n = length(img_files)
  #load Images from File List and append to Matrix
  data = zeros(img_w*img_h, length(img_files)) #allocate memory for data Matrix
  for i=1:length(img_files)
    img = open(img_files[i]) #load image
    readline(img) #Header
    imdata = read(img, img_w*img_h) #read bytes from image
    imdata = convert(Array{Float64,1}, imdata)
    data[:,i] = imdata
  end
  print("Data size:")
  println(size(data))
  return data::Array{Float64,2},facedim::Array{Int},n::Int
end

# Apply principal component analysis on the data matrix.
# Return the eigenvectors of covariance matrix of the data, the corresponding eigenvalues,
# the one-dimensional mean data matrix and a cumulated variance vector in increasing order.
function computepca(data::Array{Float64,2})
  M,N = size(data) #M: Rows; N: Columns of data Matrix
  #Calculate the mean:
  mu = zeros(1, N) ###################  Not Needed and if used should be zeros(M,1)
  mu = mean(data, dims=2)
  print("Sizeof mu:")
  println(size(mu))
  #Calculate deviations from mean:
  x = zeros(M,N) #################### Also not needed
  x = data .- mu
  ##Calculate covariance matrix:##
  #covar = zeros(M,M)
  #covar = 1/N * x*x'
  #SVD:
  U, S, V = svd(x, full=false)
  print("Sizeof U:")
  println(size(U))
  print("Sizeof V:")
  println(size(V))
  lambda = 1/N * S.^2
  # figure()
  # plot(lambda)
  # title("L")
  #Sort eigenvectors and eigenvalues:
  sortp = sortperm(S)
  # println(sortp)
  lambda = S[sortp] ####################### Why Only S and not 1/N *S.^2 again
  # U = U[:,sortp] #U is already sorted???
  #Calculate cumulated variance from cumsum(eigenvectors):
  cumvar = cumsum(lambda, dims=1)
  # print("Sizeof U:")
  # println(size(U))
  return U::Array{Float64,2},lambda::Array{Float64,1},mu::Array{Float64,2},cumvar::Array{Float64,1}
end

# Plot the cumulative variance of the principal components
function plotcumvar(cumvar::Array{Float64,1})
  PyPlot.plot(cumvar)
  title("Cumulative variance of principal components")
  return nothing::Nothing
end


# Compute required number of components to account for (at least) 75/99% of the variance
function computecomponents(cumvar::Array{Float64,1})
  total = cumvar[end]
  n75 = length(cumvar[cumvar .< 0.75*total])
  n99 = length(cumvar[cumvar .< 0.99*total])
  return n75::Int,n99::Int
end


# Display the mean face and the first 10 Eigenfaces in a single figure
function showeigenfaces(U::Array{Float64,2},mu::Array{Float64,2},facedim::Array{Int})
  #Meanface:
  figure()
  suptitle("Meanface and first 10 eigenfaces")
  PyPlot.subplot(432) #SubPlot: 2 Rows, 2 Columns, Index 1
  meanface = reshape(mu, facedim[1], facedim[2])
  imshow(meanface, cmap="gray")
  #First 10 Eigenfaces:
  # grid, frames, canvases = canvasgrid((2,5))
  # eigenfaces = zeros(facedim[1], facedim[2])
  # for i=1:10
  #   # figure()
  #   eigenfaces = eigenfaces .* reshape(U[:,i], facedim[1], facedim[2])
  #   # imshow(eigenface)
  #   # title(string("Eigenface ", i))
  # end
  PyPlot.subplot(434)
  imshow(reshape(U[:,1], facedim[1], facedim[2]), cmap="gray")
  PyPlot.subplot(435)
  imshow(reshape(U[:,2], facedim[1], facedim[2]), cmap="gray")
  PyPlot.subplot(436)
  imshow(reshape(U[:,3], facedim[1], facedim[2]), cmap="gray")
  PyPlot.subplot(436)
  imshow(reshape(U[:,4], facedim[1], facedim[2]), cmap="gray")
  PyPlot.subplot(437)
  imshow(reshape(U[:,5], facedim[1], facedim[2]), cmap="gray")
  PyPlot.subplot(438)
  imshow(reshape(U[:,6], facedim[1], facedim[2]), cmap="gray")
  PyPlot.subplot(439)
  imshow(reshape(U[:,7], facedim[1], facedim[2]), cmap="gray")
  PyPlot.subplot(4,3,10)
  imshow(reshape(U[:,8], facedim[1], facedim[2]), cmap="gray")
  PyPlot.subplot(4,3,11)
  imshow(reshape(U[:,9], facedim[1], facedim[2]), cmap="gray")
  PyPlot.subplot(4,3,12)
  imshow(reshape(U[:,10], facedim[1], facedim[2]), cmap="gray")
end


# Fetch a single face with given index out of the data matrix
function takeface(data::Array{Float64,2},facedim::Array{Int},n::Int)
  face = reshape(data[:,n], facedim[1], facedim[2])
  return face::Array{Float64,2}
end


# Project a given face into the low-dimensional space with a given number of principal
# components and reconstruct it afterwards
function computereconstruction(faceim::Array{Float64,2},U::Array{Float64,2},mu::Array{Float64,2},n::Int)
  # a = zeros(n,1)
  recon = mu
  for i=1:n
    a = Matrix(U[:,i]') * (vec(faceim)-mu) ###########################
    recon += a[1,1] * U[:,i] # a should only be a scalar right?
  end
  return recon::Array{Float64,2}
end

# Display all reconstructed faces in a single figure
function showreconstructedfaces(faceim, f5, f15, f50, f150)
  figure()
  suptitle("Reconstructed Faces for 5, 15, 50 and 150 components")
  PyPlot.subplot(221) #SubPlot: 2 Rows, 2 Columns, Index 1
  imshow(reshape(f5, 84, 96), cmap="gray")
  PyPlot.subplot(222)
  imshow(reshape(f15, 84, 96), cmap="gray")
  PyPlot.subplot(223)
  imshow(reshape(f50, 84, 96), cmap="gray")
  PyPlot.subplot(224)
  imshow(reshape(f150, 84, 96), cmap="gray")
  return nothing::Nothing
end

# Problem 2: Eigenfaces

function problem2()
  # load data
  data,facedim,N = loadfaces()

  # compute PCA
  U,lambda,mu,cumvar = computepca(data)

  # plot cumulative variance
  plotcumvar(cumvar)

  # compute necessary components for 75% / 99% variance coverage
  n75,n99 = computecomponents(cumvar)
  println(@sprintf("Necssary components for 75%% variance coverage: %i", n75))
  println(@sprintf("Necssary components for 99%% variance coverage: %i", n99))

  # plot mean face and first 10 Eigenfaces
  showeigenfaces(U,mu,facedim)

  # get a random face
  faceim = takeface(data,facedim,10)#rand(1:N))

  # reconstruct the face with 5, 15, 50, 150 principal components
  f5 = computereconstruction(faceim,U,mu,5)
  f15 = computereconstruction(faceim,U,mu,15)
  f50 = computereconstruction(faceim,U,mu,50)
  f150 = computereconstruction(faceim,U,mu,150)

  # display the reconstructed faces
  showreconstructedfaces(faceim, f5, f15, f50, f150)

  return
end

problem2()
